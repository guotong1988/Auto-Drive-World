"""行人走廊门控：Pilot-RL 训练 / 推理共用（探索门控、奖励）。"""

from __future__ import annotations

import math

from drive_env.dynamics import MAX_STEER_DEG
from drive_env.maps import TRACK_HALF_WIDTH, MapSpec

_HALF_LENGTH = 1.75
_HALF_WIDTH = 0.88


def _body_axes(heading_deg: float) -> tuple[float, float, float, float]:
  h = math.radians(heading_deg)
  fx, fy = -math.sin(h), math.cos(h)
  return fx, fy, fy, -fx


def _point_off_road(x: float, y: float, spec: MapSpec) -> float:
  best = float("inf")
  for x0, x1, y0, y1 in spec.road_rects:
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    best = min(best, math.hypot(dx, dy))
    if best == 0.0:
      return 0.0
  return best


def off_road_distance(x: float, y: float, heading_deg: float, spec: MapSpec) -> float:
  fx, fy, rx, ry = _body_axes(heading_deg)
  worst = 0.0
  for sl, sw in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
    cx = x + fx * _HALF_LENGTH * sl + rx * _HALF_WIDTH * sw
    cy = y + fy * _HALF_LENGTH * sl + ry * _HALF_WIDTH * sw
    worst = max(worst, _point_off_road(cx, cy, spec))
  return worst


def gate_lead_m(
  config: object | None = None,
  speed_kmh: float | None = None,
) -> tuple[float, float]:
  """探索门控的 (near, far) 米数。

  低速用配置下限；巡航按 TTC 拉远，让侧向绕行在碰到行人之前就有噪声样本。
  """
  near = float(getattr(config, "residual_gate_near", 16.0))
  far = float(getattr(config, "residual_gate_far", 32.0))
  if speed_kmh is None:
    return near, max(far, near + 1.0)
  speed_ms = max(float(speed_kmh), 0.0) / 3.6
  ttc_near = float(getattr(config, "residual_gate_ttc_near", 2.5))
  ttc_far = float(getattr(config, "residual_gate_ttc_far", 5.0))
  near = max(near, speed_ms * ttc_near)
  far = max(far, speed_ms * ttc_far, near + 1.0)
  return near, far


def residual_gate(
  ped_dist: float,
  config: object | None = None,
  *,
  near: float | None = None,
  far: float | None = None,
  speed_kmh: float | None = None,
) -> float:
  """靠近行人时为 1（开探索），前方路况清空时为 0。"""
  if near is None or far is None:
    near_m, far_m = gate_lead_m(config, speed_kmh)
    if near is None:
      near = near_m
    if far is None:
      far = far_m
  near_f = float(near)
  far_f = float(far)
  if ped_dist >= far_f:
    return 0.0
  if ped_dist <= near_f:
    return 1.0
  return float((far_f - ped_dist) / max(far_f - near_f, 1e-6))


def ped_body_frame(
  x: float,
  y: float,
  heading_deg: float,
  ped_xy: tuple[float, float] | None,
) -> tuple[float, float, float]:
  """返回行人在车体坐标系中的 (距离, 前方, 右侧)。"""
  if ped_xy is None:
    # fwd 必须 < 0.5，否则 residual_gate_from_ped 会把「没人」当成前方 1m。
    return 1e9, -1.0, 0.0
  fx, fy, rx, ry = _body_axes(heading_deg)
  dx, dy = ped_xy[0] - x, ped_xy[1] - y
  return math.hypot(dx, dy), dx * fx + dy * fy, dx * rx + dy * ry


def threat_pedestrian(
  x: float,
  y: float,
  heading_deg: float,
  positions: list[tuple[float, float]],
  config: object | None = None,
) -> tuple[float, float] | None:
  """行驶走廊内最近的前方行人（忽略路边 / 身后）。"""
  lane = float(getattr(config, "residual_lane_m", 5.0))
  best: tuple[float, float] | None = None
  best_key = float("inf")
  for px, py in positions:
    dist, fwd, right = ped_body_frame(x, y, heading_deg, (px, py))
    if fwd < 0.5 or abs(right) > lane:
      continue
    key = dist + 0.25 * abs(right)
    if key < best_key:
      best_key = key
      best = (px, py)
  return best


def residual_gate_from_ped(
  x: float,
  y: float,
  heading_deg: float,
  ped_xy: tuple[float, float] | None,
  config: object | None = None,
  speed_kmh: float | None = None,
) -> float:
  if ped_xy is None:
    return 0.0
  _dist, fwd, right = ped_body_frame(x, y, heading_deg, ped_xy)
  lane = float(getattr(config, "residual_lane_m", 5.0))
  if fwd < 0.5 or abs(right) > lane:
    return 0.0
  # 用沿轨迹距离，与 TTC 一致；巡航时门控比欧氏 16–32 m 更早打开。
  return residual_gate(fwd, config, speed_kmh=speed_kmh)


# hit_vehicle：车半径 1.6 m + 行人 1.35 m = 2.95 m。绕行目标必须大于这个圆。
_HIT_CAR_M = 1.6
_HIT_PED_M = 1.35
HIT_NEED_M = _HIT_CAR_M + _HIT_PED_M
_PASS_LAT_M = 4.0
_DODGE_BEHIND_M = -2.0
_GATE_ON = 0.15
_HOLD_FRAMES = 48
# 车体右侧速度超过此值视为横穿，绕行侧改走其行进方向后方。
_CROSS_V_RIGHT = 0.35


def _path_body_axes(path: object, station: float) -> tuple[float, float, float, float]:
  """参考路径切向处的 (fx, fy, rx, ry)，与车体轴同约定。"""
  heading_at = getattr(path, "heading_at", None)
  if heading_at is None:
    raise TypeError("path must provide heading_at(station)")
  h = math.radians(float(heading_at(station)))
  fx, fy = -math.sin(h), math.cos(h)
  return fx, fy, fy, -fx


def _corridor_peds(
  x: float,
  y: float,
  heading_deg: float,
  positions: list[tuple[float, float]],
  config: object | None,
  speed_kmh: float,
  *,
  behind_m: float = 0.5,
  velocities: list[tuple[float, float]] | None = None,
  path: object | None = None,
  station: float | None = None,
  vehicle_cte: float = 0.0,
) -> list[tuple[float, float, float, tuple[float, float], float]]:
  """走廊内行人：(fwd, right, dist, xy, v_right)。

  有参考路径时沿路径弧长找人（弯道 left/right 也能看见拐角上的人）；
  否则沿车头朝向。v_right 是路径/车体右侧速度。
  """
  lane = float(getattr(config, "residual_lane_m", 5.0))
  _near, far = gate_lead_m(config, speed_kmh)
  use_path = path is not None and station is not None and hasattr(path, "project")
  if use_path:
    assert path is not None and station is not None
    _fx, _fy, rx, ry = _path_body_axes(path, float(station))
  else:
    _fx, _fy, rx, ry = _body_axes(heading_deg)
  found: list[tuple[float, float, float, tuple[float, float], float]] = []
  for i, (px, py) in enumerate(positions):
    if use_path:
      assert path is not None and station is not None
      ped_s, ped_cte = path.project(px, py)
      fwd = float(ped_s) - float(station)
      right = float(vehicle_cte) - float(ped_cte)
      dist = math.hypot(px - x, py - y)
    else:
      dist, fwd, right = ped_body_frame(x, y, heading_deg, (px, py))
    if fwd < behind_m or fwd > far or abs(right) > lane:
      continue
    vx, vy = (0.0, 0.0)
    if velocities is not None and i < len(velocities):
      vx, vy = velocities[i]
    v_right = vx * rx + vy * ry
    found.append((fwd, right, dist, (px, py), v_right))
  return found


class RuleDodge:
  """规则绕行（有状态）：锁过侧，避免贴身时跟路把车拉回人身上。"""

  def __init__(self) -> None:
    self._go_left: bool | None = None
    self._hold = 0
    self.last: dict[str, float | bool | int | str] | None = None

  def reset(self) -> None:
    self._go_left = None
    self._hold = 0
    self.last = None

  def _tick_hold(self) -> None:
    if self._hold > 0:
      self._hold -= 1
    else:
      self._go_left = None

  def _evaluate(
    self,
    x: float,
    y: float,
    heading_deg: float,
    positions: list[tuple[float, float]],
    *,
    cte: float,
    speed_kmh: float,
    config: object | None,
    velocities: list[tuple[float, float]] | None,
    path: object | None,
    station: float | None,
  ) -> dict[str, float | bool | int] | None:
    """走廊决策。None = 跟中线；返回值含 lat_err（正 = 车再往左）。"""
    found = _corridor_peds(
      x,
      y,
      heading_deg,
      positions,
      config,
      speed_kmh,
      behind_m=_DODGE_BEHIND_M,
      velocities=velocities,
      path=path,
      station=station,
      vehicle_cte=cte,
    )
    if not found:
      self._tick_hold()
      self.last = None
      return None

    ahead = [fwd for fwd, _right, _dist, _xy, _vr in found if fwd > 0.0]
    gate = residual_gate(min(ahead), config, speed_kmh=speed_kmh) if ahead else 1.0
    if gate < _GATE_ON and min(fwd for fwd, _r, _d, _xy, _vr in found) > 0.5:
      self._tick_hold()
      self.last = {
        "fwd": float(min(ahead) if ahead else 0.0),
        "right": float(found[0][1]),
        "gate": float(gate),
        "n_peds": len(found),
        "dodging": False,
      }
      return None

    self._hold = _HOLD_FRAMES
    nearest = min(found, key=lambda p: p[2] + 0.25 * abs(p[1]))
    fwd, right, dist, _xy, v_right = nearest
    leftmost = min(p[1] for p in found)
    rightmost = max(p[1] for p in found)
    pass_lat = _PASS_LAT_M
    shift_left = pass_lat - leftmost
    shift_right = rightmost + pass_lat

    half = max(TRACK_HALF_WIDTH - 1.5, 3.0)
    left_room = half - float(cte)
    right_room = half + float(cte)
    go_left = self._pick_side(
      left_room,
      right_room,
      shift_left,
      shift_right,
      float(cte),
      v_right=float(v_right),
    )

    # 已锁过侧时，对面且已留出间隙的人不进目标，避免打方向穿过去。
    opp = HIT_NEED_M * 0.95
    if go_left:
      tracked = [p for p in found if p[1] > -opp]
    else:
      tracked = [p for p in found if p[1] < opp]
    if tracked:
      ref_right = min(p[1] for p in tracked) if go_left else max(p[1] for p in tracked)
    else:
      ref_right = right
    if go_left:
      target_right = max(pass_lat, ref_right)
    else:
      target_right = min(-pass_lat, ref_right)
    already_clear = (not tracked) or (
      (go_left and ref_right >= pass_lat * 0.92)
      or ((not go_left) and ref_right <= -pass_lat * 0.92)
    )
    lat_err = 0.0 if already_clear else (target_right - ref_right)
    close = (min(ahead) if ahead else abs(fwd)) < 10.0
    decision: dict[str, float | bool | int] = {
      "fwd": float(fwd),
      "right": float(right),
      "dist": float(dist),
      "gate": float(gate),
      "go_left": go_left,
      "lat_err": float(lat_err),
      "cte": float(cte),
      "speed": float(speed_kmh),
      "n_peds": len(found),
      "clear": already_clear,
      "close": close,
      "dodging": True,
      "v_right": float(v_right),
    }
    self.last = dict(decision)
    return decision

  def plan_cte(
    self,
    x: float,
    y: float,
    heading_deg: float,
    positions: list[tuple[float, float]],
    *,
    cte: float = 0.0,
    speed_kmh: float = 0.0,
    config: object | None = None,
    velocities: list[tuple[float, float]] | None = None,
    path: object | None = None,
    station: float | None = None,
  ) -> tuple[float | None, bool]:
    """沿参考路径的目标横向位置（正值在左）。None 表示跟中线。

    弯道用路径弧长当「前方」，left/right 也能绕行人；贴身时锁住当前横向，
    避免纯追踪把车拉回路中线撞回去。
    """
    ev = self._evaluate(
      x,
      y,
      heading_deg,
      positions,
      cte=cte,
      speed_kmh=speed_kmh,
      config=config,
      velocities=velocities,
      path=path,
      station=station,
    )
    if ev is None:
      return None, False
    half = max(TRACK_HALF_WIDTH - 1.5, 3.0)
    if bool(ev["clear"]):
      if bool(ev["close"]):
        held = max(-half, min(half, float(cte)))
        if self.last is not None:
          self.last["target_cte"] = float(held)
        return held, True
      return None, True
    target = float(cte) + float(ev["lat_err"])
    target = max(-half, min(half, target))
    if self.last is not None:
      self.last["target_cte"] = float(target)
    return target, True

  def apply(
    self,
    x: float,
    y: float,
    heading_deg: float,
    positions: list[tuple[float, float]],
    expert_steer: float,
    *,
    cte: float = 0.0,
    speed_kmh: float = 0.0,
    config: object | None = None,
    velocities: list[tuple[float, float]] | None = None,
    path: object | None = None,
    station: float | None = None,
  ) -> tuple[float, bool]:
    expert = float(max(-1.0, min(1.0, expert_steer)))
    ev = self._evaluate(
      x,
      y,
      heading_deg,
      positions,
      cte=cte,
      speed_kmh=speed_kmh,
      config=config,
      velocities=velocities,
      path=path,
      station=station,
    )
    if ev is None:
      return expert, False

    already_clear = bool(ev["clear"])
    lat_err = float(ev["lat_err"])
    right = float(ev["right"])
    fwd = float(ev["fwd"])
    gate = float(ev["gate"])
    dodge_cmd = 0.0
    if already_clear:
      # 侧向已经够：近处不要跟回路中线撞回去；远处仍跟路，避免在草地上抱死方向盘。
      toward_ped = (right > 0.4 and expert < 0.0) or (right < -0.4 and expert > 0.0)
      if bool(ev["close"]) and toward_ped:
        steer = 0.0
      else:
        steer = expert
    else:
      look = max(abs(fwd), 2.5)
      heading_err_deg = math.degrees(math.atan2(lat_err, look))
      dodge_cmd = heading_err_deg / max(float(MAX_STEER_DEG), 1.0)
      dodge_cmd = max(-1.0, min(1.0, dodge_cmd))
      # 远距离 heading 已经变小，不再用 gate 把转向再削一层。
      steer = expert * (1.0 - gate) + dodge_cmd
    steer = float(max(-1.0, min(1.0, steer)))

    if self.last is not None:
      self.last["expert"] = float(expert)
      self.last["dodge"] = float(dodge_cmd)
      self.last["steer"] = float(steer)
    return steer, True

  def _pick_side(
    self,
    left_room: float,
    right_room: float,
    shift_left: float,
    shift_right: float,
    cte: float,
    v_right: float = 0.0,
  ) -> bool:
    if self._go_left is not None:
      return self._go_left

    if left_room < 2.2 and right_room >= left_room:
      go_left = False
    elif right_room < 2.2 and left_room >= right_room:
      go_left = True
    elif abs(v_right) >= _CROSS_V_RIGHT:
      # 横穿：向右走则从左侧（其后方）过，不要插到行进方向前面。
      go_left = v_right > 0.0
    elif abs(shift_left - shift_right) > 0.45:
      go_left = shift_left < shift_right
    else:
      # 人在正前方：沿已有横向偏差继续往外，不要折回路中（人就在路上）。
      go_left = cte >= 0.0
    self._go_left = go_left
    return go_left

  def format_last(self) -> str:
    d = self.last
    if not d:
      return "[dodge] (no threat)"
    if not d.get("dodging", True):
      return (
        f"[dodge] far  fwd={float(d['fwd']):.1f}m right={float(d['right']):+.1f}m "
        f"gate={float(d['gate']):.2f} peds={int(d['n_peds'])}"
      )
    side = "L" if d["go_left"] else "R"
    hold = " hold" if d["clear"] else ""
    v_right = float(d.get("v_right", 0.0))
    behind = " behind" if abs(v_right) >= _CROSS_V_RIGHT else ""
    turn = " turn" if d.get("turning") else ""
    target = d.get("target_cte")
    cte_star = f" cte*={float(target):+.1f}" if target is not None else ""
    return (
      f"[dodge] {side}{hold}{behind}{turn} fwd={float(d['fwd']):.1f}m "
      f"right={float(d['right']):+.1f}m vR={v_right:+.1f} "
      f"lat_err={float(d['lat_err']):+.1f}m{cte_star} "
      f"gate={float(d['gate']):.2f} | expert={float(d.get('expert', 0.0)):+.2f} "
      f"dodge={float(d.get('dodge', 0.0)):+.2f} out={float(d.get('steer', 0.0)):+.2f} "
      f"| cte={float(d['cte']):+.1f}m spd={float(d['speed']):.0f} "
      f"peds={int(d['n_peds'])}"
    )


def rule_dodge_steer(
  x: float,
  y: float,
  heading_deg: float,
  positions: list[tuple[float, float]],
  expert_steer: float,
  *,
  cte: float = 0.0,
  speed_kmh: float = 0.0,
  config: object | None = None,
  state: RuleDodge | None = None,
  velocities: list[tuple[float, float]] | None = None,
) -> tuple[float, bool]:
  """规则绕行：走廊有人时只改转向，不改油门。

  横穿行人从行进方向后方绕；沿路行人仍走空隙更大的一侧。
  目标侧向大于撞人圆；贴身前跟路权重要降下来，避免打回人身上。
  传入 ``state`` 可锁过侧。返回 (steer, 是否在绕)。
  采集走 ``RuleExpert.predict(..., dodge=)``：沿参考路径侧移前视点，
  left/right 弯道也能绕，不会把弯道曲率冲掉。
  """
  dodge = state if state is not None else RuleDodge()
  return dodge.apply(
    x,
    y,
    heading_deg,
    positions,
    expert_steer,
    cte=cte,
    speed_kmh=speed_kmh,
    config=config,
    velocities=velocities,
  )

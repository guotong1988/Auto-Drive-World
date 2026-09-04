"""脚本化行人：在路缘等候后过马路。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from panda3d.core import CardMaker, NodePath

from drive_env.maps import (
  TRACK_HALF_WIDTH,
  MapSpec,
  crossing_sites,
  road_axes_at,
)
from drive_env.terrain import height_at

# 步行速度大致对应成人行人（m/s）。
_WALK_SPEED_MIN = 1.1
_WALK_SPEED_MAX = 1.7
_HURRY_SPEED_MAX = 2.2
_WAYPOINT_EPS = 0.45
_SPAWN_CLEAR_M = 28.0
_HIT_RADIUS = 1.35
_CURB_MARGIN = 2.2
_WAIT_MIN = 1.2
_WAIT_MAX = 3.8

# 衣着 / 肤色配色——在深色沥青上可辨认。
_BODY_COLORS = (
  (0.85, 0.35, 0.2, 1.0),
  (0.2, 0.55, 0.75, 1.0),
  (0.55, 0.7, 0.25, 1.0),
  (0.75, 0.45, 0.7, 1.0),
  (0.9, 0.7, 0.2, 1.0),
)
_SKIN = (0.92, 0.74, 0.58, 1.0)


def _heading_from_dir(dx: float, dy: float) -> float:
  """Panda 航向角（度）：0 朝向 +Y，正值左转。"""
  return math.degrees(math.atan2(-dx, dy))


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
  return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _offset_point(
  a: tuple[float, float],
  b: tuple[float, float],
  t: float,
  lateral: float,
) -> tuple[float, float]:
  """沿线段 a→b 在比例 t 处的点，再按 lateral 右手法则侧移。"""
  x, y = _lerp(a, b, t)
  dx, dy = b[0] - a[0], b[1] - a[1]
  length = math.hypot(dx, dy)
  if length < 1e-6:
    return x, y
  fx, fy = dx / length, dy / length
  # 前进方向的右侧：(fy, -fx)
  return x + fy * lateral, y - fx * lateral


@dataclass
class Pedestrian:
  """一个沿循环路点行走的行人。"""

  root: NodePath
  waypoints: list[tuple[float, float]]
  speed: float
  wp_index: int = 0
  hit_radius: float = _HIT_RADIUS
  wait_s: float = 0.0
  pause_s: float = 0.0
  pause_at_waypoints: bool = False

  @property
  def x(self) -> float:
    return float(self.root.getX())

  @property
  def y(self) -> float:
    return float(self.root.getY())

  @property
  def velocity(self) -> tuple[float, float]:
    """即将行走的世界系速度 (vx, vy)。路缘等候时仍给出过街意图方向。"""
    if len(self.waypoints) < 2:
      return 0.0, 0.0
    tx, ty = self.waypoints[self.wp_index]
    dx, dy = tx - self.x, ty - self.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
      return 0.0, 0.0
    inv = self.speed / dist
    return dx * inv, dy * inv

  def update(self, dt: float) -> None:
    if len(self.waypoints) < 2:
      return
    if self.wait_s > 0.0:
      self.wait_s -= dt
      return

    tx, ty = self.waypoints[self.wp_index]
    dx, dy = tx - self.x, ty - self.y
    dist = math.hypot(dx, dy)
    if dist < _WAYPOINT_EPS:
      self.wp_index = (self.wp_index + 1) % len(self.waypoints)
      if self.pause_at_waypoints and self.pause_s > 0.0:
        self.wait_s = self.pause_s
        return
      tx, ty = self.waypoints[self.wp_index]
      dx, dy = tx - self.x, ty - self.y
      dist = math.hypot(dx, dy)
      if dist < 1e-6:
        return

    step = min(self.speed * dt, dist)
    nx = self.x + dx / dist * step
    ny = self.y + dy / dist * step
    self.root.setPos(nx, ny, height_at(nx, ny))
    self.root.setH(_heading_from_dir(dx, dy))


class PedestrianCrowd:
  """为地图生成并更新行人。"""

  def __init__(
    self,
    parent: NodePath,
    map_spec: MapSpec,
    count: int | None = None,
    seed: int | None = None,
  ):
    self.map_spec = map_spec
    self._parent = parent
    self._count = count
    self._seed = seed
    self._rng = random.Random(seed)
    self.root = parent.attachNewNode("pedestrians")
    self.pedestrians: list[Pedestrian] = []
    n = count if count is not None else self._default_count(map_spec)
    self._spawn(n)

  @staticmethod
  def _default_count(map_spec: MapSpec) -> int:
    n_sites = len(crossing_sites(map_spec))
    return max(10, min(24, n_sites + 4))

  def _spawn(self, count: int) -> None:
    sx, sy = self.map_spec.spawn_xy
    plans: list[tuple[list[tuple[float, float]], float, bool]] = []

    for path, hurry in self._crossing_plans():
      if self._path_clear_of_spawn(path, sx, sy):
        lo, hi = (_WALK_SPEED_MIN, _HURRY_SPEED_MAX) if hurry else (
          _WALK_SPEED_MIN,
          _WALK_SPEED_MAX,
        )
        plans.append((path, self._rng.uniform(lo, hi), True))

    for path in self._jaywalk_paths():
      if self._path_clear_of_spawn(path, sx, sy):
        plans.append(
          (path, self._rng.uniform(_WALK_SPEED_MIN, _HURRY_SPEED_MAX), True)
        )

    # 再加几个路边行人，避免街上只有斑马线过街。
    edge_paths = self._edge_walk_paths()
    self._rng.shuffle(edge_paths)
    roadside_budget = max(2, count // 5)
    added_roadside = 0
    for path in edge_paths:
      if added_roadside >= roadside_budget:
        break
      if self._path_clear_of_spawn(path, sx, sy):
        plans.append(
          (path, self._rng.uniform(_WALK_SPEED_MIN, _WALK_SPEED_MAX), False)
        )
        added_roadside += 1

    if not plans:
      return

    self._rng.shuffle(plans)
    for i in range(count):
      path, speed, pauses = plans[i % len(plans)]
      start_i = self._rng.randrange(len(path))
      rotated = path[start_i:] + path[:start_i]
      ped = self._make_pedestrian(i, rotated, speed, pause_at_waypoints=pauses)
      if pauses and self._rng.random() < 0.45 and len(rotated) >= 2:
        self._place_mid_crossing(ped)
      self.pedestrians.append(ped)

  def _path_clear_of_spawn(
    self, path: list[tuple[float, float]], sx: float, sy: float
  ) -> bool:
    return all(math.hypot(x - sx, y - sy) >= _SPAWN_CLEAR_M for x, y in path)

  def _crossing_plans(self) -> list[tuple[list[tuple[float, float]], bool]]:
    """斑马线处的垂直过街；每组 1–3 人。"""
    plans: list[tuple[list[tuple[float, float]], bool]] = []
    for cx, cy in crossing_sites(self.map_spec):
      has_ns, has_ew = road_axes_at(self.map_spec, cx, cy)
      axes: list[str] = []
      if has_ns:
        axes.append("ew")
      if has_ew:
        axes.append("ns")
      for axis in axes:
        n_group = 1
        roll = self._rng.random()
        if roll < 0.20:
          n_group = 3
        elif roll < 0.65:
          n_group = 2
        for _ in range(n_group):
          along = self._rng.uniform(-2.4, 2.4)
          hurry = self._rng.random() < 0.18
          plans.append((self._cross_path(cx, cy, axis, along), hurry))
    return plans

  def _jaywalk_paths(self) -> list[list[tuple[float, float]]]:
    """斑马线之间长直路上的无标线过街。"""
    sites = crossing_sites(self.map_spec)
    sx, sy = self.map_spec.spawn_xy
    paths: list[list[tuple[float, float]]] = []
    nodes = self.map_spec.nav_nodes
    for a_name, b_name in self.map_spec.nav_edges:
      a, b = nodes[a_name], nodes[b_name]
      dx, dy = b[0] - a[0], b[1] - a[1]
      length = math.hypot(dx, dy)
      if length < 95.0:
        continue
      t = self._rng.uniform(0.28, 0.72)
      px, py = a[0] + dx * t, a[1] + dy * t
      if math.hypot(px - sx, py - sy) < _SPAWN_CLEAR_M:
        continue
      if any(math.hypot(px - zx, py - zy) < 30.0 for zx, zy in sites):
        continue
      has_ns, has_ew = road_axes_at(self.map_spec, px, py)
      if has_ns and not has_ew:
        along = self._rng.uniform(-1.8, 1.8)
        paths.append(self._cross_path(px, py, "ew", along))
      elif has_ew and not has_ns:
        along = self._rng.uniform(-1.8, 1.8)
        paths.append(self._cross_path(px, py, "ns", along))
    self._rng.shuffle(paths)
    return paths[:4]

  def _cross_path(
    self,
    cx: float,
    cy: float,
    axis: str,
    along: float = 0.0,
  ) -> list[tuple[float, float]]:
    """往返穿过车行道；端点落在草地上。"""
    curb = TRACK_HALF_WIDTH + _CURB_MARGIN
    if axis == "ew":
      return [(cx - curb, cy + along), (cx + curb, cy + along)]
    return [(cx + along, cy - curb), (cx + along, cy + curb)]

  def _place_mid_crossing(self, ped: Pedestrian) -> None:
    """直接把人放到沥青上，让车在车道里遇到行人。"""
    a = ped.waypoints[0]
    b = ped.waypoints[1]
    t = self._rng.uniform(0.22, 0.78)
    x, y = _lerp(a, b, t)
    ped.root.setPos(x, y, height_at(x, y))
    ped.root.setH(_heading_from_dir(b[0] - a[0], b[1] - a[1]))
    ped.wp_index = 1
    ped.wait_s = 0.0

  def _edge_walk_paths(self) -> list[list[tuple[float, float]]]:
    nodes = self.map_spec.nav_nodes
    paths: list[list[tuple[float, float]]] = []
    lateral_choices = (
      TRACK_HALF_WIDTH * 0.55,
      -TRACK_HALF_WIDTH * 0.55,
      TRACK_HALF_WIDTH * 0.75,
      -TRACK_HALF_WIDTH * 0.75,
    )

    for a_name, b_name in self.map_spec.nav_edges:
      a, b = nodes[a_name], nodes[b_name]
      if math.hypot(b[0] - a[0], b[1] - a[1]) < 25.0:
        continue
      lateral = self._rng.choice(lateral_choices)
      # 采样几个点，使运动保持在偏移走廊上。
      pts = [_offset_point(a, b, t, lateral) for t in (0.08, 0.35, 0.65, 0.92)]
      # 往返巡逻该路段。
      paths.append(pts + list(reversed(pts[1:-1])))
    return paths

  def _make_pedestrian(
    self,
    index: int,
    waypoints: list[tuple[float, float]],
    speed: float,
    pause_at_waypoints: bool = False,
  ) -> Pedestrian:
    x0, y0 = waypoints[0]
    root = self.root.attachNewNode(f"pedestrian_{index}")
    root.setPos(x0, y0, height_at(x0, y0))
    if len(waypoints) >= 2:
      root.setH(_heading_from_dir(waypoints[1][0] - x0, waypoints[1][1] - y0))

    body_color = _BODY_COLORS[index % len(_BODY_COLORS)]
    self._build_visual(root, body_color)
    pause_s = self._rng.uniform(_WAIT_MIN, _WAIT_MAX) if pause_at_waypoints else 0.0
    wait_s = pause_s * self._rng.random() if pause_at_waypoints else 0.0
    return Pedestrian(
      root=root,
      waypoints=waypoints,
      speed=speed,
      wp_index=1 if len(waypoints) >= 2 else 0,
      pause_s=pause_s,
      wait_s=wait_s,
      pause_at_waypoints=pause_at_waypoints,
    )

  @staticmethod
  def _build_visual(root: NodePath, body_color: tuple[float, float, float, float]) -> None:
    """简单直立交叉广告牌——任意偏航角都可辨认。"""
    # 身体（两张交叉卡片）。
    for name, yaw in (("body_a", 0.0), ("body_b", 90.0)):
      cm = CardMaker(name)
      cm.setFrame(-0.28, 0.28, 0.0, 1.35)
      card = root.attachNewNode(cm.generate())
      card.setH(yaw)
      card.setZ(0.05)
      card.setColor(*body_color)
      card.setTwoSided(True)
      card.setLightOff()

    # 头部。
    for name, yaw in (("head_a", 0.0), ("head_b", 90.0)):
      cm = CardMaker(name)
      cm.setFrame(-0.2, 0.2, 0.0, 0.4)
      card = root.attachNewNode(cm.generate())
      card.setH(yaw)
      card.setZ(1.4)
      card.setColor(*_SKIN)
      card.setTwoSided(True)
      card.setLightOff()

  def update(self, dt: float) -> None:
    for ped in self.pedestrians:
      ped.update(dt)

  @property
  def positions(self) -> list[tuple[float, float]]:
    return [(p.x, p.y) for p in self.pedestrians]

  @property
  def velocities(self) -> list[tuple[float, float]]:
    return [p.velocity for p in self.pedestrians]

  def nearest_distance(self, x: float, y: float) -> float:
    if not self.pedestrians:
      return float("inf")
    return min(math.hypot(p.x - x, p.y - y) for p in self.pedestrians)

  def hit_vehicle(self, x: float, y: float, radius: float | None = None) -> bool:
    r = self.hit_radius_default if radius is None else radius
    return any(math.hypot(p.x - x, p.y - y) < (r + p.hit_radius) for p in self.pedestrians)

  @property
  def hit_radius_default(self) -> float:
    # 约半个车身长度，用作车辆接触半径。
    return 1.6

  def reset(self, count: int | None = None) -> None:
    """清空并重新生成（例如按 R 重生时）。"""
    for ped in self.pedestrians:
      ped.root.removeNode()
    self.pedestrians.clear()
    self.root.removeNode()
    self._rng = random.Random(self._seed)
    self.root = self._parent.attachNewNode("pedestrians")
    n = (
      count
      if count is not None
      else self._count
      if self._count is not None
      else self._default_count(self.map_spec)
    )
    self._spawn(n)

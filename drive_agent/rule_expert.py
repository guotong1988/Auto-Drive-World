"""基于规则的道路跟随器：规划到终点、沿车道行驶、在旗帜处停车。

顶层规划在导航图上选出节点序列。默认（``route_policy="random"``）每次
reset 从到达终点的简单路径中抽样，使回合（及采集数据）不锁死在同一走廊。
局部控制仍用「直线-圆弧-直线」参考：沿进路口中心线行驶，直到能塞进路口
的圆弧切点，走完圆弧再接上出口车道。转向用纯追踪；速度按圆弧曲率封顶，
以便能稳住轨迹。
"""

from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict

from drive_agent.commands import (
  COMMAND_TO_ID,
  TURN_COMMAND_DEG,
  TURN_COMMAND_HOLD_DEG,
  TURN_COMMAND_PREVIEW_M,
)
from drive_env.dynamics import MAX_STEER_DEG, WHEELBASE
from drive_env.maps import TRACK_HALF_WIDTH, MapSpec, get_map

# 转弯圆弧相对路缘内收这么远（车半宽 + 跟踪余量）。
LANE_MARGIN = 3.0

KMH_PER_MS = 3.6

# 打散顶层规划时，最多保留这么多条简单路径。
_MAX_ROUTE_CANDIDATES = 32
# 允许比最短路多绕这么多跳。
_MAX_EXTRA_HOPS = 4


def _build_adjacency(
  edges: list[tuple[str, str]],
) -> dict[str, list[str]]:
  adj: dict[str, list[str]] = defaultdict(list)
  for a, b in edges:
    adj[a].append(b)
    adj[b].append(a)
  return dict(adj)


def _heading_deg(dx: float, dy: float) -> float:
  """方向向量的 Panda 航向：0 = +Y，正值左转。"""
  return math.degrees(math.atan2(-dx, dy))


def _normalize_angle_deg(angle: float) -> float:
  while angle > 180.0:
    angle -= 360.0
  while angle < -180.0:
    angle += 360.0
  return angle


class _RefPath:
  """折线参考路径，按弧长查点，供纯追踪使用。"""

  def __init__(
    self,
    points: list[tuple[float, float]],
    arc_span: tuple[float, float] | None,
    radius: float,
    handover: float,
  ):
    self.points = points
    self.arc_span = arc_span
    self.radius = radius
    # 超过该弧长后路口已在身后，图规划可以前进。
    self.handover = handover
    self.cum = [0.0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
      self.cum.append(self.cum[-1] + math.hypot(x1 - x0, y1 - y0))

  @property
  def length(self) -> float:
    return self.cum[-1]

  def project(self, x: float, y: float) -> tuple[float, float]:
    """路径上最近点：(弧长, 横向误差，正值在左)。"""
    best_d2 = float("inf")
    best = (0.0, 0.0)
    for i in range(len(self.points) - 1):
      px, py = self.points[i]
      qx, qy = self.points[i + 1]
      ex, ey = qx - px, qy - py
      elen = math.hypot(ex, ey)
      if elen < 1e-9:
        continue
      ux, uy = ex / elen, ey / elen
      t = min(elen, max(0.0, (x - px) * ux + (y - py) * uy))
      cx, cy = px + ux * t, py + uy * t
      d2 = (x - cx) ** 2 + (y - cy) ** 2
      if d2 < best_d2:
        best_d2 = d2
        best = (self.cum[i] + t, (y - py) * ux - (x - px) * uy)
    return best

  def point_at(self, s: float) -> tuple[float, float]:
    """弧长 s 处的路径点；超出终点时沿末段外推。"""
    if s <= 0.0:
      return self.points[0]
    if s >= self.length:
      (px, py), (qx, qy) = self.points[-2], self.points[-1]
      elen = math.hypot(qx - px, qy - py) or 1.0
      extra = s - self.length
      return (qx + (qx - px) / elen * extra, qy + (qy - py) / elen * extra)
    for i in range(len(self.points) - 1):
      if self.cum[i + 1] < s:
        continue
      px, py = self.points[i]
      qx, qy = self.points[i + 1]
      seg = self.cum[i + 1] - self.cum[i]
      t = (s - self.cum[i]) / (seg or 1.0)
      return (px + (qx - px) * t, py + (qy - py) * t)
    return self.points[-1]

  def on_arc(self, s: float) -> bool:
    if self.arc_span is None:
      return False
    return self.arc_span[0] - 2.0 <= s <= self.arc_span[1] + 2.0

  def distance_to_arc(self, s: float) -> float:
    """距圆弧起点还剩多少米（已进入或越过则为 0）。"""
    if self.arc_span is None:
      return float("inf")
    return max(0.0, self.arc_span[0] - s)

  def heading_at(self, s: float) -> float:
    """弧长 s 处切向航向（Panda：0 = +Y，正值左转）。"""
    delta = 0.75
    p0 = self.point_at(s)
    p1 = self.point_at(s + delta)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if math.hypot(dx, dy) < 1e-6:
      p1 = self.point_at(s - delta)
      dx, dy = p0[0] - p1[0], p0[1] - p1[1]
    return _heading_deg(dx, dy)

  def offset_point(self, s: float, left_m: float) -> tuple[float, float]:
    """弧长 s 处沿路径法向左移 left_m（正值在左，与 project 的横向误差同号）。"""
    x, y = self.point_at(s)
    h = math.radians(self.heading_at(s))
    fx, fy = -math.sin(h), math.cos(h)
    return x + (-fy) * left_m, y + fx * left_m


class RuleExpert:
  """在道路图上沿规划路径驶向终点旗帜。"""

  def __init__(
    self,
    throttle: float = 0.65,
    cruise_kmh: float = 42.0,
    turn_radius: float = 18.0,
    max_lat_accel: float = 4.0,
    brake_accel: float = 3.0,
    max_steer: float = 1.0,
    arrive_radius: float = 16.0,
    decide_radius: float = 70.0,
    goal_radius: float | None = None,
    allow_u_turn: bool = False,
    # "random"：每次 reset 从到达终点的简单路径中抽一条（增加数据多样性）。
    # "shortest"：始终锁定欧氏最短路。
    route_policy: str = "random",
    rng: random.Random | None = None,
    map_spec: MapSpec | None = None,
  ):
    self.map_spec = map_spec or get_map()
    self.throttle = throttle
    self.cruise_ms = cruise_kmh / KMH_PER_MS
    self.turn_radius = turn_radius
    self.max_lat_accel = max_lat_accel
    self.brake_accel = brake_accel
    self.max_steer = max_steer
    self.arrive_radius = arrive_radius
    self.decide_radius = decide_radius
    self.goal_radius = (
      self.map_spec.goal_radius if goal_radius is None else goal_radius
    )
    self.allow_u_turn = allow_u_turn
    if route_policy not in ("random", "shortest"):
      raise ValueError(f"unknown route_policy: {route_policy}")
    self.route_policy = route_policy
    self._rng = rng if rng is not None else random.Random()

    self.nodes = dict(self.map_spec.nav_nodes)
    self.adj = _build_adjacency(list(self.map_spec.nav_edges))
    self.goal_node = self.map_spec.goal_node
    self.spawn_node = self.map_spec.spawn_node
    self.goal_xy = self.map_spec.goal_xy

    self._prev_node: str | None = None
    self._target_node: str = self.spawn_node
    self._after_node: str | None = None
    self._route: list[str] = []
    self._path: _RefPath | None = None
    self._path_key: tuple[str | None, str, str | None] | None = None
    self.arrived = False
    self._command: str = "straight"
    self._hold_command: str | None = None
    self._hold_heading: float | None = None
    self._last_steer_log = ""
    self._steer_log_hold = 0
    self.reset()

  def plan_to_goal(
    self,
    start: str,
    goal: str | None = None,
    *,
    avoid_first: str | None = None,
  ) -> list[str]:
    """导航图上从 start 到 goal 的最短路（节点 id 列表）。

    边权为欧氏长度。当设置了 avoid_first 且不允许掉头时，第一跳不会回到
    该邻居，除非没有其他通往终点的路。
    """
    goal = goal or self.goal_node
    if start not in self.nodes:
      raise KeyError(f"unknown start node: {start}")
    if goal not in self.nodes:
      raise KeyError(f"unknown goal node: {goal}")
    if start == goal:
      return [start]

    def dijkstra(forbid_first: str | None) -> list[str] | None:
      dist: dict[str, float] = {start: 0.0}
      parent: dict[str, str | None] = {start: None}
      heap: list[tuple[float, str]] = [(0.0, start)]

      while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float("inf")):
          continue
        if node == goal:
          break
        sx, sy = self.nodes[node]
        for nxt in self.adj[node]:
          if forbid_first is not None and node == start and nxt == forbid_first:
            continue
          nx, ny = self.nodes[nxt]
          nd = d + math.hypot(nx - sx, ny - sy)
          if nd < dist.get(nxt, float("inf")):
            dist[nxt] = nd
            parent[nxt] = node
            heapq.heappush(heap, (nd, nxt))

      if goal not in parent:
        return None

      path: list[str] = []
      cur: str | None = goal
      while cur is not None:
        path.append(cur)
        cur = parent[cur]
      path.reverse()
      return path

    forbid = None if self.allow_u_turn else avoid_first
    path = dijkstra(forbid)
    if path is None and forbid is not None:
      path = dijkstra(None)
    if path is None:
      return [start]
    return path

  def enumerate_routes_to_goal(
    self,
    start: str,
    goal: str | None = None,
    *,
    avoid_first: str | None = None,
    max_routes: int = _MAX_ROUTE_CANDIDATES,
    max_extra_hops: int = _MAX_EXTRA_HOPS,
  ) -> list[list[str]]:
    """从 start 到 goal、在最短路跳数预算内的全部简单路径。

    用来让顶层规划随回合变化，而不是总锁在唯一的欧氏最短路上。
    """
    goal = goal or self.goal_node
    shortest = self.plan_to_goal(start, goal, avoid_first=avoid_first)
    if len(shortest) < 2:
      return [shortest]

    max_len = len(shortest) - 1 + max_extra_hops
    forbid = None if self.allow_u_turn else avoid_first
    routes: list[list[str]] = []

    def dfs(node: str, path: list[str]) -> None:
      if len(routes) >= max_routes:
        return
      if node == goal:
        routes.append(list(path))
        return
      if len(path) - 1 >= max_len:
        return
      for nxt in self.adj[node]:
        if nxt in path:
          continue
        if forbid is not None and len(path) == 1 and nxt == forbid:
          continue
        path.append(nxt)
        dfs(nxt, path)
        path.pop()

    dfs(start, [start])
    if not routes:
      return [shortest]
    # 稳定排序：先短后长，再按节点 id 字典序。
    routes.sort(key=lambda r: (len(r), r))
    return routes

  def _pick_route(
    self,
    start: str,
    *,
    avoid_first: str | None = None,
  ) -> list[str]:
    if self.route_policy == "shortest":
      return self.plan_to_goal(start, avoid_first=avoid_first)
    candidates = self.enumerate_routes_to_goal(start, avoid_first=avoid_first)
    return list(self._rng.choice(candidates))

  def commit_route(self, route: list[str]) -> None:
    """锁定顶层节点序列，并把图进度回退到该路线起点。"""
    if not route:
      raise ValueError("route must be non-empty")
    self._route = list(route)
    self._prev_node = route[0]
    self._target_node = route[1] if len(route) >= 2 else route[0]
    self._after_node = None
    self._path = None
    self._path_key = None
    self.arrived = False
    self._command = "straight"
    self._hold_command = None
    self._hold_heading = None
    self._last_steer_log = ""
    self._steer_log_hold = 0

  def reset(self):
    # 把出生点当作进路口节点；锁定一条通往终点的完整路线。
    if self.route_policy == "shortest":
      candidates = [self.plan_to_goal(self.spawn_node)]
    else:
      candidates = self.enumerate_routes_to_goal(self.spawn_node)
    self.commit_route(list(self._rng.choice(candidates)))
    print(
      f"[规划] {' → '.join(self._route)}  "
      f"({self.route_policy}, {len(candidates)} 条可选)",
      flush=True,
    )

  @property
  def route(self) -> list[str]:
    """本回合已锁定的顶层节点序列。"""
    return list(self._route)

  @property
  def command(self) -> str:
    """最新高层指令：straight | left | right | stop。"""
    return self._command

  @property
  def command_id(self) -> int:
    return COMMAND_TO_ID[self._command]

  @property
  def target_node(self) -> str:
    """当前图上追踪的下一跳节点。"""
    return self._target_node

  @property
  def target_pos(self) -> tuple[float, float]:
    """小地图上显示的下一目的地（若已选定出口则用排队出口）。"""
    if self._after_node is not None:
      return self.nodes[self._after_node]
    return self.nodes[self._target_node]

  def path_station(self, x: float, y: float) -> tuple[float, float, object]:
    """投影到当前参考路径：(弧长 m, 横向误差 m, path_key)。"""
    path = self._reference_path()
    station, cte = path.project(x, y)
    return station, cte, self._path_key

  def predict(
    self,
    x: float,
    y: float,
    heading_deg: float,
    speed_kmh: float | None = None,
    *,
    pedestrians: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
    dodge: object | None = None,
  ) -> tuple[float, float]:
    """返回 (油门, 转向)，转向为满打方向盘的比例。

    传入 ``dodge`` 与行人时，沿参考路径（含 left/right 圆弧）侧移前视点绕人，
    不改油门。直道与转弯同一套纯追踪，弯道曲率不会被绕行转向冲掉。
    """
    if self.arrived or self._at_goal(x, y):
      self.arrived = True
      self._command = "stop"
      self._hold_command = None
      self._hold_heading = None
      return 0.0, 0.0

    self._maybe_decide_exit(x, y)
    path = self._reference_path()
    station, cte = path.project(x, y)
    self._advance_if_past(station, x, y)
    if self._reference_path() is not path:
      path = self._reference_path()
      station, cte = path.project(x, y)

    self._command = self._compute_command(x, y, heading_deg)

    speed_ms = self.cruise_ms if speed_kmh is None else speed_kmh / KMH_PER_MS
    turning = path.on_arc(station)
    look = self._look_ahead(speed_ms, turning, path, station)
    lx, ly = path.point_at(station + look)
    if dodge is not None and pedestrians:
      plan_cte = getattr(dodge, "plan_cte", None)
      if plan_cte is not None:
        target_cte, _dodging = plan_cte(
          x,
          y,
          heading_deg,
          pedestrians,
          cte=cte,
          speed_kmh=0.0 if speed_kmh is None else float(speed_kmh),
          path=path,
          station=station,
          velocities=ped_velocities,
        )
        if target_cte is not None:
          lx, ly = path.offset_point(station + look, float(target_cte))

    bearing = _heading_deg(lx - x, ly - y)
    alpha = _normalize_angle_deg(bearing - heading_deg)
    reach = math.hypot(lx - x, ly - y)
    curvature = 2.0 * math.sin(math.radians(alpha)) / max(reach, 1.0)
    steer_deg = math.degrees(math.atan(WHEELBASE * curvature))
    steer = steer_deg / MAX_STEER_DEG
    steer = max(-self.max_steer, min(self.max_steer, steer))
    last = getattr(dodge, "last", None) if dodge is not None else None
    if isinstance(last, dict) and last.get("dodging"):
      last["steer"] = float(steer)
      last["turning"] = bool(turning)

    target_ms = self._target_speed(path, station, x, y)
    throttle = self._throttle_for(speed_ms, target_ms, speed_kmh is None, turning)

    self._log_steer(steer, turning, alpha, cte, speed_ms, target_ms)
    return throttle, steer

  # -- 参考路径 ---------------------------------------------------------

  def _reference_path(self) -> _RefPath:
    key = (self._prev_node, self._target_node, self._after_node)
    if self._path is None or self._path_key != key:
      self._path = self._build_path()
      self._path_key = key
    return self._path

  def _build_path(self) -> _RefPath:
    """进路口、可选弯道圆弧，再接出口车道。"""
    prev = self._prev_node or self.spawn_node
    px, py = self.nodes[prev]
    jx, jy = self.nodes[self._target_node]
    approach = math.hypot(jx - px, jy - py) or 1.0
    uix, uiy = (jx - px) / approach, (jy - py) / approach

    after = self._after_node
    if after is None:
      return _RefPath([(px, py), (jx, jy)], None, 0.0, approach)

    ax, ay = self.nodes[after]
    exit_len = math.hypot(ax - jx, ay - jy) or 1.0
    uox, uoy = (ax - jx) / exit_len, (ay - jy) / exit_len
    bend = _normalize_angle_deg(_heading_deg(uox, uoy) - _heading_deg(uix, uiy))

    if abs(bend) < TURN_COMMAND_DEG:
      return _RefPath(
        [(px, py), (jx, jy), (ax, ay)], None, 0.0, approach
      )

    half = math.radians(abs(bend)) * 0.5
    radius = self.turn_radius
    # 圆弧相对车道中心线的鼓出必须仍落在沥青上。
    bulge = 1.0 - math.cos(half)
    if bulge > 1e-6:
      radius = min(radius, (TRACK_HALF_WIDTH - LANE_MARGIN) / bulge)
    tangent = radius * math.tan(half)
    tangent = min(tangent, 0.45 * approach, 0.45 * exit_len)
    radius = tangent / math.tan(half)

    ex, ey = jx - uix * tangent, jy - uiy * tangent
    xx, xy = jx + uox * tangent, jy + uoy * tangent
    side = 1.0 if bend > 0 else -1.0
    cx = ex + (-uiy) * radius * side
    cy = ey + uix * radius * side

    theta0 = math.atan2(ey - cy, ex - cx)
    sweep = math.radians(bend)
    steps = max(6, int(abs(bend) / 5.0))
    points = [(px, py), (ex, ey)]
    for i in range(1, steps):
      theta = theta0 + sweep * (i / steps)
      points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    points.append((xx, xy))
    points.append((ax, ay))

    arc_start = approach - tangent
    arc_end = arc_start + radius * abs(sweep)
    return _RefPath(points, (arc_start, arc_end), radius, arc_end)

  def _look_ahead(
    self, speed_ms: float, turning: bool, path: _RefPath, station: float
  ) -> float:
    look = 5.0 + 0.6 * speed_ms
    if turning or path.distance_to_arc(station) < 25.0:
      # 过长的前视会跨过弯角、在切点之前抄近路——以前就是这样把车开上草地的。
      look = min(look, max(6.0, 0.55 * path.radius))
    return max(5.0, min(look, 16.0))

  # -- 车速 -------------------------------------------------------------

  def _target_speed(
    self, path: _RefPath, station: float, x: float, y: float
  ) -> float:
    target = self.cruise_ms

    if path.arc_span is not None:
      curve_ms = math.sqrt(self.max_lat_accel * path.radius)
      gap = path.distance_to_arc(station)
      # 比 3 m/s² 定点刹更早开始收速度，避免 42 km/h 开到切点才减速。
      turn_decel = min(self.brake_accel, 1.6)
      if station <= path.arc_span[1]:
        allowed = math.sqrt(curve_ms**2 + 2.0 * turn_decel * gap)
        target = min(target, allowed)
      elif station <= path.arc_span[1] + 8.0:
        target = min(target, curve_ms)

    if self._hold_command is not None:
      target = min(target, math.sqrt(self.max_lat_accel * max(self.turn_radius, 1.0)))

    gx, gy = self.goal_xy
    goal_gap = math.hypot(gx - x, gy - y) - self.goal_radius * 0.5
    if goal_gap < 60.0:
      target = min(target, math.sqrt(2.0 * self.brake_accel * max(goal_gap, 0.0)))

    # 最后几米缓行，避免停在旗帜前。
    return max(target, 2.0)

  def _throttle_for(
    self, speed_ms: float, target_ms: float, blind: bool, turning: bool
  ) -> float:
    if blind:
      # 没有速度反馈时，退回固定巡航油门。
      return self.throttle * (0.5 if turning else 1.0)

    err = target_ms - speed_ms
    if err > 0.2:
      return min(self.throttle, 0.15 + 0.35 * err)
    if err < -0.5:
      return max(-1.0, 0.35 * err)
    return 0.0

  # -- 图进度 -----------------------------------------------------------

  def _at_goal(self, x: float, y: float) -> bool:
    gx, gy = self.goal_xy
    return math.hypot(gx - x, gy - y) <= self.goal_radius

  def _dist_to_target(self, x: float, y: float) -> float:
    tx, ty = self.nodes[self._target_node]
    return math.hypot(tx - x, ty - y)

  def _maybe_decide_exit(self, x: float, y: float):
    if self._after_node is not None:
      return
    if self._target_node == self.goal_node:
      return
    if self._dist_to_target(x, y) > self.decide_radius:
      return
    self._after_node = self._choose_next(self._target_node, self._prev_node)

  def _advance_if_past(self, station: float, x: float, y: float):
    """路口已在车身后，切换到图上的下一跳。"""
    if self._target_node == self.goal_node:
      if self._dist_to_target(x, y) <= self.arrive_radius:
        self.arrived = True
      return

    path = self._reference_path()
    if station < path.handover:
      return

    nxt = self._after_node
    if nxt is None:
      nxt = self._choose_next(self._target_node, self._prev_node)
    if self._command in ("left", "right"):
      jx, jy = self.nodes[self._target_node]
      ax, ay = self.nodes[nxt]
      self._hold_command = self._command
      self._hold_heading = _heading_deg(ax - jx, ay - jy)
    self._prev_node = self._target_node
    self._target_node = nxt
    self._after_node = None
    self._path = None
    print(f"[切点] 进入 {self._prev_node}→{self._target_node}", flush=True)

  def _signed_bend_deg(
    self, prev: str | None, junction: str, after: str
  ) -> float:
    """路口带符号转角：正 = 左转，负 = 右转（Panda 航向）。"""
    jx, jy = self.nodes[junction]
    ax, ay = self.nodes[after]
    if prev is not None:
      px, py = self.nodes[prev]
      in_h = _heading_deg(jx - px, jy - py)
    else:
      in_h = _heading_deg(jx, jy + 220.0)
    out_h = _heading_deg(ax - jx, ay - jy)
    return _normalize_angle_deg(out_h - in_h)

  def _next_on_route(self, current: str) -> str | None:
    """已锁定顶层路线上 ``current`` 之后的节点（若有）。"""
    try:
      i = self._route.index(current)
    except ValueError:
      return None
    if i + 1 < len(self._route):
      return self._route[i + 1]
    return None

  def _peek_exit(self) -> str | None:
    if self._after_node is not None:
      return self._after_node
    if self._target_node == self.goal_node:
      return None
    nxt = self._next_on_route(self._target_node)
    if nxt is not None:
      return nxt
    path = self.plan_to_goal(self._target_node, avoid_first=self._prev_node)
    if len(path) >= 2:
      return path[1]
    return None

  def _compute_command(self, x: float, y: float, heading_deg: float) -> str:
    """当前位姿的高层导航指令（图更新之后）。"""
    if self.arrived or self._at_goal(x, y):
      self._hold_command = None
      self._hold_heading = None
      return "stop"
    if (
      self._hold_command is not None
      and self._hold_heading is not None
      and self._hold_command in ("left", "right")
    ):
      err = abs(_normalize_angle_deg(heading_deg - self._hold_heading))
      if err > TURN_COMMAND_HOLD_DEG:
        return self._hold_command
      self._hold_command = None
      self._hold_heading = None
    # 出口未锁定时不预告转弯；最后一跳驶向旗帜同理。
    if self._after_node is None:
      return "straight"

    signed = self._signed_bend_deg(
      self._prev_node, self._target_node, self._after_node
    )
    if abs(signed) <= TURN_COMMAND_DEG:
      return "straight"

    path = self._reference_path()
    station, _ = path.project(x, y)
    if path.arc_span is None:
      return "straight"
    # 与专家开始打方向对齐（前视碰到圆弧），而不是 decide_radius（70m）
    # 或弯前减速段。过早标 left/right，采集标签会把直道混进转弯头。
    if (
      not path.on_arc(station)
      and path.distance_to_arc(station) > TURN_COMMAND_PREVIEW_M
    ):
      return "straight"
    return "left" if signed > 0 else "right"

  def _choose_next(self, current: str, came_from: str | None) -> str:
    """已锁定路线上 current 的下一节点（找不到则重新规划）。"""
    nxt = self._next_on_route(current)
    if nxt is not None:
      return nxt
    # 已偏离锁定路线——从当前位置重新规划。
    self._route = self._pick_route(current, avoid_first=came_from)
    if len(self._route) >= 2:
      return self._route[1]
    return current

  def _log_steer(
    self,
    steer: float,
    turning: bool,
    alpha: float,
    cte: float,
    speed_ms: float,
    target_ms: float,
  ):
    """打印方向盘转动原因（限速输出，除非原因变化）。"""
    if abs(steer) < 0.02 and not turning:
      return

    mode = "弯道圆弧" if turning else "车道纠偏"
    direction = "左转" if steer > 0 else "右转"
    side = "偏左" if cte > 0 else "偏右"
    route = f"{self._prev_node or '-'}→{self._target_node}"
    if self._after_node is not None:
      route += f"→{self._after_node}"

    reason = (
      f"[转向] {direction} 前轮{steer * MAX_STEER_DEG:+.1f}° | {mode} "
      f"| 追踪点方位{alpha:+.1f}° | 横向{side}{abs(cte):.1f}m "
      f"| 速度{speed_ms * KMH_PER_MS:.0f}/{target_ms * KMH_PER_MS:.0f}km/h "
      f"| 路线 {route}"
    )

    key = f"{mode}|{direction}|{self._target_node}|{self._after_node}"
    if key != self._last_steer_log or self._steer_log_hold <= 0:
      # print(reason, flush=True)
      self._last_steer_log = key
      self._steer_log_hold = 15
    else:
      self._steer_log_hold -= 1

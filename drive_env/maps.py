"""多种驾驶地图布局（道路 + 导航图 + 出生点/终点）。

地图分为训练 / 测试集。用 ``--map train_maps`` 采集可覆盖全部训练布局。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TRACK_HALF_WIDTH = 10.0


@dataclass(frozen=True)
class MapSpec:
  """搭建可驾驶场地并在其上规划所需的全部信息。"""

  id: str
  name: str
  # 世界范围：(x_min, x_max, y_min, y_max)
  bounds: tuple[float, float, float, float]
  road_rects: tuple[tuple[float, float, float, float], ...]
  nav_nodes: dict[str, tuple[float, float]]
  nav_edges: tuple[tuple[str, str], ...]
  spawn_xy: tuple[float, float]
  goal_xy: tuple[float, float]
  spawn_heading: float = 0.0
  spawn_z: float = 2.5
  spawn_node: str = "spawn"
  goal_node: str = "goal"
  goal_radius: float = 12.0
  path_waypoints: tuple[tuple[float, float], ...] = ()
  # 斑马线所在路口中心：(cx, cy)。
  # 长直路上的路段过街由 crossing_sites() 追加。
  crosswalks: tuple[tuple[float, float], ...] = ()
  description: str = ""


def point_on_road(spec: MapSpec, x: float, y: float, slack: float = 0.0) -> bool:
  for x0, x1, y0, y1 in spec.road_rects:
    if (x0 - slack) <= x <= (x1 + slack) and (y0 - slack) <= y <= (y1 + slack):
      return True
  return False


def road_axes_at(
  spec: MapSpec,
  x: float,
  y: float,
  probe: float | None = None,
) -> tuple[bool, bool]:
  """返回 (x, y) 附近是否有南北向 / 东西向道路。"""
  d = TRACK_HALF_WIDTH + 2.0 if probe is None else probe
  has_ns = point_on_road(spec, x, y - d) or point_on_road(spec, x, y + d)
  has_ew = point_on_road(spec, x - d, y) or point_on_road(spec, x + d, y)
  return has_ns, has_ew


def crossing_sites(spec: MapSpec) -> tuple[tuple[float, float], ...]:
  """路口斑马线，加上长导航边上的路段过街点。"""
  sites = list(spec.crosswalks)
  min_edge = 80.0
  min_from_node = 28.0
  min_from_site = 36.0
  spacing = 100.0

  def _far_from_sites(x: float, y: float) -> bool:
    return all(math.hypot(x - sx, y - sy) >= min_from_site for sx, sy in sites)

  # 两条路交会的导航节点（覆盖无标线的 T 字 / 十字路口）。
  for name, (nx, ny) in spec.nav_nodes.items():
    if name in ("spawn", "goal") or "tip" in name:
      continue
    has_ns, has_ew = road_axes_at(spec, nx, ny)
    if has_ns and has_ew and _far_from_sites(nx, ny):
      sites.append((nx, ny))

  for a_name, b_name in spec.nav_edges:
    a, b = spec.nav_nodes[a_name], spec.nav_nodes[b_name]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < min_edge:
      continue
    n = min(3, max(1, int(length // spacing)))
    for i in range(n):
      t = (i + 1) / (n + 1)
      if t * length < min_from_node or (1.0 - t) * length < min_from_node:
        continue
      px, py = a[0] + dx * t, a[1] + dy * t
      has_ns, has_ew = road_axes_at(spec, px, py)
      if has_ns and has_ew:
        continue
      if not _far_from_sites(px, py):
        continue
      sites.append((round(px, 1), round(py, 1)))
  return tuple(sites)


def _hw() -> float:
  return TRACK_HALF_WIDTH


def _ns(x: float, y0: float, y1: float) -> tuple[float, float, float, float]:
  w = _hw()
  lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
  return (x - w, x + w, lo, hi)


def _ew(y: float, x0: float, x1: float) -> tuple[float, float, float, float]:
  w = _hw()
  lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
  return (lo, hi, y - w, y + w)


# ---------------------------------------------------------------------------
# 训练地图
# ---------------------------------------------------------------------------
_CROSSROADS = MapSpec(
  id="crossroads",
  name="十字路口",
  description="南北主干 + 双横向支路，东南绕行到旗子",
  bounds=(-240.0, 240.0, -260.0, 260.0),
  road_rects=(
    _ns(0.0, -240.0, 240.0),
    _ew(0.0, -200.0, 200.0),
    _ew(140.0, -_hw(), 200.0),
    _ns(-200.0, -_hw(), 140.0 + _hw()),
    _ew(140.0, -200.0, -_hw()),
    _ns(180.0, -_hw(), 230.0),
    _ew(200.0, _hw(), 180.0 + _hw()),
  ),
  nav_nodes={
    "spawn": (0.0, -220.0),
    "south_tip": (0.0, -240.0),
    "cross_0": (0.0, 0.0),
    "west_0": (-200.0, 0.0),
    "east_0": (180.0, 0.0),
    "east_tip_0": (200.0, 0.0),
    "cross_140": (0.0, 140.0),
    "west_140": (-200.0, 140.0),
    "east_tip_140": (200.0, 140.0),
    "cross_200": (0.0, 200.0),
    "north_tip": (0.0, 240.0),
    "east_200": (180.0, 200.0),
    "goal": (180.0, 220.0),
  },
  nav_edges=(
    ("spawn", "south_tip"),
    ("spawn", "cross_0"),
    ("cross_0", "west_0"),
    ("cross_0", "east_0"),
    ("cross_0", "cross_140"),
    ("west_0", "west_140"),
    ("east_0", "east_tip_0"),
    ("east_0", "east_200"),
    ("cross_140", "west_140"),
    ("cross_140", "east_tip_140"),
    ("cross_140", "cross_200"),
    ("cross_200", "north_tip"),
    ("cross_200", "east_200"),
    ("east_200", "goal"),
  ),
  spawn_xy=(0.0, -220.0),
  goal_xy=(180.0, 220.0),
  path_waypoints=((0.0, -220.0), (0.0, 0.0), (180.0, 0.0), (180.0, 220.0)),
  crosswalks=((0.0, 0.0), (0.0, 140.0)),
)


_L_BEND = MapSpec(
  id="l_bend",
  name="L 形弯道",
  description="先北行再东转到旗子，适合练单次直角弯",
  bounds=(-120.0, 260.0, -240.0, 120.0),
  road_rects=(
    _ns(0.0, -220.0, _hw()),
    _ew(0.0, -_hw(), 220.0),
  ),
  nav_nodes={
    "spawn": (0.0, -200.0),
    "south_tip": (0.0, -220.0),
    "corner": (0.0, 0.0),
    "east_tip": (220.0, 0.0),
    "goal": (200.0, 0.0),
  },
  nav_edges=(
    ("spawn", "south_tip"),
    ("spawn", "corner"),
    ("corner", "east_tip"),
    ("corner", "goal"),
  ),
  spawn_xy=(0.0, -200.0),
  goal_xy=(200.0, 0.0),
  path_waypoints=((0.0, -200.0), (0.0, 0.0), (200.0, 0.0)),
  crosswalks=((0.0, 0.0),),
)


_ZIGZAG = MapSpec(
  id="zigzag",
  name="折线路",
  description="两次反向直角弯，形成 S 形走廊",
  bounds=(-80.0, 280.0, -240.0, 240.0),
  road_rects=(
    _ns(0.0, -220.0, _hw()),
    _ew(0.0, -_hw(), 200.0 + _hw()),
    _ns(200.0, -_hw(), 160.0 + _hw()),
    _ew(160.0, -_hw(), 200.0 + _hw()),
    _ns(0.0, 160.0 - _hw(), 220.0),
  ),
  nav_nodes={
    "spawn": (0.0, -200.0),
    "south_tip": (0.0, -220.0),
    "turn1": (0.0, 0.0),
    "turn2": (200.0, 0.0),
    "turn3": (200.0, 160.0),
    "turn4": (0.0, 160.0),
    "north_tip": (0.0, 220.0),
    "goal": (0.0, 200.0),
  },
  nav_edges=(
    ("spawn", "south_tip"),
    ("spawn", "turn1"),
    ("turn1", "turn2"),
    ("turn2", "turn3"),
    ("turn3", "turn4"),
    ("turn4", "north_tip"),
    ("turn4", "goal"),
  ),
  spawn_xy=(0.0, -200.0),
  goal_xy=(0.0, 200.0),
  path_waypoints=(
    (0.0, -200.0),
    (0.0, 0.0),
    (200.0, 0.0),
    (200.0, 160.0),
    (0.0, 160.0),
    (0.0, 200.0),
  ),
  crosswalks=((0.0, 0.0), (200.0, 0.0), (200.0, 160.0), (0.0, 160.0)),
)


_T_JUNCTION = MapSpec(
  id="t_junction",
  name="T 字路口",
  description="北行至 T 口后右转驶向旗子",
  bounds=(-220.0, 240.0, -240.0, 120.0),
  road_rects=(
    _ns(0.0, -220.0, _hw()),
    _ew(0.0, -200.0, 200.0),
  ),
  nav_nodes={
    "spawn": (0.0, -200.0),
    "south_tip": (0.0, -220.0),
    "junction": (0.0, 0.0),
    "west_tip": (-200.0, 0.0),
    "east_tip": (200.0, 0.0),
    "goal": (180.0, 0.0),
  },
  nav_edges=(
    ("spawn", "south_tip"),
    ("spawn", "junction"),
    ("junction", "west_tip"),
    ("junction", "east_tip"),
    ("junction", "goal"),
  ),
  spawn_xy=(0.0, -200.0),
  goal_xy=(180.0, 0.0),
  path_waypoints=((0.0, -200.0), (0.0, 0.0), (180.0, 0.0)),
  crosswalks=((0.0, 0.0),),
)


_DUAL_BEND = MapSpec(
  id="dual_bend",
  name="双直角弯",
  description="北→东→北，两次同向右转练连续转弯",
  bounds=(-80.0, 260.0, -240.0, 220.0),
  road_rects=(
    _ns(0.0, -220.0, _hw()),
    _ew(0.0, -_hw(), 180.0 + _hw()),
    _ns(180.0, -_hw(), 180.0),
  ),
  nav_nodes={
    "spawn": (0.0, -200.0),
    "south_tip": (0.0, -220.0),
    "turn1": (0.0, 0.0),
    "turn2": (180.0, 0.0),
    "north_tip": (180.0, 180.0),
    "goal": (180.0, 160.0),
  },
  nav_edges=(
    ("spawn", "south_tip"),
    ("spawn", "turn1"),
    ("turn1", "turn2"),
    ("turn2", "north_tip"),
    ("turn2", "goal"),
  ),
  spawn_xy=(0.0, -200.0),
  goal_xy=(180.0, 160.0),
  path_waypoints=((0.0, -200.0), (0.0, 0.0), (180.0, 0.0), (180.0, 160.0)),
  crosswalks=((0.0, 0.0), (180.0, 0.0)),
)


_U_TURN = MapSpec(
  id="u_turn",
  name="U 形掉头",
  description="北行后东转再南下，走平行返程路到旗子",
  bounds=(-80.0, 240.0, -220.0, 220.0),
  road_rects=(
    _ns(0.0, -200.0, 160.0 + _hw()),
    _ew(160.0, -_hw(), 160.0 + _hw()),
    _ns(160.0, 160.0 + _hw(), -180.0),
  ),
  nav_nodes={
    "spawn": (0.0, -180.0),
    "south_tip": (0.0, -200.0),
    "nw": (0.0, 160.0),
    "ne": (160.0, 160.0),
    "se_tip": (160.0, -180.0),
    "goal": (160.0, -160.0),
  },
  nav_edges=(
    ("spawn", "south_tip"),
    ("spawn", "nw"),
    ("nw", "ne"),
    ("ne", "se_tip"),
    ("ne", "goal"),
  ),
  spawn_xy=(0.0, -180.0),
  goal_xy=(160.0, -160.0),
  path_waypoints=((0.0, -180.0), (0.0, 160.0), (160.0, 160.0), (160.0, -160.0)),
  crosswalks=((0.0, 160.0), (160.0, 160.0)),
)


# ---------------------------------------------------------------------------
# 测试地图（默认训练采集时留出）
# ---------------------------------------------------------------------------
_RING = MapSpec(
  id="ring",
  name="矩形环路",
  description="南→东→北 三边绕行到旗子（开口矩形）",
  bounds=(-220.0, 220.0, -220.0, 220.0),
  road_rects=(
    _ew(-160.0, -160.0, 160.0),
    _ns(160.0, -160.0, 160.0),
    _ew(160.0, -160.0, 160.0),
  ),
  nav_nodes={
    "spawn": (-120.0, -160.0),
    "sw": (-160.0, -160.0),
    "se": (160.0, -160.0),
    "ne": (160.0, 160.0),
    "nw": (-160.0, 160.0),
    "goal": (-120.0, 160.0),
  },
  nav_edges=(
    ("spawn", "sw"),
    ("spawn", "se"),
    ("se", "ne"),
    ("ne", "nw"),
    ("ne", "goal"),
  ),
  spawn_xy=(-120.0, -160.0),
  spawn_heading=-90.0,  # 朝向 +X（东）
  goal_xy=(-120.0, 160.0),
  path_waypoints=(
    (-120.0, -160.0),
    (160.0, -160.0),
    (160.0, 160.0),
    (-120.0, 160.0),
  ),
  crosswalks=(
    (160.0, -160.0),
    (160.0, 160.0),
  ),
)


_GRID = MapSpec(
  id="grid",
  name="城市网格",
  description="三纵三横街区，南侧支路汇入后最短路径需两次转弯",
  bounds=(-220.0, 220.0, -260.0, 220.0),
  road_rects=(
    _ns(-160.0, -180.0, 180.0),
    _ns(0.0, -240.0, 180.0),
    _ns(160.0, -180.0, 180.0),
    _ew(-160.0, -180.0, 180.0),
    _ew(0.0, -180.0, 180.0),
    _ew(160.0, -180.0, 180.0),
  ),
  nav_nodes={
    "spawn": (0.0, -220.0),
    "south_tip": (0.0, -240.0),
    "a0": (-160.0, -160.0),
    "b0": (0.0, -160.0),
    "c0": (160.0, -160.0),
    "a1": (-160.0, 0.0),
    "b1": (0.0, 0.0),
    "c1": (160.0, 0.0),
    "a2": (-160.0, 160.0),
    "b2": (0.0, 160.0),
    "c2": (160.0, 160.0),
    "goal": (160.0, 170.0),
  },
  nav_edges=(
    ("spawn", "south_tip"),
    ("spawn", "b0"),
    ("a0", "b0"),
    ("b0", "c0"),
    ("a0", "a1"),
    ("b0", "b1"),
    ("c0", "c1"),
    ("a1", "b1"),
    ("b1", "c1"),
    ("a1", "a2"),
    ("b1", "b2"),
    ("c1", "c2"),
    ("a2", "b2"),
    ("b2", "c2"),
    ("c2", "goal"),
  ),
  spawn_xy=(0.0, -220.0),
  goal_xy=(160.0, 170.0),
  path_waypoints=((0.0, -220.0), (0.0, 0.0), (160.0, 0.0), (160.0, 170.0)),
  crosswalks=(
    (-160.0, -160.0),
    (0.0, -160.0),
    (160.0, -160.0),
    (-160.0, 0.0),
    (0.0, 0.0),
    (160.0, 0.0),
    (-160.0, 160.0),
    (0.0, 160.0),
    (160.0, 160.0),
  ),
)


_CHICANE = MapSpec(
  id="chicane",
  name="连续变道",
  description="阶梯式同向偏移走廊（与训练 zigzag 几何不同）",
  bounds=(-80.0, 280.0, -240.0, 240.0),
  road_rects=(
    _ns(0.0, -220.0, -60.0 + _hw()),
    _ew(-60.0, -_hw(), 120.0 + _hw()),
    _ns(120.0, -60.0 - _hw(), 60.0 + _hw()),
    _ew(60.0, 120.0 - _hw(), 220.0 + _hw()),
    _ns(220.0, 60.0 - _hw(), 200.0),
  ),
  nav_nodes={
    "spawn": (0.0, -200.0),
    "south_tip": (0.0, -220.0),
    "a": (0.0, -60.0),
    "b": (120.0, -60.0),
    "c": (120.0, 60.0),
    "d": (220.0, 60.0),
    "north_tip": (220.0, 200.0),
    "goal": (220.0, 180.0),
  },
  nav_edges=(
    ("spawn", "south_tip"),
    ("spawn", "a"),
    ("a", "b"),
    ("b", "c"),
    ("c", "d"),
    ("d", "north_tip"),
    ("d", "goal"),
  ),
  spawn_xy=(0.0, -200.0),
  goal_xy=(220.0, 180.0),
  path_waypoints=(
    (0.0, -200.0),
    (0.0, -60.0),
    (120.0, -60.0),
    (120.0, 60.0),
    (220.0, 60.0),
    (220.0, 180.0),
  ),
  crosswalks=((0.0, -60.0), (120.0, -60.0), (120.0, 60.0), (220.0, 60.0)),
)


_SPUR = MapSpec(
  id="spur",
  name="支路掉头",
  description="南北干道中段东向支路，需转入支路到旗子",
  bounds=(-120.0, 240.0, -240.0, 240.0),
  road_rects=(
    _ns(0.0, -220.0, 220.0),
    _ew(80.0, -_hw(), 200.0),
  ),
  nav_nodes={
    "spawn": (0.0, -200.0),
    "south_tip": (0.0, -220.0),
    "cross": (0.0, 80.0),
    "north_tip": (0.0, 220.0),
    "spur_tip": (200.0, 80.0),
    "goal": (180.0, 80.0),
  },
  nav_edges=(
    ("spawn", "south_tip"),
    ("spawn", "cross"),
    ("cross", "north_tip"),
    ("cross", "spur_tip"),
    ("cross", "goal"),
  ),
  spawn_xy=(0.0, -200.0),
  goal_xy=(180.0, 80.0),
  path_waypoints=((0.0, -200.0), (0.0, 80.0), (180.0, 80.0)),
  crosswalks=((0.0, 80.0),),
)


_PLAZA = MapSpec(
  id="plaza",
  name="广场环线",
  description="闭合方环，西南角出发顺/逆时针到东北旗子",
  bounds=(-200.0, 200.0, -200.0, 200.0),
  road_rects=(
    _ew(-140.0, -140.0, 140.0),
    _ns(140.0, -140.0, 140.0),
    _ew(140.0, -140.0, 140.0),
    _ns(-140.0, -140.0, 140.0),
  ),
  # 西南角朝东：第一跳向东 (se)，或左转向北 (nw)。
  nav_nodes={
    "spawn": (-140.0, -140.0),
    "se": (140.0, -140.0),
    "ne": (140.0, 140.0),
    "nw": (-140.0, 140.0),
    "goal": (100.0, 140.0),
  },
  nav_edges=(
    ("spawn", "se"),
    ("spawn", "nw"),
    ("se", "ne"),
    ("ne", "nw"),
    ("ne", "goal"),
  ),
  spawn_xy=(-140.0, -140.0),
  spawn_heading=-90.0,  # 朝向 +X（东）
  goal_xy=(100.0, 140.0),
  path_waypoints=(
    (-140.0, -140.0),
    (140.0, -140.0),
    (140.0, 140.0),
    (100.0, 140.0),
  ),
  crosswalks=((140.0, -140.0), (140.0, 140.0), (-140.0, 140.0)),
)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
_TRAIN_LIST = (
  _CROSSROADS,
  _L_BEND,
  _ZIGZAG,
  _T_JUNCTION,
  _DUAL_BEND,
  _U_TURN,
)
_TEST_LIST = (
  _RING,
  _GRID,
  _CHICANE,
  _SPUR,
  _PLAZA,
)

TRAIN_MAPS: dict[str, MapSpec] = {m.id: m for m in _TRAIN_LIST}
TEST_MAPS: dict[str, MapSpec] = {m.id: m for m in _TEST_LIST}
MAPS: dict[str, MapSpec] = {**TRAIN_MAPS, **TEST_MAPS}

TRAIN_MAP_IDS: tuple[str, ...] = tuple(TRAIN_MAPS)
TEST_MAP_IDS: tuple[str, ...] = tuple(TEST_MAPS)

MAP_GROUPS: dict[str, tuple[str, ...]] = {
  "train_maps": TRAIN_MAP_IDS,
  "test_maps": TEST_MAP_IDS,
  "all": TRAIN_MAP_IDS + TEST_MAP_IDS,
}

DEFAULT_MAP_ID = "crossroads"


def list_maps() -> list[MapSpec]:
  return list(MAPS.values())


def list_train_maps() -> list[MapSpec]:
  return list(TRAIN_MAPS.values())


def list_test_maps() -> list[MapSpec]:
  return list(TEST_MAPS.values())


def get_map(map_id: str | None = None) -> MapSpec:
  key = (map_id or DEFAULT_MAP_ID).strip().lower()
  if key not in MAPS:
    known = ", ".join(MAPS)
    raise KeyError(f"unknown map '{map_id}'; choose from: {known}")
  return MAPS[key]


def resolve_maps(selection: str | None = None) -> list[str]:
  """把地图 id 或分组别名（train_maps / test_maps / all）展开为地图 id 列表。"""
  key = (selection or DEFAULT_MAP_ID).strip().lower()
  if key in MAP_GROUPS:
    return list(MAP_GROUPS[key])
  if key in MAPS:
    return [key]
  known = ", ".join([*MAPS, *MAP_GROUPS])
  raise KeyError(f"unknown map '{selection}'; choose from: {known}")


def map_choices() -> list[str]:
  """单张地图 id（给 main.py 这类单图界面用）。"""
  return list(MAPS.keys())


def collect_map_choices() -> list[str]:
  """地图 id 加上分组别名，供数据采集 / 批处理工具使用。"""
  return [*MAPS.keys(), *MAP_GROUPS.keys()]

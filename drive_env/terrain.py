"""地形网格、沥青道路、终点旗帜与边界墙。"""

from __future__ import annotations

import math

from panda3d.bullet import (
  BulletBoxShape,
  BulletRigidBodyNode,
  BulletTriangleMesh,
  BulletTriangleMeshShape,
)
from panda3d.core import (
  Geom,
  GeomNode,
  GeomTriangles,
  GeomVertexData,
  GeomVertexFormat,
  GeomVertexWriter,
  Vec3,
)

from drive_env.maps import (
  DEFAULT_MAP_ID,
  TRACK_HALF_WIDTH,
  MapSpec,
  crossing_sites,
  get_map,
  point_on_road,
  road_axes_at,
)

TERRAIN_STEP = 8.0

TRACK_EDGE_WIDTH = 0.4
TRACK_LIFT = 0.05
EDGE_LIFT = 0.08

ASPHALT = (0.16, 0.16, 0.18, 1.0)
CURB = (0.92, 0.82, 0.18, 1.0)
FLAG_POLE = (0.85, 0.85, 0.88, 1.0)
FLAG_RED = (0.9, 0.12, 0.12, 1.0)
FLAG_WHITE = (0.95, 0.95, 0.95, 1.0)

# 默认地图的向后兼容别名（测试 / 旧导入）。
_DEFAULT = get_map(DEFAULT_MAP_ID)
ROAD_RECTS = list(_DEFAULT.road_rects)
PATH_WAYPOINTS = list(_DEFAULT.path_waypoints)
NAV_NODES = dict(_DEFAULT.nav_nodes)
NAV_EDGES = list(_DEFAULT.nav_edges)
GOAL_POS = Vec3(_DEFAULT.goal_xy[0], _DEFAULT.goal_xy[1], 0.0)
GOAL_RADIUS = _DEFAULT.goal_radius
GOAL_NODE = _DEFAULT.goal_node
SPAWN_POS = Vec3(_DEFAULT.spawn_xy[0], _DEFAULT.spawn_xy[1], _DEFAULT.spawn_z)
SPAWN_HEADING = _DEFAULT.spawn_heading
SPAWN_NODE = _DEFAULT.spawn_node
WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX = _DEFAULT.bounds


def height_at(x: float, y: float) -> float:
  """轻微起伏的平原——道路附近保持接近平坦。"""
  del x
  return 0.35 * (0.5 + 0.5 * math.sin(y * 0.012))


def _grass_color(z: float) -> tuple[float, float, float, float]:
  if z > 0.4:
    return (0.32, 0.4, 0.28, 1.0)
  return (0.22, 0.32, 0.2, 1.0)


def _add_quad(
  vertex: GeomVertexWriter,
  normal: GeomVertexWriter,
  color_w: GeomVertexWriter,
  tris: GeomTriangles,
  p0: Vec3,
  p1: Vec3,
  p2: Vec3,
  p3: Vec3,
  color: tuple[float, float, float, float],
):
  """把四边形拆成两个三角形（p0-p1-p2-p3，俯视逆时针）。"""
  i0 = vertex.getWriteRow()
  for p in (p0, p1, p2, p3):
    vertex.addData3(p)
    normal.addData3(0, 0, 1)
    color_w.addData4(*color)
  tris.addVertices(i0, i0 + 1, i0 + 2)
  tris.addVertices(i0, i0 + 2, i0 + 3)


def _build_colored_geom(name: str, builder) -> GeomNode:
  fmt = GeomVertexFormat.getV3n3c4()
  vdata = GeomVertexData(name, fmt, Geom.UHStatic)
  vertex = GeomVertexWriter(vdata, "vertex")
  normal = GeomVertexWriter(vdata, "normal")
  color_w = GeomVertexWriter(vdata, "color")
  tris = GeomTriangles(Geom.UHStatic)
  builder(vertex, normal, color_w, tris)
  vdata.setNumRows(vertex.getWriteRow())
  tris.closePrimitive()
  geom = Geom(vdata)
  geom.addPrimitive(tris)
  node = GeomNode(name)
  node.addGeom(geom)
  return node


def _axis_samples(lo: float, hi: float, step: float) -> list[float]:
  """闭区间采样，使长路贴合地形，而不是一整块平板。"""
  if hi < lo:
    lo, hi = hi, lo
  pts = [lo]
  x = lo + step
  while x < hi - 1e-6:
    pts.append(x)
    x += step
  if pts[-1] < hi - 1e-6:
    pts.append(hi)
  return pts


def _build_rect_mesh(
  name: str,
  x_min: float,
  x_max: float,
  y_min: float,
  y_max: float,
  lift: float,
  color: tuple[float, float, float, float],
) -> GeomNode:
  def build(vertex, normal, color_w, tris):
    xs = _axis_samples(x_min, x_max, TERRAIN_STEP)
    ys = _axis_samples(y_min, y_max, TERRAIN_STEP)
    for j in range(len(ys) - 1):
      for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[j], ys[j + 1]
        _add_quad(
          vertex,
          normal,
          color_w,
          tris,
          Vec3(x0, y0, height_at(x0, y0) + lift),
          Vec3(x1, y0, height_at(x1, y0) + lift),
          Vec3(x1, y1, height_at(x1, y1) + lift),
          Vec3(x0, y1, height_at(x0, y1) + lift),
          color,
        )

  return _build_colored_geom(name, build)


def _build_track_visuals(parent, map_spec: MapSpec):
  """每条道路 AABB 的沥青矩形 + 路缘条。"""
  root = parent.attachNewNode("roads")
  w = TRACK_EDGE_WIDTH

  for i, (x0, x1, y0, y1) in enumerate(map_spec.road_rects):
    asphalt = root.attachNewNode(
      _build_rect_mesh(f"asphalt_{i}", x0, x1, y0, y1, TRACK_LIFT, ASPHALT)
    )
    asphalt.setLightOff()

    curbs = (
      (x0 - w, x0, y0, y1),
      (x1, x1 + w, y0, y1),
      (x0, x1, y0 - w, y0),
      (x0, x1, y1, y1 + w),
    )
    for j, (cx0, cx1, cy0, cy1) in enumerate(curbs):
      curb = root.attachNewNode(
        _build_rect_mesh(f"curb_{i}_{j}", cx0, cx1, cy0, cy1, EDGE_LIFT, CURB)
      )
      curb.setLightOff()

  return root


def _paint_zebra_band(root, prefix: str, ax: float, ay: float, across_x: bool) -> None:
  """以 (ax, ay) 为中心画五条白线。南北向道路时 across_x=True。"""
  white = (0.92, 0.92, 0.9, 1.0)
  stripe_w = 1.2
  stripe_len = 8.0
  gap = 1.2
  for k in range(5):
    offset = (k - 2) * (stripe_w + gap)
    if across_x:
      x0, x1 = ax + offset - stripe_w * 0.5, ax + offset + stripe_w * 0.5
      y0, y1 = ay - stripe_len * 0.5, ay + stripe_len * 0.5
    else:
      x0, x1 = ax - stripe_len * 0.5, ax + stripe_len * 0.5
      y0, y1 = ay + offset - stripe_w * 0.5, ay + offset + stripe_w * 0.5
    stripe = root.attachNewNode(
      _build_rect_mesh(
        f"{prefix}_{k}",
        x0,
        x1,
        y0,
        y1,
        TRACK_LIFT + 0.03,
        white,
      )
    )
    stripe.setLightOff()


def _build_crosswalks(parent, map_spec: MapSpec):
  """路口与路段过街点的斑马线。"""
  root = parent.attachNewNode("crosswalks")
  hw = TRACK_HALF_WIDTH

  for idx, (cx, cy) in enumerate(crossing_sites(map_spec)):
    has_ns, has_ew = road_axes_at(map_spec, cx, cy)
    if has_ns and has_ew:
      approaches = (
        ("s", cx, cy - hw - 1.0, True),
        ("n", cx, cy + hw + 1.0, True),
        ("w", cx - hw - 1.0, cy, False),
        ("e", cx + hw + 1.0, cy, False),
      )
      for name, ax, ay, across_x in approaches:
        if point_on_road(map_spec, ax, ay, slack=2.0):
          _paint_zebra_band(root, f"zebra{idx}_{name}", ax, ay, across_x)
    elif has_ns:
      _paint_zebra_band(root, f"zebra{idx}_mid", cx, cy, True)
    elif has_ew:
      _paint_zebra_band(root, f"zebra{idx}_mid", cx, cy, False)

  return root


def build_terrain(render, physics: "PhysicsWorld", map_spec: MapSpec | None = None):
  """草地基底网格（Bullet 碰撞）+ 道路叠加。"""
  map_spec = map_spec or get_map()
  x_min, x_max, y_min, y_max = map_spec.bounds
  step = TERRAIN_STEP

  fmt = GeomVertexFormat.getV3n3c4()
  vdata = GeomVertexData("terrain", fmt, Geom.UHStatic)
  vertex = GeomVertexWriter(vdata, "vertex")
  normal = GeomVertexWriter(vdata, "normal")
  color = GeomVertexWriter(vdata, "color")

  bullet_mesh = BulletTriangleMesh()

  xs = []
  x = x_min
  while x <= x_max + 0.001:
    xs.append(x)
    x += step

  ys = []
  y = y_min
  while y <= y_max + 0.001:
    ys.append(y)
    y += step

  grid = {}
  for j, y in enumerate(ys):
    for i, x in enumerate(xs):
      z = height_at(x, y)
      grid[(i, j)] = vertex.getWriteRow()
      vertex.addData3(x, y, z)
      color.addData4(*_grass_color(z))
      normal.addData3(0, 0, 1)

  tris = GeomTriangles(Geom.UHStatic)
  for j in range(len(ys) - 1):
    for i in range(len(xs) - 1):
      i00 = grid[(i, j)]
      i10 = grid[(i + 1, j)]
      i01 = grid[(i, j + 1)]
      i11 = grid[(i + 1, j + 1)]

      x00, y00 = xs[i], ys[j]
      x10, y10 = xs[i + 1], ys[j]
      x01, y01 = xs[i], ys[j + 1]
      x11, y11 = xs[i + 1], ys[j + 1]

      p00 = Vec3(x00, y00, height_at(x00, y00))
      p10 = Vec3(x10, y10, height_at(x10, y10))
      p01 = Vec3(x01, y01, height_at(x01, y01))
      p11 = Vec3(x11, y11, height_at(x11, y11))

      tris.addVertices(i00, i10, i01)
      tris.addVertices(i10, i11, i01)
      bullet_mesh.addTriangle(p00, p10, p01)
      bullet_mesh.addTriangle(p10, p11, p01)

  vdata.setNumRows(vertex.getWriteRow())
  tris.closePrimitive()

  geom = Geom(vdata)
  geom.addPrimitive(tris)
  geom_node = GeomNode("terrain_geom")
  geom_node.addGeom(geom)

  root = render.attachNewNode("terrain_root")
  terrain = root.attachNewNode(geom_node)
  terrain.setTwoSided(True)
  terrain.setLightOff()

  _build_track_visuals(root, map_spec)
  _build_crosswalks(root, map_spec)

  shape = BulletTriangleMeshShape(bullet_mesh, dynamic=False)
  shape.setMargin(0.04)
  body = BulletRigidBodyNode("terrain")
  body.addShape(shape)
  body_np = render.attachNewNode(body)
  physics.attach_static(body)

  return root, body_np


def build_barriers(render, physics: "PhysicsWorld", map_spec: MapSpec | None = None):
  """外围边界墙，把车留在场地内。"""
  map_spec = map_spec or get_map()
  x_min, x_max, y_min, y_max = map_spec.bounds
  walls = []
  z = 1.6
  thickness = 1.2
  half_h = 1.6

  cx = (x_min + x_max) * 0.5
  cy = (y_min + y_max) * 0.5
  specs = (
    (cx, y_min - thickness, (x_max - x_min) * 0.5 + thickness, thickness),
    (cx, y_max + thickness, (x_max - x_min) * 0.5 + thickness, thickness),
    (x_min - thickness, cy, thickness, (y_max - y_min) * 0.5 + thickness),
    (x_max + thickness, cy, thickness, (y_max - y_min) * 0.5 + thickness),
  )

  for i, (cx, cy, hx, hy) in enumerate(specs):
    shape = BulletBoxShape(Vec3(hx, hy, half_h))
    body = BulletRigidBodyNode(f"boundary_{i}")
    body.addShape(shape)
    np = render.attachNewNode(body)
    np.setPos(cx, cy, z)
    physics.attach_static(body)
    walls.append(np)

  return walls


def build_goal_flag(parent, map_spec: MapSpec | None = None):
  """地图终点处杆上的格子旗。"""
  map_spec = map_spec or get_map()
  root = parent.attachNewNode("goal_flag")
  gx, gy = map_spec.goal_xy
  gz = height_at(gx, gy)

  pole_half = Vec3(0.12, 0.12, 4.0)
  pole = root.attachNewNode(
    _build_box_geom("flag_pole", pole_half, FLAG_POLE)
  )
  pole.setPos(gx, gy, gz + pole_half.z)
  pole.setLightOff()

  tile = 0.9
  cols, rows = 3, 2
  for r in range(rows):
    for c in range(cols):
      color = FLAG_RED if (r + c) % 2 == 0 else FLAG_WHITE
      fx0 = gx + 0.2 + c * tile
      fx1 = fx0 + tile
      fz0 = gz + 5.5 + r * tile
      fz1 = fz0 + tile
      card = root.attachNewNode(
        _build_vertical_quad(f"flag_{r}_{c}", fx0, gy, fz0, fx1, gy, fz1, color)
      )
      card.setTwoSided(True)
      card.setLightOff()

  marker = root.attachNewNode(
    _build_rect_mesh(
      "goal_pad",
      gx - 3.0,
      gx + 3.0,
      gy - 3.0,
      gy + 3.0,
      TRACK_LIFT + 0.02,
      (0.85, 0.15, 0.15, 1.0),
    )
  )
  marker.setLightOff()

  return root


def _build_box_geom(name: str, half: Vec3, color: tuple[float, float, float, float]) -> GeomNode:
  hx, hy, hz = half.x, half.y, half.z
  faces = (
    (Vec3(-hx, -hy, hz), Vec3(hx, -hy, hz), Vec3(hx, hy, hz), Vec3(-hx, hy, hz), Vec3(0, 0, 1)),
    (Vec3(-hx, hy, -hz), Vec3(hx, hy, -hz), Vec3(hx, -hy, -hz), Vec3(-hx, -hy, -hz), Vec3(0, 0, -1)),
    (Vec3(-hx, hy, -hz), Vec3(-hx, hy, hz), Vec3(hx, hy, hz), Vec3(hx, hy, -hz), Vec3(0, 1, 0)),
    (Vec3(hx, -hy, -hz), Vec3(hx, -hy, hz), Vec3(-hx, -hy, hz), Vec3(-hx, -hy, -hz), Vec3(0, -1, 0)),
    (Vec3(hx, -hy, -hz), Vec3(hx, hy, -hz), Vec3(hx, hy, hz), Vec3(hx, -hy, hz), Vec3(1, 0, 0)),
    (Vec3(-hx, -hy, hz), Vec3(-hx, hy, hz), Vec3(-hx, hy, -hz), Vec3(-hx, -hy, -hz), Vec3(-1, 0, 0)),
  )

  def build(vertex, normal, color_w, tris):
    for p0, p1, p2, p3, n in faces:
      i0 = vertex.getWriteRow()
      for p in (p0, p1, p2, p3):
        vertex.addData3(p)
        normal.addData3(n)
        color_w.addData4(*color)
      tris.addVertices(i0, i0 + 1, i0 + 2)
      tris.addVertices(i0, i0 + 2, i0 + 3)

  return _build_colored_geom(name, build)


def _build_vertical_quad(
  name: str,
  x0: float,
  y0: float,
  z0: float,
  x1: float,
  y1: float,
  z1: float,
  color: tuple[float, float, float, float],
) -> GeomNode:
  def build(vertex, normal, color_w, tris):
    _add_quad(
      vertex,
      normal,
      color_w,
      tris,
      Vec3(x0, y0, z0),
      Vec3(x1, y1, z0),
      Vec3(x1, y1, z1),
      Vec3(x0, y0, z1),
      color,
    )

  return _build_colored_geom(name, build)


def spawn_point(map_spec: MapSpec | None = None) -> tuple[Vec3, float]:
  map_spec = map_spec or get_map()
  sx, sy = map_spec.spawn_xy
  return Vec3(sx, sy, map_spec.spawn_z), map_spec.spawn_heading


def goal_reached(x: float, y: float, map_spec: MapSpec | None = None) -> bool:
  map_spec = map_spec or get_map()
  gx, gy = map_spec.goal_xy
  r = map_spec.goal_radius
  dx = x - gx
  dy = y - gy
  return dx * dx + dy * dy <= r * r


def world_bounds(map_spec: MapSpec | None = None) -> tuple[float, float, float, float]:
  map_spec = map_spec or get_map()
  return map_spec.bounds

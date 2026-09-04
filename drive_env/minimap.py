"""显示道路、车辆与终点的 2D 小地图叠加层。"""

from __future__ import annotations

from panda3d.core import (
  CardMaker,
  NodePath,
  TransparencyAttrib,
  Vec3,
)

from drive_env.maps import MapSpec, get_map


class MiniMap:
  """画在 aspect2d 上的角落地图。"""

  def __init__(
    self,
    aspect2d: NodePath,
    map_spec: MapSpec | None = None,
    size: float = 0.42,
  ):
    self.map_spec = map_spec or get_map()
    self.size = size
    self.x_min, self.x_max, self.y_min, self.y_max = self.map_spec.bounds
    self.world_w = self.x_max - self.x_min
    self.world_h = self.y_max - self.y_min

    # 右下角
    self.root = aspect2d.attachNewNode("minimap")
    self.root.setPos(1.28 - size * 0.55, 0, -0.92 + size * 0.55)
    self.root.setTransparency(TransparencyAttrib.MAlpha)

    self._build_frame()
    self._build_roads()
    self._build_goal()
    self._build_target()
    self._build_car()
    self._ped_markers: list[NodePath] = []

  def _build_frame(self):
    half = self.size * 0.5
    pad = 0.008

    border = CardMaker("minimap_border")
    border.setFrame(-half - pad, half + pad, -half - pad, half + pad)
    edge = self.root.attachNewNode(border.generate())
    edge.setColor(0.75, 0.78, 0.82, 0.95)
    edge.setBin("fixed", 39)
    edge.setDepthTest(False)
    edge.setDepthWrite(False)

    cm = CardMaker("minimap_bg")
    cm.setFrame(-half, half, -half, half)
    bg = self.root.attachNewNode(cm.generate())
    bg.setColor(0.05, 0.07, 0.1, 0.82)
    bg.setBin("fixed", 40)
    bg.setDepthTest(False)
    bg.setDepthWrite(False)

  def _world_to_map(self, x: float, y: float) -> tuple[float, float]:
    half = self.size * 0.5
    mx = ((x - self.x_min) / self.world_w - 0.5) * self.size
    my = ((y - self.y_min) / self.world_h - 0.5) * self.size
    return max(-half, min(half, mx)), max(-half, min(half, my))

  def _build_roads(self):
    roads = self.root.attachNewNode("minimap_roads")
    roads.setBin("fixed", 41)
    roads.setDepthTest(False)
    roads.setDepthWrite(False)

    for i, (x0, x1, y0, y1) in enumerate(self.map_spec.road_rects):
      lx0, ly0 = self._world_to_map(x0, y0)
      lx1, ly1 = self._world_to_map(x1, y1)
      cm = CardMaker(f"road_{i}")
      cm.setFrame(min(lx0, lx1), max(lx0, lx1), min(ly0, ly1), max(ly0, ly1))
      np = roads.attachNewNode(cm.generate())
      np.setColor(0.35, 0.36, 0.4, 0.95)

  def _build_goal(self):
    gx, gy = self._world_to_map(*self.map_spec.goal_xy)
    s = 0.018
    cm = CardMaker("goal")
    cm.setFrame(gx - s, gx + s, gy - s, gy + s)
    self.goal_np = self.root.attachNewNode(cm.generate())
    self.goal_np.setColor(0.95, 0.2, 0.2, 1)
    self.goal_np.setBin("fixed", 42)
    self.goal_np.setDepthTest(False)
    self.goal_np.setDepthWrite(False)

  def _build_target(self):
    """下一导航路点（每帧更新）。"""
    s = 0.014
    cm = CardMaker("nav_target")
    cm.setFrame(-s, s, -s, s)
    self.target_np = self.root.attachNewNode(cm.generate())
    self.target_np.setColor(1.0, 0.85, 0.15, 1)
    self.target_np.setBin("fixed", 42)
    self.target_np.setDepthTest(False)
    self.target_np.setDepthWrite(False)
    self.target_np.hide()

  def _build_car(self):
    s = 0.022
    cm = CardMaker("car")
    cm.setFrame(-s * 0.55, s * 0.55, -s * 0.7, s * 0.7)
    self.car_np = self.root.attachNewNode(cm.generate())
    self.car_np.setColor(0.2, 0.85, 1.0, 1)
    self.car_np.setBin("fixed", 43)
    self.car_np.setDepthTest(False)
    self.car_np.setDepthWrite(False)

  def update(
    self,
    pos: Vec3,
    heading_deg: float,
    target: tuple[float, float] | None = None,
    pedestrians: list[tuple[float, float]] | None = None,
  ):
    mx, my = self._world_to_map(pos.x, pos.y)
    self.car_np.setPos(mx, 0, my)
    self.car_np.setR(-heading_deg)

    if target is None:
      self.target_np.hide()
    else:
      tx, ty = self._world_to_map(target[0], target[1])
      self.target_np.setPos(tx, 0, ty)
      self.target_np.show()

    self._sync_pedestrian_markers(pedestrians or [])

  def _sync_pedestrian_markers(self, positions: list[tuple[float, float]]) -> None:
    while len(self._ped_markers) < len(positions):
      s = 0.012
      cm = CardMaker(f"ped_{len(self._ped_markers)}")
      cm.setFrame(-s, s, -s, s)
      marker = self.root.attachNewNode(cm.generate())
      marker.setColor(1.0, 0.55, 0.15, 1)
      marker.setBin("fixed", 42)
      marker.setDepthTest(False)
      marker.setDepthWrite(False)
      self._ped_markers.append(marker)

    for i, marker in enumerate(self._ped_markers):
      if i < len(positions):
        px, py = self._world_to_map(*positions[i])
        marker.setPos(px, 0, py)
        marker.show()
      else:
        marker.hide()

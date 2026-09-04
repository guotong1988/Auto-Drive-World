from panda3d.core import AmbientLight, DirectionalLight, Vec3

from drive_env.maps import MapSpec, get_map
from drive_env.pedestrian import PedestrianCrowd
from drive_env.physics import PhysicsWorld
from drive_env.terrain import (
  build_barriers,
  build_goal_flag,
  build_terrain,
  goal_reached,
  spawn_point,
  world_bounds,
)


class World:
  """驾驶场地：路网、护栏、终点旗帜与灯光。"""

  def __init__(self, render, physics: PhysicsWorld, map_spec: MapSpec | None = None):
    self.map_spec = map_spec or get_map()
    self.root = render.attachNewNode("world")
    self.physics = physics

    self.terrain_visual, self.terrain_body = build_terrain(
      render, physics, self.map_spec
    )
    self.terrain_visual.reparentTo(self.root)
    self.barriers = build_barriers(render, physics, self.map_spec)
    self.goal_flag = build_goal_flag(self.root, self.map_spec)

    self.spawn_pos, self.spawn_h = spawn_point(self.map_spec)
    gx, gy = self.map_spec.goal_xy
    self.goal_pos = Vec3(gx, gy, 0.0)
    self.path_waypoints = list(self.map_spec.path_waypoints)
    self.road_rects = list(self.map_spec.road_rects)
    self.bounds = world_bounds(self.map_spec)
    self.pedestrians = PedestrianCrowd(self.root, self.map_spec)

    self._build_lighting()

  def reached_goal(self, x: float, y: float) -> bool:
    return goal_reached(x, y, self.map_spec)

  def _build_lighting(self):
    ambient = AmbientLight("ambient")
    ambient.setColor((0.45, 0.45, 0.5, 1))
    ambient_np = self.root.attachNewNode(ambient)
    self.root.setLight(ambient_np)

    sun = DirectionalLight("sun")
    sun.setColor((0.95, 0.92, 0.85, 1))
    sun_np = self.root.attachNewNode(sun)
    sun_np.setHpr(Vec3(-35, -55, 0))
    self.root.setLight(sun_np)

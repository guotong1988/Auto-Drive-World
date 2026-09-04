from panda3d.bullet import BulletWorld
from panda3d.core import Vec3


class PhysicsWorld:
  """Bullet 仿真世界。"""

  GRAVITY = Vec3(0, 0, -22.0)

  def __init__(self):
    self.world = BulletWorld()
    self.world.setGravity(self.GRAVITY)

  def attach_static(self, body_node):
    body_node.setMass(0)
    body_node.setFriction(1.1)
    self.world.attachRigidBody(body_node)

  def attach_dynamic(self, body_node):
    self.world.attachRigidBody(body_node)

  def attach_vehicle(self, vehicle):
    self.world.attachVehicle(vehicle)

  def step(self, dt: float):
    max_substeps = max(1, int(dt * 120) + 1)
    self.world.doPhysics(dt, max_substeps, 1.0 / 120.0)

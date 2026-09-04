import math

from panda3d.bullet import (
  BulletBoxShape,
  BulletRigidBodyNode,
  BulletVehicle,
  ZUp,
)
from panda3d.core import CardMaker, Point3, TransformState, Vec3

from drive_env.dynamics import MAX_STEER_DEG, STEER_RATE_DEG
from drive_env.physics import PhysicsWorld

# BC 转弯头标签约 98% 是油门 0（弯中滑行）。MSE 会留下约 -0.05 的残差；
# 若把任意负油门都当刹车，车速会掉到转弯头从未见过的区间（数据里 left
# 最低约 22 km/h），油门抬不起来，整车刹停。专家真正限速刹车约 -0.18。
THROTTLE_BRAKE_DEADZONE = 0.12


def apply_drive(vehicle: "Vehicle", throttle: float, steer: float) -> None:
  """有符号油门 → 发动机/刹车。自动驾驶不倒车。

  (-deadzone, 0) 当滑行；只有明显为负才刹车（专家过弯限速约 -0.18）。
  """
  throttle = max(-1.0, min(1.0, float(throttle)))
  steer = max(-1.0, min(1.0, float(steer)))
  if throttle <= -THROTTLE_BRAKE_DEADZONE:
    vehicle.set_input(0.0, steer, brake=min(1.0, -throttle))
  else:
    vehicle.set_input(max(0.0, throttle), steer, brake=0.0)


class Vehicle:
  """带街机操控的 Bullet 射线投射车辆。"""

  STEER_CLAMP = MAX_STEER_DEG
  STEER_RATE = STEER_RATE_DEG
  STEER_SPEED_TAPER = 0.62
  MAX_ENGINE = 2800.0
  MAX_BRAKE = 180.0
  MAX_SPEED_KMH = 135.0

  WHEEL_OFFSETS = (
    (Point3(-0.78, 1.15, 0.22), True),
    (Point3(0.78, 1.15, 0.22), True),
    (Point3(-0.78, -1.15, 0.22), False),
    (Point3(0.78, -1.15, 0.22), False),
  )

  def __init__(self, render, physics: PhysicsWorld, spawn_pos: Vec3, spawn_h: float):
    self.physics = physics
    self.throttle = 0.0
    self.brake_input = 0.0
    self.steer_input = 0.0
    self.steering = 0.0
    self.max_speed_kmh = float(self.MAX_SPEED_KMH)

    self.chassis_np = self._create_chassis(render, spawn_pos, spawn_h)
    self._build_visuals()
    self.vehicle = self._create_bullet_vehicle()
    physics.attach_vehicle(self.vehicle)

  def _create_chassis(self, render, pos: Vec3, heading: float):
    shape = BulletBoxShape(Vec3(0.88, 1.75, 0.32))
    ts = TransformState.makePos(Point3(0, 0, 0.32))

    chassis = BulletRigidBodyNode("vehicle")
    chassis.addShape(shape, ts)
    chassis.setMass(820.0)
    chassis.setFriction(0.9)
    chassis.setDeactivationEnabled(False)

    np = render.attachNewNode(chassis)
    np.setPos(pos)
    np.setH(heading)
    self.physics.attach_dynamic(chassis)
    return np

  def _build_visuals(self):
    root = self.chassis_np

    body_cm = CardMaker("body")
    body_cm.setFrame(-0.9, 0.9, -1.8, 1.8)
    body = root.attachNewNode(body_cm.generate())
    body.setP(-90)
    body.setZ(0.35)
    body.setColor(0.15, 0.55, 0.95, 1)
    body.setLightOff()

    cabin_cm = CardMaker("cabin")
    cabin_cm.setFrame(-0.65, 0.65, -0.55, 0.55)
    cabin = root.attachNewNode(cabin_cm.generate())
    cabin.setP(-90)
    cabin.setZ(0.72)
    cabin.setY(-0.35)
    cabin.setColor(0.08, 0.12, 0.2, 1)
    cabin.setLightOff()

    self._wheel_nodes = []
    wheel_color = (0.1, 0.1, 0.1, 1)
    for x, y in ((-0.75, 1.15), (0.75, 1.15), (-0.75, -1.15), (0.75, -1.15)):
      wheel_cm = CardMaker(f"wheel_{x}_{y}")
      wheel_cm.setFrame(-0.24, 0.24, -0.36, 0.36)
      wheel = root.attachNewNode(wheel_cm.generate())
      wheel.setP(-90)
      wheel.setPos(x, y, 0.18)
      wheel.setColor(*wheel_color)
      wheel.setLightOff()
      self._wheel_nodes.append(wheel)

  def _create_bullet_vehicle(self):
    vehicle = BulletVehicle(self.physics.world, self.chassis_np.node())
    vehicle.setCoordinateSystem(ZUp)

    tuning = vehicle.getTuning()
    tuning.setSuspensionStiffness(38.0)
    tuning.setSuspensionCompression(2.2)
    tuning.setSuspensionDamping(2.6)
    tuning.setMaxSuspensionTravelCm(28.0)
    tuning.setFrictionSlip(2.2)
    tuning.setMaxSuspensionForce(120000.0)

    for idx, (hub, is_front) in enumerate(self.WHEEL_OFFSETS):
      wheel = vehicle.createWheel()
      wheel.setChassisConnectionPointCs(hub)
      wheel.setFrontWheel(is_front)
      wheel.setWheelDirectionCs(Vec3(0, 0, -1))
      wheel.setWheelAxleCs(Vec3(1, 0, 0))
      wheel.setWheelRadius(0.34)
      wheel.setMaxSuspensionTravelCm(28.0)
      wheel.setSuspensionStiffness(38.0)
      wheel.setWheelsDampingRelaxation(2.6)
      wheel.setWheelsDampingCompression(2.2)
      wheel.setFrictionSlip(2.4 if is_front else 2.8)
      wheel.setRollInfluence(0.12 if is_front else 0.09)
      if idx < len(self._wheel_nodes):
        wheel.setNode(self._wheel_nodes[idx].node())

    vehicle.resetSuspension()
    return vehicle

  @property
  def node(self):
    return self.chassis_np

  def set_input(self, throttle: float, steer: float, brake: float = 0.0):
    """steer 是目标前轮转角（满打方向盘的比例），不是角速度。"""
    self.throttle = max(-1.0, min(1.0, throttle))
    self.steer_input = max(-1.0, min(1.0, steer))
    self.brake_input = max(0.0, min(1.0, brake))

  def update(self, dt: float):
    self._update_steering(dt)
    self._apply_drive_forces()
    self.physics.step(dt)

  def speed_kmh(self) -> float:
    """地面绝对速度（km/h，始终 ≥ 0）。"""
    vel = self.chassis_np.node().getLinearVelocity()
    ms = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
    return ms * 3.6

  def forward_speed_kmh(self) -> float:
    """沿车体 +Y 轴的有符号速度；倒车为负。"""
    vel = self.chassis_np.node().getLinearVelocity()
    forward = self.chassis_np.getQuat().getForward()
    return (vel.x * forward.x + vel.y * forward.y + vel.z * forward.z) * 3.6

  def _speed_steer_limit(self) -> float:
    """随车速收窄可用转向角，避免满打把车打转。"""
    t = min(1.0, self.speed_kmh() / self.MAX_SPEED_KMH)
    return self.STEER_CLAMP * (1.0 - self.STEER_SPEED_TAPER * t)

  def _update_steering(self, dt: float):
    target = self.steer_input * self.STEER_CLAMP
    step = self.STEER_RATE * dt
    delta = target - self.steering
    if delta > step:
      delta = step
    elif delta < -step:
      delta = -step
    self.steering += delta

    limit = self._speed_steer_limit()
    self.steering = max(-limit, min(limit, self.steering))

    for i in range(self.vehicle.getNumWheels()):
      wheel = self.vehicle.getWheel(i)
      # Panda3D 接收度数形式的转向值，内部再换算。
      if wheel.isFrontWheel():
        self.vehicle.setSteeringValue(self.steering, i)

  def _apply_drive_forces(self):
    engine = 0.0
    brake = self.MAX_BRAKE * self.brake_input
    forward_speed = self.forward_speed_kmh()
    reverse_limit = self.max_speed_kmh * 0.35

    if self.brake_input > 0:
      # 空格刹车切断发动机，避免车轮与刹车片对打。
      engine = 0.0
    elif self.throttle > 0:
      if forward_speed < -3.0:
        # 正在倒退：W 先刹车，再向前开。
        brake = max(brake, self.MAX_BRAKE * self.throttle)
      elif forward_speed < self.max_speed_kmh:
        engine = self.MAX_ENGINE * self.throttle
    elif self.throttle < 0:
      if forward_speed > 3.0:
        # 正在前进：S/↓ 先刹车，接近停下后再倒车。
        brake = max(brake, self.MAX_BRAKE * (-self.throttle))
      elif forward_speed > -reverse_limit:
        engine = -self.MAX_ENGINE * 0.45 * (-self.throttle)

    for i in range(self.vehicle.getNumWheels()):
      self.vehicle.applyEngineForce(engine, i)
      self.vehicle.setBrake(brake, i)

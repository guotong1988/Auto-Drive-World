import math

from panda3d.core import NodePath, Vec3


# 策略观测镜头（采集 / BC / PPO / eval / main 自动驾驶共用）。
# 3D 窗口仍用 ChaseCamera，不把跟随画面送给网络。
EGO_FORWARD_M = 0.9
EGO_HEIGHT_M = 1.4
EGO_LOOK_AHEAD_M = 18.0
EGO_LOOK_Z_M = 0.8
# 横屏缓冲上的水平 FOV。60° 在 10 m 处左右约 ±5.8 m，盖住 5 m 走廊；
# 行人在 16 m 处约占 5 px 高（60×80），直立剪影对沥青对比足够。
EGO_FOV_DEG = 60.0


class ChaseCamera:
  """跟随物理底盘的第三人称相机——只给人口看窗口，不进策略。"""

  def __init__(self, camera: NodePath, target: NodePath):
    self.camera = camera
    self.target = target
    self.distance = 18.0
    self.height = 7.5
    self.smooth = 9.0
    self._pos = target.getPos() + Vec3(0, -self.distance, self.height)
    self._look = target.getPos() + Vec3(0, 0, 1.0)

  def update(self, dt: float):
    h = self.target.getH()
    rad = math.radians(h)

    behind = Vec3(
      math.sin(rad) * self.distance,
      -math.cos(rad) * self.distance,
      self.height,
    )
    pos = self.target.getPos()
    desired_pos = pos + behind
    desired_look = pos + Vec3(0, 0, 1.4)

    t = min(1.0, self.smooth * dt)
    self._pos = self._pos + (desired_pos - self._pos) * t
    self._look = self._look + (desired_look - self._look) * t

    self.camera.setPos(self._pos)
    self.camera.lookAt(self._look)


class EgoCamera:
  """车头前视：挡风玻璃高度，看向正前方行人躯干高度。

  挂在场景根下、每帧按底盘位姿写世界坐标（与 ChaseCamera 相同），
  无平滑滞后，航向与车头一致。
  """

  def __init__(self, camera: NodePath, target: NodePath):
    self.camera = camera
    self.target = target
    self.update(0.0)

  def update(self, dt: float = 0.0) -> None:
    del dt
    pos = self.target.getPos()
    rad = math.radians(self.target.getH())
    fx, fy = -math.sin(rad), math.cos(rad)
    self.camera.setPos(
      pos.x + fx * EGO_FORWARD_M,
      pos.y + fy * EGO_FORWARD_M,
      pos.z + EGO_HEIGHT_M,
    )
    self.camera.lookAt(
      pos.x + fx * EGO_LOOK_AHEAD_M,
      pos.y + fy * EGO_LOOK_AHEAD_M,
      pos.z + EGO_LOOK_Z_M,
    )

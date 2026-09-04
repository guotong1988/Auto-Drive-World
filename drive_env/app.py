from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import WindowProperties

from drive_agent.capture import (
  EgoCapture,
  OffscreenCapture,
  capture_rgb_chw,
)
from drive_agent.config import PilotNetConfig
from drive_agent.controller import SteeringController
from drive_env.camera import ChaseCamera
from drive_env.maps import MapSpec, get_map
from drive_env.minimap import MiniMap
from drive_env.physics import PhysicsWorld
from drive_env.ui_font import load_ui_font
from drive_env.vehicle import Vehicle, apply_drive
from drive_env.world import World


class RacingGame(ShowBase):
  def __init__(
    self,
    collect: bool = False,
    collect_output: str = "data/driving",
    collect_stride: int = 2,
    checkpoint: str | None = None,
    map_id: str | None = None,
  ):
    ShowBase.__init__(self)

    self.map_spec: MapSpec = get_map(map_id)

    props = WindowProperties()
    props.setTitle(f"Auto Drive World — {self.map_spec.name}")
    props.setSize(800, 600)
    self.win.requestProperties(props)

    self.setBackgroundColor(0.53, 0.75, 0.92)

    self.physics = PhysicsWorld()
    self.world = World(self.render, self.physics, self.map_spec)
    self.vehicle = Vehicle(
      self.render,
      self.physics,
      self.world.spawn_pos,
      self.world.spawn_h,
    )
    # 关掉默认 trackball，否则 ChaseCamera 写的是鼠标节点下的局部坐标。
    self.disableMouse()
    self.chase_cam = ChaseCamera(self.camera, self.vehicle.node)
    self.chase_cam.update(100.0)
    # 规划器始终来自 RuleExpert（导航指令）。Pilot-RL / PilotNet
    # 提供转向和油门。
    self.controller = SteeringController(
      checkpoint=checkpoint,
      map_spec=self.map_spec,
    )
    self.expert = self.controller.expert
    self._use_model = self.controller.uses_model
    print(f"map: {self.map_spec.id} ({self.map_spec.name}) — {self.map_spec.description}")
    if checkpoint and not self._use_model:
      print(f"checkpoint not found, falling back to rule expert: {checkpoint}")
    elif self.controller.uses_pilot_rl:
      print(f"pilot-rl autopilot ready: {checkpoint}")
    elif self._use_model:
      print(f"model autopilot ready: {checkpoint}")
    self._autopilot_max_speed_kmh = Vehicle.MAX_SPEED_KMH / 3.0
    self.minimap = MiniMap(self.aspect2d, self.map_spec)
    self._autopilot = False
    self._goal_reached = False
    self._ped_hit = False

    self._collect = collect
    self._collect_stride = max(1, collect_stride)
    self._collect_config = PilotNetConfig()
    if self._use_model:
      self._collect_config = self.controller.config
    self._collect_dir = Path(collect_output)
    self._episode_idx = 0
    self._frame_idx = 0
    self._images: list[np.ndarray] = []
    self._commands: list[int] = []
    self._speeds: list[float] = []
    self._steers: list[float] = []
    self._throttles: list[float] = []
    self._episode_saved = False
    self._frame_rgb: np.ndarray | None = None
    self._model_view_updated = False
    if self._collect:
      self._collect_dir.mkdir(parents=True, exist_ok=True)
      self._episode_idx = self._next_episode_index()

    # 模型必须吃 EgoCapture（与 collect / PPO / eval 相同：800×600、
    # 车头前视 60° FOV）。窗口是跟随相机，不能把窗口画面送给网络。
    self._model_capture: EgoCapture | None = None
    self._offscreen = None
    if self._collect or self._use_model:
      try:
        self._model_capture = EgoCapture(
          self._collect_config.image_width,
          self._collect_config.image_height,
        )
        self._model_capture.bind(self.render, self.vehicle.node)
      except Exception as exc:  # noqa: BLE001 — 降级为窗口截图
        print(f"ego capture unavailable ({exc}); falling back")
        self._model_capture = None
        self._offscreen = OffscreenCapture.create(
          self,
          self._collect_config.image_width,
          self._collect_config.image_height,
        )

    self._keys = {
      "forward": False,
      "back": False,
      "left": False,
      "right": False,
      "brake": False,
    }
    self._bind_input()
    self._setup_hud()

    self.taskMgr.add(self._update, "update")

  def _next_episode_index(self) -> int:
    existing = list(self._collect_dir.glob("episode_*.npz"))
    if not existing:
      return 0
    nums = []
    for path in existing:
      try:
        nums.append(int(path.stem.split("_")[1]))
      except (IndexError, ValueError):
        continue
    return (max(nums) + 1) if nums else 0

  def _bind_input(self):
    for key, name in (
      ("w", "forward"),
      ("arrow_up", "forward"),
      ("s", "back"),
      ("arrow_down", "back"),
      ("a", "left"),
      ("arrow_left", "left"),
      ("d", "right"),
      ("arrow_right", "right"),
      ("space", "brake"),
    ):
      self.accept(key, self._set_key, [name, True])
      self.accept(f"{key}-up", self._set_key, [name, False])

    self.accept("escape", self.userExit)
    self.accept("t", self._toggle_autopilot)
    self.accept("r", self._respawn)

  def _set_key(self, name: str, value: bool):
    self._keys[name] = value

  def _toggle_autopilot(self):
    self._autopilot = not self._autopilot
    if self._autopilot:
      self.vehicle.max_speed_kmh = self._autopilot_max_speed_kmh
      self._snap_chase_cameras()
    else:
      self.vehicle.max_speed_kmh = float(Vehicle.MAX_SPEED_KMH)

  def _snap_chase_cameras(self) -> None:
    self.chase_cam.update(100.0)
    if self._model_capture is not None and self._model_capture.ego is not None:
      self._model_capture.ego.update(0.0)

  def _clear_episode_buffer(self):
    self._images.clear()
    self._commands.clear()
    self._speeds.clear()
    self._steers.clear()
    self._throttles.clear()
    self._frame_idx = 0
    self._episode_saved = False

  def _respawn(self):
    if self._collect and self._images and not self._episode_saved:
      print(f"discarded incomplete episode ({len(self._steers)} frames)")
    self._clear_episode_buffer()

    self.vehicle.node.setPos(self.world.spawn_pos)
    self.vehicle.node.setH(self.world.spawn_h)
    self.vehicle.node.node().setLinearVelocity((0, 0, 0))
    self.vehicle.node.node().setAngularVelocity((0, 0, 0))
    self.vehicle.steering = 0.0
    self.vehicle.steer_input = 0.0
    self.vehicle.throttle = 0.0
    self._goal_reached = False
    self._ped_hit = False
    self.world.pedestrians.reset()
    self.expert.reset()
    self._snap_chase_cameras()
    if self._autopilot:
      self.vehicle.max_speed_kmh = self._autopilot_max_speed_kmh
    else:
      self.vehicle.max_speed_kmh = float(Vehicle.MAX_SPEED_KMH)

  def _setup_hud(self):
    from direct.gui.OnscreenText import OnscreenText

    hud_kw = dict(
      pos=(-1.28, 0.92),
      scale=0.045,
      fg=(1, 1, 1, 1),
      align=0,
      mayChange=True,
    )
    self._ui_font = load_ui_font(self.loader)
    if self._ui_font is not None:
      hud_kw["font"] = self._ui_font

    self.hud = OnscreenText(
      text="WASD / 方向键: 驾驶  |  Space: 刹车  |  T: 自动驾驶  |  R: 重置  |  Esc: 退出",
      **hud_kw,
    )

  def _set_overlays_hidden(self, hidden: bool):
    if hidden:
      self.hud.hide()
      self.minimap.root.hide()
    else:
      self.hud.show()
      self.minimap.root.show()

  def _capture_frame(self, dt: float) -> np.ndarray | None:
    if self._frame_rgb is not None:
      return self._frame_rgb
    if self._model_capture is not None:
      # 与 collect 相同：独立车头前视 + 立刻渲染，不用窗口跟随镜头。
      image = self._model_capture.read_rgb_chw(dt)
      self._model_view_updated = True
      self._frame_rgb = image
      return image
    if self._offscreen is not None:
      image = self._offscreen.read_rgb_chw()
      self._frame_rgb = image
      return image

    self._set_overlays_hidden(True)
    image = capture_rgb_chw(
      self,
      self._collect_config.image_width,
      self._collect_config.image_height,
    )
    self._set_overlays_hidden(False)
    self._frame_rgb = image
    return image

  def _maybe_record_frame(self, dt: float):
    if not self._collect or self._goal_reached or self._episode_saved:
      return
    if self._frame_idx % self._collect_stride != 0:
      if (
        not self._model_view_updated
        and self._model_capture is not None
        and self._model_capture.ego is not None
      ):
        self._model_capture.ego.update(dt)
      self._frame_idx += 1
      return

    image = self._capture_frame(dt)
    if image is None:
      return
    self._images.append(image)
    self._commands.append(int(self.expert.command_id))
    self._speeds.append(float(self.vehicle.speed_kmh()))
    self._steers.append(float(self.vehicle.steer_input))
    self._throttles.append(float(self.vehicle.throttle))
    self._frame_idx += 1

  def _save_successful_episode(self):
    if not self._collect or self._episode_saved or not self._images:
      return

    rel = f"episode_{self._episode_idx:04d}.npz"
    np.savez_compressed(
      self._collect_dir / rel,
      images=np.stack(self._images, axis=0),
      command=np.asarray(self._commands, dtype=np.int64),
      speed=np.asarray(self._speeds, dtype=np.float32),
      steer=np.asarray(self._steers, dtype=np.float32),
      throttle=np.asarray(self._throttles, dtype=np.float32),
    )
    print(
      f"saved {rel} ({len(self._steers)} frames @ "
      f"{self._collect_config.image_width}x{self._collect_config.image_height})"
    )
    self._episode_saved = True
    self._episode_idx += 1
    self._write_manifest()

  def _write_manifest(self):
    episodes = sorted(p.name for p in self._collect_dir.glob("episode_*.npz"))
    manifest = {
      "episodes": episodes,
      "image_shape": [
        3,
        self._collect_config.image_height,
        self._collect_config.image_width,
      ],
      "labels": ["command", "speed", "steer", "throttle"],
      "commands": ["straight", "left", "right", "stop"],
      "camera": "ego",
      "note": "only successful goal-reaching episodes are kept; images are windshield ego camera",
    }
    with (self._collect_dir / "manifest.json").open("w") as f:
      json.dump(manifest, f, indent=2)

  def _read_input(self, dt: float):
    if self._autopilot:
      if self._goal_reached or self.expert.arrived:
        self._goal_reached = True
        self.vehicle.set_input(0.0, 0.0, brake=1.0)
        return

      pos = self.vehicle.node.getPos()
      heading = self.vehicle.node.getH()
      image = self._capture_frame(dt) if self._use_model else None
      throttle, steer = self.controller.predict(
        image,
        pos.x,
        pos.y,
        heading,
        speed_kmh=self.vehicle.speed_kmh(),
        pedestrians=self.world.pedestrians.positions,
      )
      if self.expert.arrived:
        self._goal_reached = True
        self.vehicle.set_input(0.0, 0.0, brake=1.0)
      else:
        apply_drive(self.vehicle, throttle, steer)
      return

    throttle = 0.0
    if self._keys["forward"]:
      throttle += 1.0
    if self._keys["back"]:
      throttle -= 1.0

    steer = 0.0
    if self._keys["left"]:
      steer += 1.0
    if self._keys["right"]:
      steer -= 1.0

    brake = 1.0 if self._keys["brake"] else 0.0
    self.vehicle.set_input(throttle, steer, brake)

  def _distance_to_goal(self) -> float:
    pos = self.vehicle.node.getPos()
    g = self.world.goal_pos
    dx = pos.x - g.x
    dy = pos.y - g.y
    return (dx * dx + dy * dy) ** 0.5

  def _update(self, task: Task):
    dt = min(globalClock.getDt(), 0.05)
    already_done = self._goal_reached
    self._frame_rgb = None
    self._model_view_updated = False
    self._read_input(dt)
    self.vehicle.update(dt)
    self.world.pedestrians.update(dt)
    self.chase_cam.update(dt)
    will_collect = (
      self._collect and not already_done and not self._goal_reached
    )
    if (
      not self._model_view_updated
      and not will_collect
      and self._model_capture is not None
      and self._model_capture.ego is not None
    ):
      self._model_capture.ego.update(dt)

    pos = self.vehicle.node.getPos()
    if self.world.pedestrians.hit_vehicle(pos.x, pos.y):
      self._ped_hit = True

    target = None if self.expert.arrived else self.expert.target_pos
    self.minimap.update(
      pos,
      self.vehicle.node.getH(),
      target,
      pedestrians=self.world.pedestrians.positions,
    )

    # 行驶中记录跟随视角；仅在成功到达后落盘。
    if not already_done and not self._goal_reached:
      self._maybe_record_frame(dt)

    if not already_done:
      if self.world.reached_goal(pos.x, pos.y) or self.expert.arrived:
        self._goal_reached = True
      if self._goal_reached:
        self._save_successful_episode()

    if self._autopilot:
      if self.controller.uses_pilot_rl:
        mode = "Pilot-RL 自动驾驶"
      elif self._use_model:
        mode = "模型自动驾驶"
      else:
        mode = "规则自动驾驶"
    else:
      mode = "手动驾驶"
    if self._goal_reached:
      goal_txt = "已到达旗子，停车!"
    else:
      goal_txt = f"距旗子 {self._distance_to_goal():.0f}m"

    ped_n = len(self.world.pedestrians.pedestrians)
    nearest = self.world.pedestrians.nearest_distance(pos.x, pos.y)
    if self._ped_hit:
      ped_txt = "撞到行人!"
    elif nearest < 25.0:
      ped_txt = f"行人 {ped_n} 最近 {nearest:.0f}m"
    else:
      ped_txt = f"行人 {ped_n}"

    collect_txt = ""
    if self._collect:
      if self._episode_saved:
        collect_txt = f"  |  已保存 episode_{self._episode_idx - 1:04d} (R 继续)"
      else:
        collect_txt = f"  |  采集中 {len(self._steers)} 帧"

    self.hud.setText(
      f"地图 {self.map_spec.name}  |  速度 {self.vehicle.speed_kmh():.0f} km/h  |  "
      f"{mode}  |  {goal_txt}  |  {ped_txt}{collect_txt}  |  "
      "Space: 刹车  |  T: 自动驾驶  |  R: 重置  |  Esc: 退出"
    )
    return task.cont

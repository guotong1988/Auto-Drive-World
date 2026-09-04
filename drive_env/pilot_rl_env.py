"""PilotNet PPO 环境：图像+指令+速度 → 转向与油门。

默认无窗口离屏渲染；``headless=False`` 时弹出 3D 跟随相机窗口。
"""

from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from panda3d.core import NodePath, loadPrcFileData

# 必须在导入 ShowBase / 建窗口之前关掉音频库（与 collect.py 相同）。
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "notify-level-audio error")

from drive_agent.capture import EgoCapture, enable_headless_prc  # noqa: E402
from drive_agent.config import PilotNetConfig, PilotRLConfig  # noqa: E402
from drive_agent.ped_safety import (  # noqa: E402
  off_road_distance,
  ped_body_frame,
  residual_gate_from_ped,
  threat_pedestrian,
)
from drive_agent.rule_expert import RuleExpert  # noqa: E402
from drive_env.camera import ChaseCamera  # noqa: E402
from drive_env.maps import MapSpec, get_map, resolve_maps  # noqa: E402
from drive_env.minimap import MiniMap  # noqa: E402
from drive_env.pedestrian import PedestrianCrowd  # noqa: E402
from drive_env.physics import PhysicsWorld  # noqa: E402
from drive_env.terrain import build_barriers, build_terrain  # noqa: E402
from drive_env.ui_font import load_ui_font  # noqa: E402
from drive_env.vehicle import Vehicle, apply_drive  # noqa: E402


def _make_viewer() -> Any:
  """PPO 训练用的 3D 窗口。仅在 ``headless=False`` 时导入 ShowBase。"""
  from direct.showbase.ShowBase import ShowBase
  from panda3d.core import WindowProperties

  class PilotViewer(ShowBase):
    def __init__(self) -> None:
      ShowBase.__init__(self)
      self.disableMouse()
      props = WindowProperties()
      props.setTitle("Pilot PPO")
      props.setSize(800, 600)
      self.win.requestProperties(props)
      self.setBackgroundColor(0.53, 0.75, 0.92)

  return PilotViewer()


class DrivePilotEnv:
  """视觉策略输出转向与油门。RuleExpert 只提供导航指令与到达判定。

  观测为 (图像 CHW float32 [0,1], command_id, speed_kmh)。
  动作为 (steer, throttle) ∈ [-1, 1]^2；油门由策略直接执行，不再按前方行人降速。
  """

  def __init__(
    self,
    map_ids: list[str] | None = None,
    config: PilotRLConfig | None = None,
    seed: int | None = None,
    headless: bool = True,
    map_offset: int = 0,
  ):
    self.config = config or PilotRLConfig()
    self.map_ids = list(map_ids or resolve_maps("train_maps"))
    self.headless = bool(headless)
    self._map_i = int(map_offset) % max(1, len(self.map_ids))
    self._rng = np.random.default_rng(
      seed if seed is not None else self.config.seed
    )
    self.pilot_cfg = PilotNetConfig(
      image_height=self.config.image_height,
      image_width=self.config.image_width,
      num_commands=self.config.num_commands,
      throttle=self.config.throttle_prior,
    )
    self._viewer: Any | None = None
    self._chase: ChaseCamera | None = None
    self._minimap: MiniMap | None = None
    self._hud: Any | None = None
    if self.headless:
      enable_headless_prc()
    else:
      self._viewer = _make_viewer()
      self._setup_hud()
    self.capture = EgoCapture(
      self.pilot_cfg.image_width,
      self.pilot_cfg.image_height,
    )

    self.render_root: NodePath | None = None
    self.physics: PhysicsWorld | None = None
    self.vehicle: Vehicle | None = None
    self.expert: RuleExpert | None = None
    self.pedestrians: PedestrianCrowd | None = None
    self.map_spec: MapSpec | None = None

    self._command_id = 0
    self._rule_throttle = 0.0
    self._rule_steer = 0.0
    self._goal_dist_prev = 0.0
    self._station_prev = 0.0
    self._path_key_prev: object | None = None
    self._steps = 0
    self._max_steps = int(self.config.max_episode_seconds / self.config.dt)

  @property
  def action_dim(self) -> int:
    return self.config.action_dim

  def _clear_scene(self) -> None:
    if self.vehicle is not None:
      self.vehicle.node.removeNode()
    if self.pedestrians is not None:
      for ped in self.pedestrians.pedestrians:
        ped.root.removeNode()
      self.pedestrians.root.removeNode()
    if self.render_root is not None:
      self.render_root.removeNode()
    self.vehicle = None
    self.pedestrians = None
    self.physics = None
    self.expert = None
    self.render_root = None
    self._chase = None

  def _build_scene(self, map_id: str) -> None:
    self._clear_scene()
    self.map_spec = get_map(map_id)
    if self._viewer is not None:
      from panda3d.core import WindowProperties

      self.render_root = self._viewer.render.attachNewNode("pilot_rl_render")
      title = WindowProperties()
      title.setTitle(f"Pilot PPO — {self.map_spec.name}")
      self._viewer.win.requestProperties(title)
    else:
      self.render_root = NodePath("pilot_rl_render")
    self.physics = PhysicsWorld()
    build_terrain(self.render_root, self.physics, self.map_spec)
    build_barriers(self.render_root, self.physics, self.map_spec)

    sx, sy = self.map_spec.spawn_xy
    self.vehicle = Vehicle(
      self.render_root,
      self.physics,
      (sx, sy, self.map_spec.spawn_z),
      self.map_spec.spawn_heading,
    )
    self.vehicle.max_speed_kmh = (
      Vehicle.MAX_SPEED_KMH * self.config.autopilot_speed_frac
    )

    ped_seed = int(self._rng.integers(0, 2**31 - 1))
    ped_max = getattr(self.config, "rl_ped_max", None)
    if ped_max is not None and int(ped_max) <= 0:
      ped_count = 0
    else:
      # 与 World / main.py 相同：PedestrianCrowd._default_count（约 10–24）
      ped_count = None
    self.pedestrians = PedestrianCrowd(
      self.render_root, self.map_spec, count=ped_count, seed=ped_seed
    )
    self.expert = RuleExpert(
      throttle=self.pilot_cfg.throttle,
      map_spec=self.map_spec,
      route_policy="random",
      rng=random.Random(ped_seed),
    )
    self.capture.bind(self.render_root, self.vehicle.node)
    if self._viewer is not None:
      self._chase = ChaseCamera(self._viewer.camera, self.vehicle.node)
      self._chase.update(100.0)
      self._rebuild_minimap()

  @property
  def rule_steer(self) -> float:
    """上一拍规则规划器转向（与当前观测同一时刻）。"""
    return float(self._rule_steer)

  @property
  def rule_throttle(self) -> float:
    """上一拍规则规划器油门（与当前观测同一时刻）。"""
    return float(self._rule_throttle)

  def dodge_gate(self) -> float:
    """车道前方有行人时为 1，走廊清空时为 0。"""
    if self.vehicle is None or self.pedestrians is None:
      return 0.0
    pos = self.vehicle.node.getPos()
    heading = self.vehicle.node.getH()
    threat = threat_pedestrian(
      pos.x, pos.y, heading, self.pedestrians.positions, self.config
    )
    return float(
      residual_gate_from_ped(
        pos.x,
        pos.y,
        heading,
        threat,
        self.config,
        speed_kmh=self.vehicle.speed_kmh(),
      )
    )

  def _read_obs(self, cam_dt: float) -> tuple[np.ndarray, int, float]:
    assert self.vehicle is not None and self.expert is not None
    pos = self.vehicle.node.getPos()
    throttle, steer = self.expert.predict(
      pos.x, pos.y, self.vehicle.node.getH(), self.vehicle.speed_kmh()
    )
    self._rule_throttle = float(throttle)
    self._rule_steer = float(steer)
    self._command_id = int(self.expert.command_id)
    image = self.capture.read_rgb_chw(cam_dt).astype(np.float32) / 255.0
    self._pump_viewer(cam_dt)
    return image, self._command_id, float(self.vehicle.speed_kmh())

  def _setup_hud(self) -> None:
    """窗口左上角状态条；2D 叠加，不进入策略观测。"""
    from direct.gui.OnscreenText import OnscreenText

    assert self._viewer is not None
    hud_kw: dict[str, Any] = dict(
      pos=(-1.28, 0.92),
      scale=0.045,
      fg=(1, 1, 1, 1),
      align=0,
      mayChange=True,
    )
    font = load_ui_font(self._viewer.loader)
    if font is not None:
      hud_kw["font"] = font
    self._hud = OnscreenText(text="", **hud_kw)

  def _rebuild_minimap(self) -> None:
    """换图时重建小地图（道路/终点随 MapSpec 烘焙）。"""
    if self._viewer is None or self.map_spec is None:
      return
    if self._minimap is not None:
      self._minimap.root.removeNode()
    self._minimap = MiniMap(self._viewer.aspect2d, self.map_spec)

  def _update_window_overlay(self) -> None:
    if (
      self.vehicle is None
      or self.pedestrians is None
      or self.expert is None
      or self.map_spec is None
    ):
      return
    pos = self.vehicle.node.getPos()
    target = None if self.expert.arrived else self.expert.target_pos
    if self._minimap is not None:
      self._minimap.update(
        pos,
        self.vehicle.node.getH(),
        target,
        pedestrians=self.pedestrians.positions,
      )
    if self._hud is not None:
      ped_n = len(self.pedestrians.pedestrians)
      nearest = self.pedestrians.nearest_distance(pos.x, pos.y)
      gx, gy = self.map_spec.goal_xy
      goal_dist = math.hypot(pos.x - gx, pos.y - gy)
      if nearest < 25.0:
        ped_txt = f"行人 {ped_n} 最近 {nearest:.0f}m"
      else:
        ped_txt = f"行人 {ped_n}"
      self._hud.setText(
        f"地图 {self.map_spec.name}  |  "
        f"速度 {self.vehicle.speed_kmh():.0f} km/h  |  "
        f"{ped_txt}  |  距旗子 {goal_dist:.0f}m"
      )

  def _pump_viewer(self, dt: float) -> None:
    """刷新 3D 窗口；关窗后训练继续离屏跑。"""
    if self._viewer is None:
      return
    win = self._viewer.win
    if win is None or win.isClosed():
      return
    if self._chase is not None:
      self._chase.update(dt)
    self._update_window_overlay()
    self._viewer.taskMgr.step()

  def reset(self, map_id: str | None = None) -> tuple[np.ndarray, int, float]:
    if map_id is None:
      map_id = self.map_ids[self._map_i % len(self.map_ids)]
      self._map_i += 1
    self._build_scene(map_id)
    assert self.vehicle is not None and self.map_spec is not None and self.expert is not None
    self._steps = 0
    pos = self.vehicle.node.getPos()
    gx, gy = self.map_spec.goal_xy
    self._goal_dist_prev = math.hypot(pos.x - gx, pos.y - gy)
    obs = self._read_obs(cam_dt=0.0)
    station, _cte, key = self.expert.path_station(pos.x, pos.y)
    self._station_prev = station
    self._path_key_prev = key
    return obs

  def step(
    self, action: np.ndarray
  ) -> tuple[tuple[np.ndarray, int, float], float, bool, dict]:
    assert (
      self.vehicle is not None
      and self.expert is not None
      and self.pedestrians is not None
      and self.map_spec is not None
    )
    cfg = self.config
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    steer = float(np.clip(action[0], -1.0, 1.0))
    if action.shape[0] >= 2:
      model_throttle = float(np.clip(action[1], -1.0, 1.0))
    else:
      model_throttle = float(np.clip(self._rule_throttle, -1.0, 1.0))

    reward = 0.0
    done = False
    info: dict = {
      "map": self.map_spec.id,
      "hit": False,
      "success": False,
      "timeout": False,
      "waypoint": False,
      "throttle": model_throttle,
      "steer": steer,
      "cmd": self._command_id,
      "dodge_gate": self.dodge_gate(),
    }
    ticks = max(1, int(cfg.action_repeat))
    elapsed = 0
    off_done = float(cfg.offroad_done_m)
    off_term = float(cfg.reward_offroad_done)

    for _ in range(ticks):
      applied_throttle = float(np.clip(model_throttle, -1.0, 1.0))

      if self.expert.arrived:
        self.vehicle.set_input(0.0, 0.0, brake=1.0)
      else:
        apply_drive(self.vehicle, applied_throttle, steer)

      self.vehicle.update(cfg.dt)
      self.pedestrians.update(cfg.dt)
      self._steps += 1
      elapsed += 1
      self._pump_viewer(cfg.dt)

      pos = self.vehicle.node.getPos()
      heading = self.vehicle.node.getH()
      throttle, expert_steer = self.expert.predict(
        pos.x, pos.y, heading, self.vehicle.speed_kmh()
      )
      throttle = float(np.clip(throttle, -1.0, 1.0))
      self._rule_throttle = throttle
      self._rule_steer = float(expert_steer)
      self._command_id = int(self.expert.command_id)

      gx, gy = self.map_spec.goal_xy
      goal_dist = math.hypot(pos.x - gx, pos.y - gy)
      self._goal_dist_prev = goal_dist
      station, cte, path_key = self.expert.path_station(pos.x, pos.y)
      if path_key != self._path_key_prev:
        progress = 0.0
      else:
        progress = station - self._station_prev
      # path_key = (prev, target, after)；target 变了才是切到下一黄点。
      waypoint_reached = (
        self._path_key_prev is not None
        and path_key is not None
        and path_key[1] != self._path_key_prev[1]
      )
      self._station_prev = station
      self._path_key_prev = path_key

      nearest = self.pedestrians.nearest_distance(pos.x, pos.y)
      speed = self.vehicle.speed_kmh()
      hit = self.pedestrians.hit_vehicle(pos.x, pos.y)
      off = off_road_distance(pos.x, pos.y, heading, self.map_spec)
      reached = goal_dist <= self.map_spec.goal_radius or self.expert.arrived

      reward += cfg.reward_time
      reward += cfg.reward_progress * progress
      if waypoint_reached and off <= 0.5 and not hit:
        reward += float(getattr(cfg, "reward_waypoint", 0.0))
        info["waypoint"] = True
      threat_now = threat_pedestrian(
        pos.x, pos.y, heading, self.pedestrians.positions, cfg
      )
      gate = float(
        residual_gate_from_ped(
          pos.x, pos.y, heading, threat_now, cfg, speed_kmh=speed
        )
      )
      cte_clip = float(getattr(cfg, "cte_clip_m", 6.0))
      cte_scale = 1.0 - gate * (1.0 - float(getattr(cfg, "cte_dodge_scale", 0.0)))
      reward += float(getattr(cfg, "reward_cte", 0.0)) * min(abs(cte), cte_clip) * cte_scale
      if off <= 0.5:
        reward += float(getattr(cfg, "reward_on_road", 0.0))
      if nearest < cfg.proximity_range:
        closeness = 1.0 - nearest / cfg.proximity_range
        reward += cfg.reward_proximity * (closeness ** cfg.proximity_power)
      _, fwd, right = ped_body_frame(pos.x, pos.y, heading, threat_now)
      speed_ms = speed / 3.6
      dodge_lat = float(getattr(cfg, "reward_dodge_lat", 0.0))
      if (
        dodge_lat != 0.0
        and threat_now is not None
        and gate >= float(cfg.explore_gate_min)
        and fwd > 0.5
      ):
        lane = max(float(cfg.residual_lane_m), 1e-6)
        lat = min(abs(right) / lane, 1.0)
        spd_frac = min(max(speed, 0.0) / 20.0, 1.0)
        reward += dodge_lat * lat * spd_frac
      if (
        threat_now is not None
        and cfg.ttc_horizon_s > 0.0
        and fwd > 0.5
        and speed_ms > 0.5
      ):
        ttc = fwd / speed_ms
        if ttc < cfg.ttc_horizon_s:
          reward += cfg.reward_ttc * (1.0 - ttc / cfg.ttc_horizon_s)
      if speed < cfg.stall_speed_kmh and not self.expert.arrived:
        reward += cfg.reward_stall
      if off > 0.5:
        reward += cfg.reward_offroad * min(off, 5.0)

      info.update(
        {
          "goal_dist": goal_dist,
          "nearest_ped": nearest,
          "off_road": off,
          "hit": hit,
          "waypoint": bool(info.get("waypoint")),
          "speed_kmh": speed,
          "throttle": applied_throttle,
          "steer": steer,
          "cmd": self._command_id,
          "dodge_gate": gate,
        }
      )
      if hit:
        info["hit"] = True
        if bool(getattr(cfg, "terminate_on_hit", True)):
          reward += cfg.reward_collision
          done = True
          info["terminal"] = "collision"
          break
      if reached:
        reward += cfg.reward_goal
        done = True
        info["success"] = True
        info["terminal"] = "goal"
        break
      if off >= off_done and bool(getattr(cfg, "terminate_on_offroad", False)):
        reward += off_term
        done = True
        info["terminal"] = "offroad"
        break
      if self._steps >= self._max_steps:
        reward += cfg.reward_timeout
        done = True
        info["timeout"] = True
        info["terminal"] = "timeout"
        break

    if done:
      h, w = cfg.image_height, cfg.image_width
      obs = (np.zeros((3, h, w), dtype=np.float32), self._command_id, 0.0)
    else:
      obs = self._read_obs(cam_dt=cfg.dt * elapsed)
    return obs, float(reward), done, info

  def close(self) -> None:
    self._clear_scene()
    self.capture.close()
    if self._minimap is not None:
      self._minimap.root.removeNode()
      self._minimap = None
    self._hud = None
    if self._viewer is not None:
      self._viewer.destroy()
      self._viewer = None
    self._chase = None

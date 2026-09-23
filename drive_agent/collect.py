#!/usr/bin/env python3
"""采集车头前视帧、导航指令、车速、转向与油门；只保留成功到达的回合。

默认会短暂扰动执行转向（标签仍是专家纠偏），使数据覆盖偏离中线后的回正，
否则 BC 只见路过画面中央，闭环一偏就失控。``--no-disturb`` 关掉扰动，
标签更贴中线、闭环更稳，但偏离后回正样本变少。油门标签是专家限速，不叠噪声。
``--dodge`` 在规划路径前方（直行或 left/right 弯道）有人时沿路径侧移绕开
（标签与执行一致、不刹车）；撞人的局丢掉。
车角轻擦草地仍保留；冲进草地太深或累计压草过久则丢弃重采。

默认开窗口（ShowBase，窗口是跟随相机给人看）；落盘/策略一律车头前视。
``--headless`` 走离屏缓冲，不弹出 3D 窗口。
``--headless --workers N`` 多张地图并行采集：每张地图一个子进程，最多 N 个同时跑；
子进程按地图打标签落盘 ``episode_<map>_XXXX.npz``（单图多 worker 时再按分片加
``_sK``），互不撞名；全部结束后由父进程统一写 manifest。
两种模式物理步长都是 ``1/60`` s、默认 ``--stride 2``（约 30 Hz 落盘），
避免高刷新率屏幕把同一段路采成更多帧。
旧跟随相机 npz 与当前镜头不一致，须重新采集。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from panda3d.core import NodePath, Vec3, loadPrcFileData

# Panda3D 1.10.x 的 OpenAL Soft 在较新 macOS 上初始化 HRTF 时可能 SIGTRAP。
# 必须在导入 ShowBase / 建窗口之前关掉音频库（与 main.py 相同）。
loadPrcFileData("", "audio-library-name null")

from drive_agent.capture import enable_headless_prc
from drive_agent.config import PilotNetConfig
from drive_agent.ped_safety import (
  HIT_NEED_M,
  RuleDodge,
  off_road_distance,
  ped_body_frame,
)
from drive_agent.rule_expert import RuleExpert
from drive_env.maps import collect_map_choices, get_map, resolve_maps
from drive_env.camera import EGO_FOV_DEG
from drive_env.physics import PhysicsWorld
from drive_env.vehicle import Vehicle
from drive_env.world import World

HEADLESS_DT = 1.0 / 60.0


def episode_filename(idx: int, tag: str = "") -> str:
  """``episode_0003.npz``；带标签时 ``episode_<tag>_0003.npz``。"""
  if tag:
    return f"episode_{tag}_{idx:04d}.npz"
  return f"episode_{idx:04d}.npz"


def _episode_regex(tag: str = "") -> re.Pattern[str]:
  if tag:
    return re.compile(rf"^episode_{re.escape(tag)}_(\d+)\.npz$")
  return re.compile(r"^episode_(\d+)\.npz$")


def write_manifest(output_dir: Path, pilot_config: PilotNetConfig | None = None) -> int:
  """扫描目录里全部 ``episode_*.npz`` 写 manifest.json，返回 episode 数。"""
  cfg = pilot_config or PilotNetConfig()
  output_dir = Path(output_dir)
  episodes = sorted(p.name for p in output_dir.glob("episode_*.npz"))
  manifest = {
    "episodes": episodes,
    "image_shape": [3, cfg.image_height, cfg.image_width],
    "labels": ["command", "speed", "steer", "throttle"],
    "commands": ["straight", "left", "right", "stop"],
    "camera": "ego",
    "fov_deg": EGO_FOV_DEG,
    "note": (
      "successful goal-reaching episodes; images are windshield-height "
      "forward ego camera (not chase); steer labels are the expert "
      "correction (including recovery after injected steer disturbances "
      "and, if --dodge, rule steer-around-pedestrians); "
      "throttle labels are the expert speed controller; speed is km/h state"
    ),
  }
  with (output_dir / "manifest.json").open("w") as f:
    json.dump(manifest, f, indent=2)
  print(f"manifest written with {len(episodes)} episodes")
  return len(episodes)


class _CollectSession:
  """回合缓冲、DART 扰动与 npz 落盘；窗口 / 无窗口共用。"""

  def _init_session(self, args: argparse.Namespace) -> None:
    self.args = args
    # 不要用 self.config：ShowBase 已占用该名给 Panda3D ConfigVariableManager。
    self.pilot_config = PilotNetConfig()
    self._images: list[np.ndarray] = []
    self._commands: list[int] = []
    self._speeds: list[float] = []
    self._steers: list[float] = []
    self._throttles: list[float] = []
    self._ctes: list[float] = []
    self._frame_idx = 0
    self._sim_time = 0.0
    self._disturb_left = 0
    self._disturb_steer = 0.0
    self._output_dir = Path(args.output)
    self._output_dir.mkdir(parents=True, exist_ok=True)
    # 并行采集时每个子进程独占一个标签，文件名互不相撞。
    self._file_tag = str(getattr(args, "file_tag", "") or "")
    self._episode_idx = self._next_episode_index()
    self._success_count = 0
    self._target_success = int(args.episodes)
    self._dodge = RuleDodge()
    self._dodge_log_left = 0
    self._grass_seconds = 0.0
    self._max_off_road = 0.0

  def _episode_filename(self, idx: int) -> str:
    return episode_filename(idx, self._file_tag)

  def _next_episode_index(self) -> int:
    """只数本标签下已有的 episode（无标签时只数 ``episode_XXXX.npz``）。"""
    pattern = _episode_regex(self._file_tag)
    nums: list[int] = []
    for path in self._output_dir.glob("episode_*.npz"):
      m = pattern.match(path.name)
      if m is not None:
        nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 0

  def _applied_steer(self, expert_steer: float) -> float:
    """DART：短暂叠转向噪声，让下一状态离开中线；标签仍用专家转向。"""
    if self.expert.command == "stop":
      self._disturb_left = 0
      return expert_steer
    amp = float(self.args.disturb_amp)
    prob = float(self.args.disturb_prob)
    if amp <= 0.0 or prob <= 0.0:
      return expert_steer
    # 起步几秒先走稳，避免一出生就打满方向盘冲出路面。
    if self._disturb_left <= 0:
      if self._sim_time >= 3.0 and random.random() < prob:
        self._disturb_steer = random.uniform(-amp, amp)
        self._disturb_left = max(1, int(self.args.disturb_frames))
      else:
        return expert_steer
    self._disturb_left -= 1
    return max(-1.0, min(1.0, expert_steer + self._disturb_steer))

  def _expert_control(
    self,
    x: float,
    y: float,
    heading: float,
    speed_kmh: float,
  ) -> tuple[float, float]:
    """规则油门/转向；``--dodge`` 时沿规划路径（含转弯圆弧）绕行人。"""
    dodge_on = bool(getattr(self.args, "dodge", False))
    crowd = self.world.pedestrians if dodge_on else None
    return self.expert.predict(
      x,
      y,
      heading,
      speed_kmh,
      pedestrians=None if crowd is None else crowd.positions,
      ped_velocities=None if crowd is None else crowd.velocities,
      dodge=self._dodge if dodge_on else None,
    )

  def _labeled_steer(self, expert_steer: float) -> tuple[float, float]:
    """标签转向，以及实际执行的转向。

    ``--dodge`` 时规则专家已把绕行叠进 ``expert_steer``（直道与转弯同一套）；
    正在绕人则关掉 DART，标签与执行都是绕行转向。
    """
    steer = float(expert_steer)
    dodging = bool(
      getattr(self.args, "dodge", False)
      and self._dodge.last
      and self._dodge.last.get("dodging")
    )
    if dodging:
      self._maybe_log_dodge()
      self._disturb_left = 0
      return steer, steer
    return steer, self._applied_steer(steer)

  def _maybe_log_dodge(self) -> None:
    self._dodge_log_left -= 1
    if self._dodge_log_left > 0:
      return
    print(self._dodge.format_last())
    self._dodge_log_left = 24

  def _hit_detail(self) -> str:
    pos = self.vehicle.node.getPos()
    heading = float(self.vehicle.node.getH())
    best = float("inf")
    best_fwd = 0.0
    best_right = 0.0
    for px, py in self.world.pedestrians.positions:
      dist, fwd, right = ped_body_frame(pos.x, pos.y, heading, (px, py))
      if dist < best:
        best = dist
        best_fwd = fwd
        best_right = right
    dodge_s = self._dodge.format_last()
    return (
      f"dist={best:.2f}m need>{HIT_NEED_M:.2f} "
      f"hit_fwd={best_fwd:+.1f}m hit_right={best_right:+.1f}m | {dodge_s}"
    )

  def _hit_pedestrian(self) -> bool:
    pos = self.vehicle.node.getPos()
    return bool(self.world.pedestrians.hit_vehicle(pos.x, pos.y))

  def _too_much_grass(self, x: float, y: float, heading: float) -> str | None:
    """按本帧位姿更新压草统计，过多则返回丢弃原因。"""
    off = off_road_distance(x, y, heading, self.map_spec)
    self._max_off_road = max(self._max_off_road, off)
    if off > float(self.args.grass_ok_m):
      self._grass_seconds += HEADLESS_DT
    return self._grass_discard_reason()

  def _grass_discard_reason(self) -> str | None:
    """车角轻擦出路缘不算；冲太深或累计压草过久则丢弃。"""
    abort_m = float(self.args.grass_abort_m)
    max_s = float(self.args.grass_max_seconds)
    if self._max_off_road >= abort_m:
      return (
        f"too much grass (off_road={self._max_off_road:.2f}m>={abort_m:.2f}m, "
        f"on_grass={self._grass_seconds:.1f}s)"
      )
    if self._grass_seconds > max_s:
      return (
        f"too much grass (on_grass={self._grass_seconds:.1f}s>{max_s:.1f}s, "
        f"max_off={self._max_off_road:.2f}m)"
      )
    return None

  def _reset_vehicle(self) -> None:
    self.vehicle.node.setPos(self.world.spawn_pos)
    self.vehicle.node.setH(self.world.spawn_h)
    body = self.vehicle.node.node()
    body.setLinearVelocity(Vec3(0, 0, 0))
    body.setAngularVelocity(Vec3(0, 0, 0))
    self.vehicle.steering = 0.0
    self.vehicle.steer_input = 0.0
    self.vehicle.throttle = 0.0
    self.expert.reset()
    self.world.pedestrians.reset()
    self._disturb_left = 0
    self._disturb_steer = 0.0
    self._dodge.reset()
    self._dodge_log_left = 0

  def _clear_episode_buffers(self) -> None:
    self._frame_idx = 0
    self._sim_time = 0.0
    self._images.clear()
    self._commands.clear()
    self._speeds.clear()
    self._steers.clear()
    self._throttles.clear()
    self._ctes.clear()
    self._grass_seconds = 0.0
    self._max_off_road = 0.0

  def _save_episode(self) -> None:
    if not self._images:
      print(f"episode {self._episode_idx:04d} empty — skipped")
      return

    rel = self._episode_filename(self._episode_idx)
    cte = np.asarray(self._ctes, dtype=np.float32)
    np.savez_compressed(
      self._output_dir / rel,
      images=np.stack(self._images, axis=0),
      command=np.asarray(self._commands, dtype=np.int64),
      speed=np.asarray(self._speeds, dtype=np.float32),
      steer=np.asarray(self._steers, dtype=np.float32),
      throttle=np.asarray(self._throttles, dtype=np.float32),
      cte=cte,
      map_id=np.asarray(self.map_spec.id),
    )
    abs_cte = np.abs(cte) if cte.size else np.zeros(1, dtype=np.float32)
    recover = float(np.mean(abs_cte > 1.5)) if cte.size else 0.0
    n = len(self._steers)
    hz = (n / self._sim_time) if self._sim_time > 0.0 else 0.0
    print(
      f"saved {rel} ({n} frames @ "
      f"{self.pilot_config.image_width}x{self.pilot_config.image_height}, "
      f"{self._sim_time:.1f}s sim, {hz:.0f} Hz) "
      f"[{self.map_spec.id}]  "
      f"|cte|={float(abs_cte.mean()):.2f}m max={float(abs_cte.max()):.2f}m "
      f"recover(>1.5m)={recover:.0%}  "
      f"max_off={self._max_off_road:.2f}m on_grass={self._grass_seconds:.1f}s"
    )

  def _write_manifest(self) -> None:
    if bool(getattr(self.args, "skip_manifest", False)):
      # 并行子进程：manifest 由父进程在全部 worker 结束后统一写，避免并发覆盖。
      return
    write_manifest(self._output_dir, self.pilot_config)

  def _advance_episode(self, saved: bool = True) -> bool:
    """保存后清缓冲并重置车辆。返回是否还要继续采。"""
    if saved:
      self._success_count += 1
    self._episode_idx += 1
    self._clear_episode_buffers()
    self._reset_vehicle()
    if self._success_count >= self._target_success:
      self._write_manifest()
      return False
    return True


def _make_expert(args: argparse.Namespace, map_spec) -> RuleExpert:
  return RuleExpert(
    throttle=PilotNetConfig().throttle,
    map_spec=map_spec,
    route_policy="random",
    rng=random.Random(args.seed),
  )


def _make_vehicle(render, physics: PhysicsWorld, world: World) -> Vehicle:
  vehicle = Vehicle(render, physics, world.spawn_pos, world.spawn_h)
  # 与 main / drive_agent.sim_expert 的自动驾驶限速一致，使约 42 km/h 巡航可稳定达到。
  vehicle.max_speed_kmh = Vehicle.MAX_SPEED_KMH / 3.0
  return vehicle


def _run_windowed(args: argparse.Namespace) -> None:
  """开窗口采集（原 ShowBase 逻辑）。仅在未传 ``--headless`` 时导入窗口栈。"""
  loadPrcFileData("", "audio-library-name null")
  from direct.showbase.ShowBase import ShowBase
  from direct.task import Task
  from panda3d.core import WindowProperties

  from drive_agent.capture import make_policy_ego_capture
  from drive_env.camera import ChaseCamera

  class DataCollector(ShowBase, _CollectSession):
    def __init__(self, collector_args: argparse.Namespace):
      ShowBase.__init__(self)

      self.map_spec = get_map(collector_args.map)

      props = WindowProperties()
      props.setTitle(f"Auto Drive — collect ({self.map_spec.name})")
      props.setSize(800, 600)
      self.win.requestProperties(props)
      self.setBackgroundColor(0.53, 0.75, 0.92)

      self.expert = _make_expert(collector_args, self.map_spec)
      self.physics = PhysicsWorld()
      self.world = World(self.render, self.physics, self.map_spec)
      print(f"map: {self.map_spec.id} ({self.map_spec.name})")
      self.vehicle = _make_vehicle(self.render, self.physics, self.world)
      self.chase_cam = ChaseCamera(self.camera, self.vehicle.node)
      self._init_session(collector_args)
      self.capture = make_policy_ego_capture(
        self.pilot_config.image_width,
        self.pilot_config.image_height,
      )
      self.capture.bind(self.render, self.vehicle.node)

      self.taskMgr.add(self._collect, "collect")

    def _collect(self, task: Task):
      # 必须与无窗口同一固定步长。显示器 dt（尤其 ProMotion 120Hz）
      # 会让同一段路多采近一倍帧，npz 体积对不上，BC 时序也不一致。
      dt = HEADLESS_DT
      self._sim_time += dt

      if self.expert.arrived or self.world.reached_goal(
        self.vehicle.node.getPos().x,
        self.vehicle.node.getPos().y,
      ):
        self.vehicle.set_input(0.0, 0.0, brake=1.0)
        self.vehicle.update(dt)
        self.chase_cam.update(dt)
        grass = self._grass_discard_reason()
        if grass is not None:
          print(
            f"episode {self._episode_idx:04d} {grass} — discarded, retrying"
          )
          return self._after_episode(task, saved=False)
        self._save_episode()
        return self._after_episode(task)

      pos = self.vehicle.node.getPos()
      heading = self.vehicle.node.getH()
      throttle, expert_steer = self._expert_control(
        pos.x, pos.y, heading, self.vehicle.speed_kmh()
      )
      _, cte, _ = self.expert.path_station(pos.x, pos.y)
      label_steer, applied = self._labeled_steer(expert_steer)
      self.vehicle.set_input(throttle, applied)
      self.vehicle.update(dt)
      self.world.pedestrians.update(dt)
      self.chase_cam.update(dt)

      if bool(getattr(self.args, "dodge", False)) and self._hit_pedestrian():
        print(
          f"episode {self._episode_idx:04d} hit a pedestrian — discarded, retrying  "
          f"({self._hit_detail()})"
        )
        return self._after_episode(task, saved=False)

      grass = self._too_much_grass(
        self.vehicle.node.getPos().x,
        self.vehicle.node.getPos().y,
        float(self.vehicle.node.getH()),
      )
      if grass is not None:
        print(
          f"episode {self._episode_idx:04d} {grass} — discarded, retrying"
        )
        return self._after_episode(task, saved=False)

      if self._frame_idx % self.args.stride == 0:
        image = self.capture.read_rgb_chw(dt)
        self._images.append(image)
        self._commands.append(int(self.expert.command_id))
        self._speeds.append(float(self.vehicle.speed_kmh()))
        self._steers.append(float(label_steer))
        self._throttles.append(float(throttle))
        self._ctes.append(float(cte))

      self._frame_idx += 1

      if self._sim_time >= self.args.max_seconds:
        print(
          f"episode {self._episode_idx:04d} timed out after "
          f"{self._sim_time:.1f}s sim ({self._frame_idx} frames) — discarded, retrying"
        )
        return self._after_episode(task, saved=False)

      return task.cont

    def _after_episode(self, task: Task, saved: bool = True):
      if not self._advance_episode(saved=saved):
        self.capture.close()
        self.userExit()
        return task.done
      return task.cont

  DataCollector(args).run()


class HeadlessDataCollector(_CollectSession):
  """无窗口采集：``window-type none`` + 离屏车头前视，与开窗口落盘同一套镜头。"""

  def __init__(self, args: argparse.Namespace):
    from drive_agent.capture import make_policy_ego_capture

    enable_headless_prc()
    self.map_spec = get_map(args.map)
    self.expert = _make_expert(args, self.map_spec)
    self._render = NodePath("collect_headless")
    self.physics = PhysicsWorld()
    self.world = World(self._render, self.physics, self.map_spec)
    print(f"map: {self.map_spec.id} ({self.map_spec.name}) [headless]")
    self.vehicle = _make_vehicle(self._render, self.physics, self.world)
    self._init_session(args)
    self.capture = make_policy_ego_capture(
      self.pilot_config.image_width,
      self.pilot_config.image_height,
    )
    self.capture.bind(self._render, self.vehicle.node)

  def _reset_vehicle(self) -> None:
    super()._reset_vehicle()
    if self.capture.ego is not None:
      self.capture.ego.update(0.0)

  def _step(self, dt: float) -> bool:
    """推进一帧。返回 False 表示已采满目标成功回合。"""
    self._sim_time += dt

    if self.expert.arrived or self.world.reached_goal(
      self.vehicle.node.getPos().x,
      self.vehicle.node.getPos().y,
    ):
      self.vehicle.set_input(0.0, 0.0, brake=1.0)
      self.vehicle.update(dt)
      grass = self._grass_discard_reason()
      if grass is not None:
        print(
          f"episode {self._episode_idx:04d} {grass} — discarded, retrying"
        )
        return self._advance_episode(saved=False)
      self._save_episode()
      return self._advance_episode(saved=True)

    pos = self.vehicle.node.getPos()
    heading = self.vehicle.node.getH()
    throttle, expert_steer = self._expert_control(
      pos.x, pos.y, heading, self.vehicle.speed_kmh()
    )
    _, cte, _ = self.expert.path_station(pos.x, pos.y)
    label_steer, applied = self._labeled_steer(expert_steer)
    self.vehicle.set_input(throttle, applied)
    self.vehicle.update(dt)
    self.world.pedestrians.update(dt)

    if bool(getattr(self.args, "dodge", False)) and self._hit_pedestrian():
      print(
        f"episode {self._episode_idx:04d} hit a pedestrian — discarded, retrying  "
        f"({self._hit_detail()})"
      )
      return self._advance_episode(saved=False)

    grass = self._too_much_grass(
      self.vehicle.node.getPos().x,
      self.vehicle.node.getPos().y,
      float(self.vehicle.node.getH()),
    )
    if grass is not None:
      print(
        f"episode {self._episode_idx:04d} {grass} — discarded, retrying"
      )
      return self._advance_episode(saved=False)

    if self._frame_idx % self.args.stride == 0:
      image = self.capture.read_rgb_chw(dt)
      self._images.append(image)
      self._commands.append(int(self.expert.command_id))
      self._speeds.append(float(self.vehicle.speed_kmh()))
      self._steers.append(float(label_steer))
      self._throttles.append(float(throttle))
      self._ctes.append(float(cte))

    self._frame_idx += 1

    if self._sim_time >= self.args.max_seconds:
      print(
        f"episode {self._episode_idx:04d} timed out after "
        f"{self._sim_time:.1f}s sim ({self._frame_idx} frames) — discarded, retrying"
      )
      return self._advance_episode(saved=False)

    return True

  def run(self) -> None:
    try:
      while self._step(HEADLESS_DT):
        pass
    finally:
      self.capture.close()
      self._render.removeNode()


def _log_disturb(args: argparse.Namespace) -> None:
  if float(args.disturb_prob) <= 0.0 or float(args.disturb_amp) <= 0.0:
    print("disturb: off (--no-disturb or --disturb-prob 0)")
  else:
    print(
      f"disturb: on  prob={float(args.disturb_prob):.3f}  "
      f"amp={float(args.disturb_amp):.2f}  frames={int(args.disturb_frames)}"
    )
  if bool(getattr(args, "dodge", False)):
    print("dodge: on  (path-offset steer-around on straight/left/right; hit episodes discarded)")
  else:
    print("dodge: off")
  print(
    f"grass: keep if off<={float(args.grass_ok_m):.2f}m; "
    f"discard if off>={float(args.grass_abort_m):.2f}m or "
    f"on_grass>{float(args.grass_max_seconds):.1f}s"
  )


def _resolve_map_selection(text: str) -> list[str]:
  """``l_bend`` / ``train_maps`` / ``l_bend,hook,test_maps`` → 去重后的地图 id 列表。"""
  ids: list[str] = []
  for part in str(text).split(","):
    part = part.strip()
    if not part:
      continue
    for map_id in resolve_maps(part):
      if map_id not in ids:
        ids.append(map_id)
  if not ids:
    raise KeyError(f"empty map selection: {text!r}")
  return ids


def _run_one_map(args: argparse.Namespace, map_id: str) -> None:
  one = argparse.Namespace(**vars(args))
  one.map = map_id
  _log_disturb(one)
  if args.headless:
    HeadlessDataCollector(one).run()
  else:
    _run_windowed(one)


def _collect_child_cmd(
  args: argparse.Namespace,
  map_id: str,
  *,
  episodes: int | None = None,
  seed: int | None = None,
  file_tag: str | None = None,
  skip_manifest: bool = False,
) -> list[str]:
  cmd = [
    sys.executable,
    "-u",
    "-m",
    "drive_agent.collect",
    "--map",
    map_id,
    "--output",
    args.output,
    "--episodes",
    str(args.episodes if episodes is None else episodes),
    "--max-seconds",
    str(args.max_seconds),
    "--stride",
    str(args.stride),
    "--seed",
    str(args.seed if seed is None else seed),
    "--disturb-prob",
    str(args.disturb_prob),
    "--disturb-amp",
    str(args.disturb_amp),
    "--disturb-frames",
    str(args.disturb_frames),
    "--grass-ok-m",
    str(args.grass_ok_m),
    "--grass-abort-m",
    str(args.grass_abort_m),
    "--grass-max-seconds",
    str(args.grass_max_seconds),
  ]
  if args.headless:
    cmd.append("--headless")
  if args.no_disturb:
    cmd.append("--no-disturb")
  if bool(getattr(args, "dodge", False)):
    cmd.append("--dodge")
  if file_tag:
    cmd += ["--file-tag", file_tag]
  if skip_manifest:
    cmd.append("--skip-manifest")
  return cmd


def _run_map_group(args: argparse.Namespace, map_ids: list[str]) -> None:
  """每张地图用新子进程采集（ShowBase 只能启动一次）。"""
  mode = "headless" if args.headless else "windowed"
  print(
    f"collecting {args.episodes} successful episode(s) on each of "
    f"{len(map_ids)} maps ({mode}): {', '.join(map_ids)}"
  )
  _log_disturb(args)
  for map_id in map_ids:
    print(f"\n=== {map_id} ===")
    subprocess.run(_collect_child_cmd(args, map_id), check=True)


# ---------------------------------------------------------------------------
# 并行采集（仅 --headless）
# ---------------------------------------------------------------------------
class _CollectJob:
  """一个子进程要采的 (地图, 分片)。"""

  def __init__(self, map_id: str, shard: int, n_shards: int, episodes: int, seed: int):
    self.map_id = map_id
    self.shard = shard
    self.n_shards = n_shards
    self.episodes = episodes
    self.seed = seed
    self.tag = map_id if n_shards == 1 else f"{map_id}_s{shard}"
    self.returncode: int | None = None
    self.saved = 0
    self.seconds = 0.0
    self.log_path: Path | None = None


def _plan_jobs(args: argparse.Namespace, map_ids: list[str], workers: int) -> list[_CollectJob]:
  """worker 比地图多时，把每张图的 episodes 再切成分片，让所有 worker 都有活。"""
  n_maps = len(map_ids)
  episodes = int(args.episodes)
  n_shards = 1
  if workers > n_maps:
    n_shards = max(1, min(episodes, math.ceil(workers / n_maps)))
  jobs: list[_CollectJob] = []
  for map_id in map_ids:
    base, rem = divmod(episodes, n_shards)
    for shard in range(n_shards):
      n = base + (1 if shard < rem else 0)
      if n <= 0:
        continue
      # 同一张图的不同分片必须换种子，否则路线 / 扰动序列完全重复。
      seed = int(args.seed) + shard * 1009
      jobs.append(_CollectJob(map_id, shard, n_shards, n, seed))
  return jobs


def _count_tagged_episodes(output_dir: Path, tag: str) -> int:
  pattern = _episode_regex(tag)
  return sum(1 for p in output_dir.glob("episode_*.npz") if pattern.match(p.name))


def _run_job(args: argparse.Namespace, job: _CollectJob, print_lock: threading.Lock) -> _CollectJob:
  output_dir = Path(args.output)
  log_dir = output_dir / "logs"
  log_dir.mkdir(parents=True, exist_ok=True)
  job.log_path = log_dir / f"collect_{job.tag}.log"
  before = _count_tagged_episodes(output_dir, job.tag)
  cmd = _collect_child_cmd(
    args,
    job.map_id,
    episodes=job.episodes,
    seed=job.seed,
    file_tag=job.tag,
    skip_manifest=True,
  )
  env = dict(os.environ)
  env["PYTHONUNBUFFERED"] = "1"
  t0 = time.monotonic()
  with job.log_path.open("w") as log:
    log.write(" ".join(cmd) + "\n")
    proc = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
      env=env,
    )
    assert proc.stdout is not None
    prefix = f"[{job.tag}] "
    for line in proc.stdout:
      log.write(line)
      stripped = line.rstrip("\n")
      if not stripped:
        continue
      # 规则专家的规划 / 切点日志很密，并行时只回显采集进度到终端；全文在 log。
      if args.worker_verbose or not stripped.startswith(("[规划]", "[切点]", "[dodge]")):
        with print_lock:
          print(prefix + stripped, flush=True)
    job.returncode = proc.wait()
  job.seconds = time.monotonic() - t0
  job.saved = _count_tagged_episodes(output_dir, job.tag) - before
  return job


def _run_parallel(args: argparse.Namespace, map_ids: list[str], workers: int) -> None:
  """多张地图 / 多个分片同时用子进程离屏采集，最多 ``workers`` 个并发。"""
  jobs = _plan_jobs(args, map_ids, workers)
  workers = max(1, min(workers, len(jobs)))
  n_shards = jobs[0].n_shards if jobs else 1
  print(
    f"parallel collect: {len(jobs)} job(s) on {len(map_ids)} map(s) "
    f"x {n_shards} shard(s), {workers} worker(s) (headless)"
  )
  print(f"maps: {', '.join(map_ids)}")
  _log_disturb(args)
  print(f"episode files: episode_<map>[_sK]_XXXX.npz ; worker logs: {Path(args.output) / 'logs'}")

  print_lock = threading.Lock()
  t0 = time.monotonic()
  done: list[_CollectJob] = []
  with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="collect") as pool:
    futures = [pool.submit(_run_job, args, job, print_lock) for job in jobs]
    for fut in futures:
      job = fut.result()
      done.append(job)
      status = "ok" if job.returncode == 0 else f"FAILED rc={job.returncode}"
      with print_lock:
        print(
          f"=== done {job.tag}: {status}, {job.saved}/{job.episodes} episodes "
          f"in {job.seconds:.0f}s (log: {job.log_path})",
          flush=True,
        )

  total = write_manifest(Path(args.output))
  failed = [j for j in done if j.returncode != 0]
  saved = sum(j.saved for j in done)
  print(
    f"parallel collect finished: {saved} new episode(s), {total} total in manifest, "
    f"{len(failed)} failed job(s), wall {time.monotonic() - t0:.0f}s"
  )
  if failed:
    for job in failed:
      print(f"  failed: {job.tag} (rc={job.returncode}) see {job.log_path}")
    raise SystemExit(1)


def main() -> None:
  parser = argparse.ArgumentParser(description="Collect driving dataset from simulator")
  parser.add_argument(
    "--map",
    type=str,
    default="train_maps",
    help=(
      "Map id, or train_maps / test_maps / all; comma-separated list also accepted "
      f"(e.g. l_bend,hook,test_maps). Known: {', '.join(collect_map_choices())}"
    ),
  )
  parser.add_argument("--output", type=str, default="data/driving")
  parser.add_argument(
    "--episodes",
    type=int,
    default=6,
    help="Successful episodes to keep per map",
  )
  parser.add_argument(
    "--max-seconds",
    type=float,
    default=120.0,
    help="Abort and discard episode after this much simulation time",
  )
  parser.add_argument("--stride", type=int, default=2, help="Save every N simulation frames")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--no-disturb",
    action="store_true",
    help="Do not inject steer noise (cleaner centerline labels; less recovery coverage)",
  )
  parser.add_argument(
    "--disturb-prob",
    type=float,
    default=0.04,
    help="Per-frame chance to start a short steer disturbance (0 disables)",
  )
  parser.add_argument(
    "--disturb-amp",
    type=float,
    default=0.35,
    help="Steer noise amplitude added to the executed action, not the label",
  )
  parser.add_argument(
    "--disturb-frames",
    type=int,
    default=12,
    help="How many physics frames each disturbance burst lasts",
  )
  parser.add_argument(
    "--dodge",
    action="store_true",
    help=(
      "Rule steer-around when a pedestrian is ahead on the planned path "
      "(straight, left, or right; no braking). "
      "Labels include the dodge; episodes that hit a person are discarded"
    ),
  )
  parser.add_argument(
    "--grass-ok-m",
    type=float,
    default=0.5,
    help="Ignore chassis overhang past the curb up to this many metres (a little grass is ok)",
  )
  parser.add_argument(
    "--grass-abort-m",
    type=float,
    default=1.8,
    help="Discard the episode if any chassis corner is this far onto grass",
  )
  parser.add_argument(
    "--grass-max-seconds",
    type=float,
    default=2.0,
    help="Discard the episode if time with off-road > --grass-ok-m exceeds this",
  )
  parser.add_argument(
    "--headless",
    action="store_true",
    help="Offscreen ego-camera capture without opening a 3D window",
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help=(
      "Parallel headless collectors (one subprocess per map, at most N at once; "
      "0 = cpu count). With more workers than maps, each map's episodes are split "
      "into shards. Requires --headless"
    ),
  )
  parser.add_argument(
    "--worker-verbose",
    action="store_true",
    help="With --workers>1, echo every child line (planner/dodge logs) instead of progress only",
  )
  parser.add_argument(
    "--file-tag",
    type=str,
    default="",
    help="(internal) name episodes episode_<tag>_XXXX.npz so parallel children never collide",
  )
  parser.add_argument(
    "--skip-manifest",
    action="store_true",
    help="(internal) do not write manifest.json; the parent process writes it once at the end",
  )
  args = parser.parse_args()
  if args.no_disturb:
    args.disturb_prob = 0.0

  workers = int(args.workers)
  if workers <= 0:
    workers = os.cpu_count() or 1
  if workers > 1 and not args.headless:
    parser.error("--workers > 1 requires --headless (windowed ShowBase cannot run in parallel)")

  random.seed(args.seed)
  np.random.seed(args.seed)

  try:
    map_ids = _resolve_map_selection(args.map)
  except KeyError as exc:
    parser.error(str(exc))
  if workers > 1:
    _run_parallel(args, map_ids, workers)
  elif len(map_ids) == 1:
    _run_one_map(args, map_ids[0])
  else:
    _run_map_group(args, map_ids)


if __name__ == "__main__":
  main()

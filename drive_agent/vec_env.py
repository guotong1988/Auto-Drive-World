"""PPO 采集用的批量环境：单环境同进程，多环境 spawn 并行仿真。

策略留在训练进程里批量推理；worker 只跑 Panda3D 仿真，不碰 PyTorch。
OpenGL 上下文不能跨进程继承，必须用 spawn（不能 fork）。
"""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from multiprocessing.context import SpawnContext
from multiprocessing.connection import Connection
from typing import Any

import numpy as np


@dataclass
class VecObs:
  images: np.ndarray
  commands: np.ndarray
  speeds: np.ndarray
  gates: np.ndarray


def _py_info(info: dict) -> dict:
  out: dict[str, Any] = {}
  for key, value in info.items():
    if isinstance(value, np.generic):
      value = value.item()
    elif isinstance(value, np.ndarray):
      value = value.tolist()
    out[str(key)] = value
  return out


def _pack_obs(obs: tuple, gate: float) -> tuple:
  image, command, speed = obs
  return (
    np.ascontiguousarray(image, dtype=np.float32),
    int(command),
    float(speed),
    float(gate),
  )


def _stack_packs(packs: list[tuple]) -> VecObs:
  return VecObs(
    images=np.stack([p[0] for p in packs], axis=0),
    commands=np.asarray([p[1] for p in packs], dtype=np.int64),
    speeds=np.asarray([p[2] for p in packs], dtype=np.float32),
    gates=np.asarray([p[3] for p in packs], dtype=np.float32),
  )


def _pilot_env_worker(conn: Connection, kwargs: dict[str, Any]) -> None:
  """子进程入口：建离屏仿真，按主进程指令 reset/step。"""
  env = None
  try:
    from drive_agent.config import PilotRLConfig
    from drive_env.pilot_rl_env import DrivePilotEnv

    cfg = PilotRLConfig(
      **{
        k: v
        for k, v in (kwargs.get("config") or {}).items()
        if k in PilotRLConfig.__dataclass_fields__
      }
    )
    env = DrivePilotEnv(
      map_ids=list(kwargs["map_ids"]),
      config=cfg,
      seed=int(kwargs["seed"]),
      headless=True,
      map_offset=int(kwargs.get("map_offset", 0)),
    )
    conn.send(("ready",))
    while True:
      msg = conn.recv()
      op = msg[0]
      if op == "close":
        env.close()
        return
      if op == "reset":
        obs = env.reset()
        conn.send(("ok", _pack_obs(obs, env.dodge_gate())))
        continue
      if op == "step":
        obs, reward, done, info = env.step(msg[1])
        if done:
          obs = env.reset()
        conn.send(
          (
            "ok",
            _pack_obs(obs, env.dodge_gate()),
            float(reward),
            bool(done),
            _py_info(info),
          )
        )
        continue
      raise RuntimeError(f"unknown env worker op: {op!r}")
  except Exception:
    try:
      conn.send(("err", traceback.format_exc()))
    except Exception:
      pass
    if env is not None:
      try:
        env.close()
      except Exception:
        pass


def _recv(conn: Connection) -> tuple:
  msg = conn.recv()
  if not msg or msg[0] == "err":
    detail = msg[1] if msg and len(msg) > 1 else "empty worker payload"
    raise RuntimeError(f"env worker failed:\n{detail}")
  if msg[0] != "ok":
    raise RuntimeError(f"unexpected worker message: {msg[0]!r}")
  return msg[1:]


class VecDrivePilotEnv:
  """N 个 DrivePilotEnv 的同步接口。

  ``num_envs==1`` 时在本进程跑（支持 ``--window``）；
  ``num_envs>1`` 时每个环境一个 spawn 进程，step 时并行渲染。
  """

  def __init__(
    self,
    map_ids: list[str],
    config: Any,
    seed: int,
    *,
    num_envs: int = 1,
    headless: bool = True,
  ):
    self.num_envs = max(1, int(num_envs))
    self.map_ids = list(map_ids)
    self._local: Any | None = None
    self._ctx: SpawnContext | None = None
    self._procs: list[Any] = []
    self._conns: list[Connection] = []
    if self.num_envs == 1:
      from drive_env.pilot_rl_env import DrivePilotEnv

      self._local = DrivePilotEnv(
        map_ids=self.map_ids,
        config=config,
        seed=seed,
        headless=headless,
        map_offset=0,
      )
      return
    if not headless:
      raise ValueError("num_envs>1 requires headless=True (no --window)")
    import multiprocessing as mp

    self._ctx = mp.get_context("spawn")
    cfg_dict = asdict(config)
    for i in range(self.num_envs):
      parent, child = self._ctx.Pipe(duplex=True)
      proc = self._ctx.Process(
        target=_pilot_env_worker,
        args=(
          child,
          {
            "map_ids": self.map_ids,
            "config": cfg_dict,
            "seed": int(seed) + i * 1009,
            "map_offset": i,
          },
        ),
        name=f"pilot-env-{i}",
        daemon=True,
      )
      proc.start()
      child.close()
      self._procs.append(proc)
      self._conns.append(parent)
    for i, conn in enumerate(self._conns):
      try:
        msg = conn.recv()
      except EOFError as exc:
        self.close()
        raise RuntimeError(
          f"env worker {i} exited before ready (spawn failed)"
        ) from exc
      if not msg or msg[0] != "ready":
        detail = msg[1] if msg and len(msg) > 1 else "worker did not start"
        self.close()
        raise RuntimeError(f"env worker {i} failed to start:\n{detail}")
    print(f"parallel collect: {self.num_envs} env workers (spawn)")

  def reset(self) -> VecObs:
    if self._local is not None:
      obs = self._local.reset()
      return _stack_packs([_pack_obs(obs, self._local.dodge_gate())])
    for conn in self._conns:
      conn.send(("reset",))
    packs = [_recv(conn)[0] for conn in self._conns]
    return _stack_packs(packs)

  def step(
    self, actions: np.ndarray
  ) -> tuple[VecObs, np.ndarray, np.ndarray, list[dict]]:
    actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, -1)
    if self._local is not None:
      obs, reward, done, info = self._local.step(actions[0])
      if done:
        obs = self._local.reset()
      next_obs = _stack_packs([_pack_obs(obs, self._local.dodge_gate())])
      return (
        next_obs,
        np.asarray([reward], dtype=np.float32),
        np.asarray([done], dtype=np.bool_),
        [info],
      )
    for i, conn in enumerate(self._conns):
      try:
        conn.send(("step", np.ascontiguousarray(actions[i], dtype=np.float32)))
      except (BrokenPipeError, EOFError, OSError) as exc:
        raise RuntimeError(f"env worker {i} pipe closed during step") from exc
    packs: list[tuple] = []
    rewards = np.zeros(self.num_envs, dtype=np.float32)
    dones = np.zeros(self.num_envs, dtype=np.bool_)
    infos: list[dict] = []
    for i, conn in enumerate(self._conns):
      pack, reward, done, info = _recv(conn)
      packs.append(pack)
      rewards[i] = float(reward)
      dones[i] = bool(done)
      infos.append(info)
    return _stack_packs(packs), rewards, dones, infos

  def close(self) -> None:
    if self._local is not None:
      self._local.close()
      self._local = None
      return
    for conn in self._conns:
      try:
        conn.send(("close",))
      except (BrokenPipeError, EOFError, OSError):
        pass
    for proc in self._procs:
      proc.join(timeout=5.0)
      if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
    for conn in self._conns:
      try:
        conn.close()
      except Exception:
        pass
    self._procs = []
    self._conns = []


def make_pilot_envs(
  map_ids: list[str],
  config: Any,
  seed: int,
  *,
  num_envs: int = 1,
  headless: bool = True,
) -> VecDrivePilotEnv:
  n = max(1, int(num_envs))
  if n > 1 and not headless:
    print("warning: --window cannot use parallel envs; falling back to num_envs=1")
    n = 1
  return VecDrivePilotEnv(
    map_ids=map_ids,
    config=config,
    seed=seed,
    num_envs=n,
    headless=headless,
  )

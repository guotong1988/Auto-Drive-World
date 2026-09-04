#!/usr/bin/env python3
"""在 PPO 环境中对 BC / Pilot-RL 做闭环评估（不加探索噪声）。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from drive_agent.config import PilotNetConfig, PilotRLConfig
from drive_agent.model import PilotNet, is_late_fusion_state_dict, load_pilotnet_weights
from drive_agent.pilot_rl_model import load_pilot_rl
from drive_env.maps import resolve_maps
from drive_env.pilot_rl_env import DrivePilotEnv


def _as_image_tensor(
  image: np.ndarray, device: torch.device
) -> torch.Tensor:
  image_np = np.asarray(image, dtype=np.float32)
  if image_np.max() > 1.5:
    image_np = image_np / 255.0
  return torch.from_numpy(image_np).unsqueeze(0).to(device)


def _clip_action(steer: float, throttle: float) -> np.ndarray:
  return np.array(
    [
      float(max(-1.0, min(1.0, steer))),
      float(max(-1.0, min(1.0, throttle))),
    ],
    dtype=np.float32,
  )


def load_act_fn(path: Path, device: torch.device):
  """返回 (act_fn, kind)。act_fn(image, command, speed, gate) → (2,) 动作。"""
  ckpt = torch.load(path, map_location=device, weights_only=False)
  is_pilot_rl = ckpt.get("kind") == "pilot_rl" or (
    isinstance(ckpt.get("config"), dict) and "throttle_prior" in ckpt["config"]
  )
  if is_pilot_rl:
    model = load_pilot_rl(path, device=device)

    def act(
      image: np.ndarray,
      command: int,
      speed_kmh: float,
      gate: float | None = None,
    ) -> np.ndarray:
      img = _as_image_tensor(image, device)
      cmd = torch.tensor([int(command)], dtype=torch.long, device=device)
      spd = torch.tensor([float(speed_kmh)], dtype=torch.float32, device=device)
      cfg = model.config
      pin = bool(getattr(cfg, "pin_bc_empty", True))
      action, _, _ = model.act(
        img,
        cmd,
        spd,
        deterministic=True,
        pin_bc_gate=gate if pin else None,
        pin_bc_min=float(getattr(cfg, "explore_gate_min", 0.2)),
      )
      vec = action.reshape(-1)
      steer = float(vec[0].item())
      throttle = float(vec[1].item()) if vec.numel() > 1 else 0.65
      return _clip_action(steer, throttle)

    return act, "pilot_rl"

  cfg = PilotNetConfig()
  if "config" in ckpt:
    fields = set(PilotNetConfig.__dataclass_fields__)
    merged = {**PilotNetConfig().__dict__, **ckpt["config"]}
    cfg = PilotNetConfig(**{k: merged[k] for k in fields})
  model = PilotNet(cfg).to(device)
  sd = ckpt["model"]
  if is_late_fusion_state_dict(sd):
    raise RuntimeError(
      f"{path} is the old late-fusion PilotNet. "
      "Re-train: python -m drive_agent.train --data data/driving "
      "--checkpoint checkpoints/pilotnet.pt"
    )
  try:
    load_pilotnet_weights(model, sd)
  except RuntimeError as exc:
    raise RuntimeError(
      f"{path} does not match branched "
      "(image, command, speed) → command-selected (steer, throttle). "
      "Re-train BC."
    ) from exc
  model.eval()

  def act(
    image: np.ndarray,
    command: int,
    speed_kmh: float,
    gate: float | None = None,
  ) -> np.ndarray:
    img = _as_image_tensor(image, device)
    cmd = torch.tensor([int(command)], dtype=torch.long, device=device)
    spd = torch.tensor([float(speed_kmh)], dtype=torch.float32, device=device)
    out = model(img, cmd, spd).reshape(-1)
    steer = float(out[0].item())
    throttle = float(out[1].item()) if out.numel() > 1 else 0.65
    return _clip_action(steer, throttle)

  return act, "pilotnet"


def _mean(xs: list[float]) -> float:
  return float(np.mean(xs)) if xs else 0.0


def _print_episode(map_id: str, info: dict, ep_len: int, ep_return: float) -> None:
  terminal = str(info.get("terminal", "?"))
  print(
    f"[ep] {map_id:12s}  {terminal:10s}  "
    f"len={ep_len:4d}  ret={ep_return:+7.1f}  "
    f"dist={float(info.get('goal_dist', 0.0)):6.1f}  "
    f"spd={float(info.get('speed_kmh', 0.0)):5.1f}  "
    f"ped={float(info.get('nearest_ped', 0.0)):5.1f}  "
    f"off={float(info.get('off_road', 0.0)):4.2f}  "
    f"thr={float(info.get('throttle', 0.0)):+5.2f}  "
    f"str={float(info.get('steer', 0.0)):+5.2f}  "
    f"|str|={float(info.get('abs_steer', 0.0)):4.2f}  "
    f"cmd={int(info.get('cmd', -1))}  "
    f"g={float(info.get('dodge_gate', 0.0)):.2f}"
  )


def _print_summary(rows: list[dict]) -> None:
  if not rows:
    print("no episodes")
    return
  maps = list(dict.fromkeys(r["map"] for r in rows))
  print()
  print(
    f"{'map':<12} {'n':>3}  {'success':>7}  {'offroad':>7}  {'hit':>5}  "
    f"{'timeout':>7}  {'len':>5}  {'dist':>5}  {'|str|':>5}"
  )
  groups = [*((mid, [r for r in rows if r["map"] == mid]) for mid in maps)]
  if len(maps) > 1:
    groups.append(("ALL", rows))
  for name, group in groups:
    n = len(group)
    print(
      f"{name:<12} {n:3d}  "
      f"{_mean([r['success'] for r in group]):7.2f}  "
      f"{_mean([r['offroad'] for r in group]):7.2f}  "
      f"{_mean([r['hit'] for r in group]):5.2f}  "
      f"{_mean([r['timeout'] for r in group]):7.2f}  "
      f"{_mean([r['len'] for r in group]):5.0f}  "
      f"{_mean([r['dist'] for r in group]):5.0f}  "
      f"{_mean([r['abs_steer'] for r in group]):5.2f}"
    )
  counts = Counter(r["terminal"] for r in rows)
  parts = [f"{k}={v}" for k, v in counts.most_common()]
  print(f"terminals: {', '.join(parts)}")


def run_eval(
  *,
  map_selection: str,
  checkpoint: str | None = None,
  steer: str = "model",
  episodes: int = 3,
  no_peds: bool = False,
  like_main: bool = False,
  device: str = "auto",
  seed: int = 42,
) -> list[dict]:
  """无窗口闭环评估；返回每回合汇总行。"""
  cfg = PilotRLConfig()
  cfg.seed = int(seed)
  if no_peds:
    cfg.rl_ped_max = 0
  if like_main:
    cfg.terminate_on_hit = False
    cfg.terminate_on_offroad = False

  if device == "auto":
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    torch_device = torch.device(device)

  map_ids = resolve_maps(map_selection)
  env = DrivePilotEnv(map_ids=map_ids, config=cfg, seed=cfg.seed)

  act_fn = None
  kind = "expert"
  if steer == "model":
    if not checkpoint:
      env.close()
      raise ValueError("steer=model requires a checkpoint")
    path = Path(checkpoint)
    if not path.is_file():
      env.close()
      raise FileNotFoundError(f"checkpoint not found: {path}")
    act_fn, kind = load_act_fn(path, torch_device)

  term = "goal/timeout" if like_main else "goal/hit/timeout"
  print(
    f"eval | maps={map_ids} | device={torch_device} | steer={steer} "
    f"kind={kind} | peds={'off' if no_peds else 'on'} | "
    f"episodes/map={episodes} | deterministic | "
    f"offroad_done={cfg.offroad_done_m}m | terminals={term}"
  )
  if steer == "model":
    print(f"checkpoint={checkpoint}")

  rows: list[dict] = []
  try:
    for map_id in map_ids:
      for _ in range(max(1, int(episodes))):
        image, command, speed = env.reset(map_id=map_id)
        ep_return = 0.0
        ep_len = 0
        abs_steer = 0.0
        hit_any = False
        max_off = 0.0
        info: dict = {}
        while True:
          if steer == "expert":
            action = _clip_action(float(env.rule_steer), float(env.rule_throttle))
          else:
            assert act_fn is not None
            action = act_fn(image, command, speed, env.dodge_gate())
          (image, command, speed), reward, done, info = env.step(action)
          ep_return += float(reward)
          ep_len += 1
          abs_steer += abs(float(info.get("steer", action[0])))
          hit_any = hit_any or bool(info.get("hit"))
          max_off = max(max_off, float(info.get("off_road", 0.0)))
          if done:
            break
        mean_abs = abs_steer / max(ep_len, 1)
        info = dict(info)
        info["abs_steer"] = mean_abs
        info["hit"] = hit_any
        info["off_road"] = max_off
        _print_episode(map_id, info, ep_len, ep_return)
        terminal = str(info.get("terminal", "?"))
        rows.append(
          {
            "map": map_id,
            "terminal": terminal,
            "success": 1.0 if info.get("success") else 0.0,
            "offroad": 1.0 if max_off >= cfg.offroad_done_m else 0.0,
            "hit": 1.0 if info.get("hit") else 0.0,
            "timeout": 1.0 if info.get("timeout") else 0.0,
            "len": float(ep_len),
            "dist": float(info.get("goal_dist", 0.0)),
            "abs_steer": mean_abs,
            "return": ep_return,
          }
        )
  finally:
    env.close()

  _print_summary(rows)
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Deterministic closed-loop eval in DrivePilotEnv "
      "(same terminals as PPO: pedestrian hit, timeout; grass does not end)"
    )
  )
  parser.add_argument("--map", type=str, default="train_maps")
  parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/pilotnet.pt",
    help="BC PilotNet or Pilot-RL weights; ignored when --steer expert",
  )
  parser.add_argument(
    "--steer",
    choices=("model", "expert"),
    default="model",
    help="model = visual policy; expert = rule steer+throttle (env ceiling)",
  )
  parser.add_argument(
    "--episodes",
    type=int,
    default=3,
    help="episodes per map",
  )
  parser.add_argument(
    "--no-peds",
    action="store_true",
    help="spawn no pedestrians (lane-following only)",
  )
  parser.add_argument(
    "--like-main",
    action="store_true",
    help=(
      "match main.py autopilot terminals: hitting a person or clipping "
      "grass does not end the episode (only goal / timeout)"
    ),
  )
  parser.add_argument("--device", type=str, default="auto")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  run_eval(
    map_selection=args.map,
    checkpoint=args.checkpoint,
    steer=args.steer,
    episodes=args.episodes,
    no_peds=args.no_peds,
    like_main=args.like_main,
    device=args.device,
    seed=args.seed,
  )


if __name__ == "__main__":
  main()

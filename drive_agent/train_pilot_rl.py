#!/usr/bin/env python3
"""用 PPO 微调 PilotNet 转向与油门；导航指令仍由规则规划器负责。

输入是 (画面, 导航指令, 速度)。训练时从策略高斯采样探索；
空路钉完整冻结 BC。靠近行人时策略梯度可进 CNN/主干；value 不回传特征。
绕开行人/到旗子有正奖励；压草只扣分不结束，撞人仍结束回合。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from drive_agent.config import PilotRLConfig
from drive_agent.pilot_rl_model import (
  PilotActorCritic,
  load_pilot_rl,
  save_pilot_rl,
)
from drive_agent.ppo_pilot import PilotPPOTrainer
from drive_agent.vec_env import make_pilot_envs
from drive_env.maps import resolve_maps


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="PPO fine-tune of PilotNet (steer, throttle)"
  )
  parser.add_argument(
    "--map",
    type=str,
    default="ring",
    help="BC already reaches the flag here; train_maps still go off-road at 90° bends",
  )
  parser.add_argument(
    "--pilot-checkpoint",
    type=str,
    default="checkpoints/pilotnet.pt",
    help="BC PilotNet weights used to initialize the policy",
  )
  parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/pilot_rl.pt",
  )
  parser.add_argument("--total-steps", type=int, default=None)
  parser.add_argument("--rollout-steps", type=int, default=None)
  parser.add_argument("--lr", type=float, default=None)
  parser.add_argument("--device", type=str, default="auto")
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--log-every", type=int, default=1)
  parser.add_argument(
    "--num-envs",
    type=int,
    default=None,
    help="Parallel sim workers for rollouts (1=in-process; >1 spawn + batched GPU act)",
  )
  parser.add_argument(
    "--window",
    action="store_true",
    help="Open a 3D chase-camera window for watching (policy still uses ego cam; much slower)",
  )
  args = parser.parse_args()

  cfg = PilotRLConfig()
  if args.total_steps is not None:
    cfg.total_steps = args.total_steps
  if args.rollout_steps is not None:
    cfg.rollout_steps = args.rollout_steps
  if args.lr is not None:
    cfg.lr = args.lr
  if args.seed is not None:
    cfg.seed = args.seed
  if args.num_envs is not None:
    cfg.num_envs = max(1, int(args.num_envs))

  set_seed(cfg.seed)
  if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device(args.device)

  map_ids = resolve_maps(args.map)
  env = make_pilot_envs(
    map_ids=map_ids,
    config=cfg,
    seed=cfg.seed,
    num_envs=cfg.num_envs,
    headless=not args.window,
  )
  cfg.num_envs = env.num_envs
  model = PilotActorCritic(cfg)
  path = Path(args.pilot_checkpoint)
  if not path.is_file():
    raise FileNotFoundError(f"PilotNet checkpoint not found: {path}")
  model.load_pilotnet_checkpoint(path, device=device)
  trainer = PilotPPOTrainer(env, model, cfg, device=device)

  print(
    f"Pilot PPO | maps={map_ids} | device={device} | "
    f"{'window' if args.window else 'headless'} | "
    f"num_envs={env.num_envs} | "
    f"init={args.pilot_checkpoint} | action=steer+throttle | "
    f"freeze_cnn={cfg.freeze_features} trunk_lr={cfg.trunk_lr_mult} "
    f"feat_lr={cfg.features_lr_mult} bc_kl={cfg.bc_kl_coef} "
    f"explore_gate={cfg.explore_gate_min} "
    f"lead={cfg.residual_gate_near}/{cfg.residual_gate_far}m "
    f"ttc={cfg.residual_gate_ttc_near}/{cfg.residual_gate_ttc_far}s "
    f"sample=N(mu,std) pin_bc={cfg.pin_bc_empty} | "
    f"total_steps={cfg.total_steps} rollout={cfg.rollout_steps}"
    f"x{env.num_envs}={cfg.rollout_steps * env.num_envs} "
    f"term=hit/{'offroad/' if cfg.terminate_on_offroad else ''}timeout"
  )

  steps = 0
  update_i = 0
  history: list[dict] = []
  out = Path(args.checkpoint)
  best_out = out.with_name(out.stem + "_best.pt")
  best_success = -1.0
  best_return = float("-inf")
  best_step = 0
  stale_updates = 0
  stopped_early = False

  try:
    while steps < cfg.total_steps:
      trainer.set_progress(steps)
      batch = trainer.collect_rollout()
      metrics = trainer.update(batch)
      steps += cfg.rollout_steps * env.num_envs
      update_i += 1
      stats = trainer.stats()
      row = {"step": steps, **metrics, **stats}
      history.append(row)
      if update_i % max(1, args.log_every) == 0:
        print(
          f"step {steps:7d}  "
          f"return {stats['ep_return_mean']:+7.2f}  "
          f"success {stats['success_rate']:.2f}  "
          f"hit {stats['hit_rate']:.2f}  "
          f"to {stats['timeout_rate']:.2f}  "
          f"off {stats['offroad_rate']:.2f}  "
          f"n {int(min(20, stats['episodes']))}  "
          f"len {stats['ep_len_mean']:.0f}  "
          f"dist {stats['goal_dist_mean']:.0f}  "
          f"spd {stats['speed_mean']:.0f}  "
          f"thr {stats['throttle_mean']:+.2f}  "
          f"pi {metrics['policy_loss']:.3f}  "
          f"v {metrics['value_loss']:.3f}  "
          f"H {metrics['entropy']:.3f}  "
          f"kl {metrics['approx_kl']:.3f}  "
          f"bc {metrics.get('bc_kl', 0.0):.3f}  "
          f"act {metrics['abs_action']:.2f}  "
          f"early {stats['early_rate']:.2f}"
        )

      if stats["episodes"] >= 10:
        success_now = float(stats["success_rate"])
        return_now = float(stats["ep_return_mean"])
        success_gain = float(cfg.early_stop_slack)
        return_gain = float(getattr(cfg, "early_stop_return_slack", 1.0))
        improved_success = success_now > best_success + success_gain
        improved_return = return_now > best_return + return_gain
        if improved_success:
          best_success = success_now
        if improved_return:
          best_return = return_now
        if improved_success or (
          best_success >= 0.0
          and success_now + 1e-9 >= best_success
          and improved_return
        ):
          best_step = steps
          save_pilot_rl(
            best_out,
            model,
            step=steps,
            extra={
              "history": history[-50:],
              "best_success": best_success,
              "best_return": best_return,
            },
          )
        if improved_success or improved_return:
          stale_updates = 0
        elif best_success >= 0.0:
          stale_updates += 1

      if update_i % 5 == 0 or steps >= cfg.total_steps:
        save_pilot_rl(out, model, step=steps, extra={"history": history[-50:]})

      if (
        cfg.early_stop_patience > 0
        and best_success >= 0.0
        and stale_updates >= cfg.early_stop_patience
      ):
        stopped_early = True
        print(
          f"early stop @ step={steps}: no success/return gain for "
          f"{stale_updates} updates "
          f"(success {stats['success_rate']:.2f} best {best_success:.2f}, "
          f"return {stats['ep_return_mean']:+.1f} best {best_return:+.1f})"
        )
        break
  finally:
    stats_path = out.with_name(out.stem + "_stats.json")
    stats_path.write_text(json.dumps(history, indent=2))
    if best_out.is_file() and best_success >= 0.0:
      best_model = load_pilot_rl(best_out, device=device)
      save_pilot_rl(
        out,
        best_model,
        step=best_step,
        extra={
          "history": history,
          "best_success": best_success,
          "best_return": best_return,
        },
      )
    else:
      save_pilot_rl(out, model, step=steps, extra={"history": history})
    env.close()
    print(f"saved {out} and {stats_path}")
    if best_success >= 0.0:
      print(
        f"best {best_out} @ step={best_step} "
        f"success={best_success:.2f} return={best_return:+.1f}"
      )
      if stopped_early:
        print("stopped early: success and return both plateaued")


if __name__ == "__main__":
  main()

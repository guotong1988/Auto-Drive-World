#!/usr/bin/env python3
"""Auto Drive World — Panda3D 驾驶原型入口。"""

from __future__ import annotations

import argparse

# Panda3D 1.10.x 的 OpenAL Soft 在较新 macOS 上初始化 HRTF 时可能 SIGTRAP。
from panda3d.core import loadPrcFileData

loadPrcFileData("", "audio-library-name null")

from drive_env.maps import collect_map_choices, resolve_maps


def main():
  parser = argparse.ArgumentParser(description="Auto Drive World")
  parser.add_argument(
    "--map",
    type=str,
    default="crossroads",
    choices=collect_map_choices(),
    help="Driving map layout (train_maps / test_maps / all need --headless)",
  )
  parser.add_argument(
    "--collect",
    action="store_true",
    help="Record windshield ego-camera frames; save episode only after reaching the flag",
  )
  parser.add_argument(
    "--output",
    type=str,
    default="data/driving",
    help="Directory for collected episode_*.npz files",
  )
  parser.add_argument(
    "--stride",
    type=int,
    default=2,
    help="Save every N simulation frames while collecting",
  )
  parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="PilotNet / Pilot-RL checkpoint; press T to let the model drive",
  )
  parser.add_argument(
    "--headless",
    action="store_true",
    help="Closed-loop eval without a 3D window (auto-drive; same env as PPO)",
  )
  parser.add_argument(
    "--episodes",
    type=int,
    default=3,
    help="Episodes per map when using --headless",
  )
  args = parser.parse_args()

  if args.headless:
    if args.collect:
      parser.error("--collect is windowed; use drive_agent.collect --headless")
    from drive_agent.eval_pilot import run_eval

    run_eval(
      map_selection=args.map,
      checkpoint=args.checkpoint,
      steer="model" if args.checkpoint else "expert",
      episodes=args.episodes,
    )
    return

  map_ids = resolve_maps(args.map)
  if len(map_ids) != 1:
    parser.error(
      "windowed mode takes a single map; use --headless for "
      "train_maps / test_maps / all"
    )

  from drive_env.app import RacingGame

  RacingGame(
    collect=args.collect,
    collect_output=args.output,
    collect_stride=args.stride,
    checkpoint=args.checkpoint,
    map_id=map_ids[0],
  ).run()


if __name__ == "__main__":
  main()

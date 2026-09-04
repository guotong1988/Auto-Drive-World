#!/usr/bin/env python3
"""训练 PilotNet：共享 CNN，按导航指令分头预测（转向，油门）。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from drive_agent.config import PilotNetConfig
from drive_agent.dataset import DrivingDataset, load_manifest
from drive_agent.model import PilotNet


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
  model: PilotNet,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
) -> float:
  model.train()
  total_loss = 0.0
  count = 0

  for batch in loader:
    images = batch["image"].to(device)
    commands = batch["command"].to(device)
    speeds = batch["speed"].to(device)
    target = torch.cat([batch["steer"], batch["throttle"]], dim=1).to(device)
    pred = model(images, commands, speeds)
    loss = torch.nn.functional.mse_loss(pred, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    count += 1

  return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(model: PilotNet, loader: DataLoader, device: torch.device) -> float:
  model.eval()
  total_loss = 0.0
  count = 0

  for batch in loader:
    images = batch["image"].to(device)
    commands = batch["command"].to(device)
    speeds = batch["speed"].to(device)
    target = torch.cat([batch["steer"], batch["throttle"]], dim=1).to(device)
    pred = model(images, commands, speeds)
    loss = torch.nn.functional.mse_loss(pred, target)
    total_loss += loss.item()
    count += 1

  return total_loss / max(count, 1)


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Train branched PilotNet: shared CNN, command-selected "
      "(steer, throttle) heads from image + speed"
    )
  )
  parser.add_argument("--data", type=str, default="data/driving")
  parser.add_argument("--checkpoint", type=str, default="checkpoints/pilotnet.pt")
  parser.add_argument("--epochs", type=int, default=None)
  parser.add_argument("--batch-size", type=int, default=None)
  parser.add_argument("--lr", type=float, default=None)
  parser.add_argument("--device", type=str, default="auto")
  args = parser.parse_args()

  config = PilotNetConfig()
  if args.epochs is not None:
    config.epochs = args.epochs
  if args.batch_size is not None:
    config.batch_size = args.batch_size
  if args.lr is not None:
    config.lr = args.lr

  set_seed(config.seed)

  if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device(args.device)

  data_dir = Path(args.data)
  manifest = load_manifest(data_dir)
  episodes = manifest["episodes"]
  random.shuffle(episodes)

  val_count = max(1, int(len(episodes) * config.val_ratio))
  val_eps = episodes[:val_count]
  train_eps = episodes[val_count:] or val_eps

  train_ds = DrivingDataset(data_dir, train_eps)
  val_ds = DrivingDataset(data_dir, val_eps)
  print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

  train_loader = DataLoader(
    train_ds,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=device.type == "cuda",
  )
  val_loader = DataLoader(
    val_ds,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda",
  )

  model = PilotNet(config).to(device)
  print(
    f"arch: branched ({config.num_commands} command heads, "
    f"shared CNN + speed)"
  )
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay,
  )

  best_val = float("inf")
  ckpt_path = Path(args.checkpoint)
  ckpt_path.parent.mkdir(parents=True, exist_ok=True)

  for epoch in range(1, config.epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = evaluate(model, val_loader, device)
    print(f"epoch {epoch:02d}  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")

    if val_loss < best_val:
      best_val = val_loss
      torch.save(
        {
          "model": model.state_dict(),
          "config": config.__dict__,
          "val_mse": val_loss,
          "arch": "branched",
        },
        ckpt_path,
      )
      print(f"  saved checkpoint -> {ckpt_path}")

  stats = {
    "best_val_mse": best_val,
    "train_samples": len(train_ds),
    "val_samples": len(val_ds),
    "checkpoint": str(ckpt_path),
  }
  with (ckpt_path.parent / "train_stats.json").open("w") as f:
    json.dump(stats, f, indent=2)


if __name__ == "__main__":
  main()

"""采集驾驶帧的数据集加载器。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from drive_agent.commands import COMMAND_TO_ID

# 前视相机绕光心转，模拟航向误差（灭点在画面中部）。
_CAR_ANCHOR_Y = 0.50
# 规则专家在约 40 km/h、中线处：航向偏 10° → 转向约 0.122（左为正）。
_STEER_PER_HEADING_DEG = 0.0122
# 已有明显打方向的帧（绕行/过弯）不再叠航向增强，避免和绕行标签对着干。
_AUG_STEER_MAX = 0.08
_GRASS_FILL = (34, 90, 42)


def warp_heading_error(image_chw: np.ndarray, heading_err_deg: float) -> np.ndarray:
  """模拟车身航向误差：前视画面绕光心旋转。

  ``heading_err_deg`` 与 Panda 航向一致：正 = 车头偏左，路的灭点移到画面右侧。
  """
  if abs(heading_err_deg) < 1e-6:
    return image_chw
  hw = np.transpose(
    np.clip(np.asarray(image_chw) * 255.0, 0, 255).astype(np.uint8),
    (1, 2, 0),
  )
  height, width = hw.shape[:2]
  # 车头偏左时世界在画面里顺时针转 → PIL 逆时针角取负。
  warped = Image.fromarray(hw).rotate(
    -float(heading_err_deg),
    resample=Image.BILINEAR,
    center=(width / 2.0, height * _CAR_ANCHOR_Y),
    fillcolor=_GRASS_FILL,
  )
  return np.transpose(np.asarray(warped, dtype=np.float32), (2, 0, 1)) / 255.0


def recovery_steer_delta(heading_err_deg: float) -> float:
  """航向偏左（正）时应向右打（负），幅度与规则专家在直道上一致。"""
  return -float(heading_err_deg) * _STEER_PER_HEADING_DEG


class DrivingDataset(Dataset):
  """从回合 .npz 文件加载 (图像, 指令, 速度, 转向, 油门)。"""

  def __init__(self, data_dir: str | Path, episode_paths: list[str] | None = None):
    self.data_dir = Path(data_dir)
    manifest = load_manifest(self.data_dir)
    self.episode_paths = episode_paths or manifest["episodes"]
    cam = manifest.get("camera")
    if cam != "ego":
      print(
        "warning: driving data is not windshield ego-camera "
        f"(manifest camera={cam!r}). Recollect with drive_agent.collect; "
        "old chase-camera frames will not match the policy camera."
      )

    self._images: list[np.ndarray] = []
    self._commands: list[int] = []
    self._speeds: list[float] = []
    self._steers: list[float] = []
    self._throttles: list[float] = []
    self._warned_missing_command = False
    self.augment = False
    self.heading_aug_deg = 0.0

    for rel in self.episode_paths:
      episode = np.load(self.data_dir / rel)
      images = episode["images"]
      steers = episode["steer"].astype(np.float32)
      if "throttle" not in episode or "speed" not in episode:
        raise ValueError(
          f"{rel} missing 'throttle'/'speed'. Re-collect with "
          "python -m drive_agent.collect (old steer-only dumps cannot train "
          "the (steer, throttle) PilotNet)."
        )
      throttles = episode["throttle"].astype(np.float32)
      speeds = episode["speed"].astype(np.float32)
      if "command" in episode:
        commands = episode["command"].astype(np.int64)
      else:
        if not self._warned_missing_command:
          print(
            "warning: episode files missing 'command'; defaulting to straight (0). "
            "Re-collect data for conditional training."
          )
          self._warned_missing_command = True
        commands = np.full(len(steers), COMMAND_TO_ID["straight"], dtype=np.int64)

      # uint8 CHW 帧按原样存储；旧的浮点数组已经在 [0, 1]。
      if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
      else:
        images = images.astype(np.float32)
        if images.max() > 1.5:
          images = images / 255.0
      for img, cmd, speed, steer, throttle in zip(
        images, commands, speeds, steers, throttles, strict=True
      ):
        self._images.append(img)
        self._commands.append(int(cmd))
        self._speeds.append(float(speed))
        self._steers.append(float(steer))
        self._throttles.append(float(throttle))

  def __len__(self) -> int:
    return len(self._images)

  def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    image = self._images[idx]
    steer = float(self._steers[idx])
    if self.augment and abs(steer) < _AUG_STEER_MAX and self.heading_aug_deg > 0.0:
      heading_err = random.uniform(-self.heading_aug_deg, self.heading_aug_deg)
      image = warp_heading_error(image, heading_err)
      steer = max(-1.0, min(1.0, steer + recovery_steer_delta(heading_err)))
    image = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    command = torch.tensor(self._commands[idx], dtype=torch.long)
    speed = torch.tensor(self._speeds[idx], dtype=torch.float32)
    steer = torch.tensor([steer], dtype=torch.float32)
    throttle = torch.tensor([self._throttles[idx]], dtype=torch.float32)
    return {
      "image": image,
      "command": command,
      "speed": speed,
      "steer": steer,
      "throttle": throttle,
    }


def load_manifest(data_dir: str | Path) -> dict:
  path = Path(data_dir) / "manifest.json"
  with path.open() as f:
    return json.load(f)

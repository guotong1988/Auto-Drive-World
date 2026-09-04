"""PilotNet — 条件 CNN：共享视觉主干，按导航指令分头预测转向与油门。"""

from __future__ import annotations

import torch
import torch.nn as nn

from drive_agent.config import PilotNetConfig

TRUNK_HIDDEN = 10


def speed_token(speeds: torch.Tensor, scale_kmh: float) -> torch.Tensor:
  """把 km/h 车速变成 (B, 1) 特征，按巡航尺度归一并截到 [0, 2]。"""
  scale = max(float(scale_kmh), 1e-6)
  return (speeds.reshape(-1, 1).to(dtype=torch.float32) / scale).clamp(0.0, 2.0)


def select_by_command(outputs: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
  """outputs: (B, num_commands, D)；commands: (B,) → (B, D)。"""
  n_cmd = outputs.size(1)
  dim = outputs.size(-1)
  idx = (
    commands.long()
    .reshape(-1, 1, 1)
    .clamp(0, n_cmd - 1)
    .expand(-1, 1, dim)
  )
  return outputs.gather(1, idx).squeeze(1)


def build_command_trunk(in_dim: int) -> nn.Sequential:
  """视觉+车速 joint → 10 维隐层。"""
  return nn.Sequential(
    nn.Linear(in_dim, 100),
    nn.ELU(inplace=True),
    nn.Linear(100, 50),
    nn.ELU(inplace=True),
    nn.Linear(50, TRUNK_HIDDEN),
    nn.ELU(inplace=True),
  )


def build_action_out(out_dim: int) -> nn.Linear:
  """隐层 → 动作。"""
  return nn.Linear(TRUNK_HIDDEN, out_dim)


def is_late_fusion_state_dict(state: dict) -> bool:
  """旧版单头拼接指令的 checkpoint（``head.*``）。"""
  return any(key.startswith("head.") for key in state)


def _trim_ped_columns(sd: dict) -> dict:
  """最后一层若宽于隐层，丢掉行人列。"""
  out = dict(sd)
  i = 0
  while f"outs.{i}.weight" in out:
    weight = out[f"outs.{i}.weight"]
    if (
      isinstance(weight, torch.Tensor)
      and weight.ndim == 2
      and int(weight.shape[1]) > TRUNK_HIDDEN
    ):
      out[f"outs.{i}.weight"] = weight[:, :TRUNK_HIDDEN].contiguous()
    i += 1
  return out


def remap_legacy_pilotnet_state(sd: dict) -> dict:
  """把旧 Sequential ``heads.*`` 映射成 ``trunks.*`` + ``outs.*``；丢掉行人列。"""
  if any(key.startswith("heads.") for key in sd) and not any(
    key.startswith("trunks.") for key in sd
  ):
    out = {k: v for k, v in sd.items() if not k.startswith("heads.")}
    n_cmd = 0
    while f"heads.{n_cmd}.0.weight" in sd:
      n_cmd += 1
    for i in range(n_cmd):
      for idx in (0, 2, 4):
        for suffix in ("weight", "bias"):
          old = f"heads.{i}.{idx}.{suffix}"
          if old in sd:
            out[f"trunks.{i}.{idx}.{suffix}"] = sd[old]
      w_key = f"heads.{i}.6.weight"
      b_key = f"heads.{i}.6.bias"
      if w_key not in sd or b_key not in sd:
        continue
      weight = sd[w_key]
      bias = sd[b_key]
      rows = int(weight.shape[0])
      trimmed = weight.new_zeros(rows, TRUNK_HIDDEN)
      copy_cols = min(int(weight.shape[1]), TRUNK_HIDDEN)
      trimmed[:, :copy_cols] = weight[:, :copy_cols]
      out[f"outs.{i}.weight"] = trimmed
      out[f"outs.{i}.bias"] = bias
    sd = out
  return _trim_ped_columns(sd)


def load_pilotnet_weights(model: PilotNet, sd: dict) -> None:
  """加载 BC 权重；兼容旧 Sequential heads，以及带行人列的最后一层。"""
  remapped = remap_legacy_pilotnet_state(sd)
  model.load_state_dict(remapped)


class PilotNet(nn.Module):
  """RGB 帧 + 车速进共享主干；离散指令选择对应头。"""

  def __init__(self, config: PilotNetConfig | None = None) -> None:
    super().__init__()
    self.config = config or PilotNetConfig()

    # padding 保证 60×80 输入仍有可用空间尺寸（经典 PilotNet
    # 按 66×200 等更大图设计，这里不加 padding 会塌掉）。
    self.features = nn.Sequential(
      nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
      nn.ELU(inplace=True),
      nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
      nn.ELU(inplace=True),
      nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
      nn.ELU(inplace=True),
      nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
      nn.ELU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ELU(inplace=True),
      nn.Dropout(p=0.5),
    )

    with torch.no_grad():
      dummy = torch.zeros(1, 3, self.config.image_height, self.config.image_width)
      flat = int(self.features(dummy).view(1, -1).shape[1])

    self._flat = flat
    vis_in = flat + 1
    out_dim = max(1, int(self.config.action_dim))
    n_cmd = max(1, int(self.config.num_commands))
    self.trunks = nn.ModuleList(
      [build_command_trunk(vis_in) for _ in range(n_cmd)]
    )
    self.outs = nn.ModuleList(
      [build_action_out(out_dim) for _ in range(n_cmd)]
    )

  def joint_features(self, images: torch.Tensor, speeds: torch.Tensor) -> torch.Tensor:
    """共享视觉 + 车速，形状 (B, flat+1)。"""
    feat = self.features(images).view(images.size(0), -1)
    spd = speed_token(speeds, self.config.speed_scale_kmh)
    return torch.cat([feat, spd], dim=1)

  def forward(
    self,
    images: torch.Tensor,
    commands: torch.Tensor,
    speeds: torch.Tensor,
  ) -> torch.Tensor:
    """images: (B, 3, H, W)，范围 [0, 1]；commands: (B,) int64；speeds: (B,) km/h。

    返回 (B, action_dim)，默认 [steer, throttle]，范围 [-1, 1]。
    未选中的头不进入损失，因此 left 头不会被直行样本带偏。
    """
    vis = self.joint_features(images, speeds)
    h_all = torch.stack([trunk(vis) for trunk in self.trunks], dim=1)
    stacked = torch.stack(
      [torch.tanh(out(h_all[:, i])) for i, out in enumerate(self.outs)],
      dim=1,
    )
    return select_by_command(stacked, commands)

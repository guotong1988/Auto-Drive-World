"""Actor-Critic：微调 PilotNet（转向+油门）；导航指令仍由规划器负责。"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from drive_agent.config import PilotNetConfig, PilotRLConfig
from drive_agent.model import (
  TRUNK_HIDDEN,
  PilotNet,
  is_late_fusion_state_dict,
  load_pilotnet_weights,
  select_by_command,
  speed_token,
)


class PilotActorCritic(nn.Module):
  """tanh 动作上的高斯策略；CNN 主干从 BC PilotNet 初始化。"""

  LOG_STD_MIN = -5.0
  LOG_STD_MAX = -0.5

  def __init__(self, config: PilotRLConfig | None = None) -> None:
    super().__init__()
    self.config = config or PilotRLConfig()
    cfg = self.config
    pilot_cfg = PilotNetConfig(
      image_height=cfg.image_height,
      image_width=cfg.image_width,
      num_commands=cfg.num_commands,
      action_dim=cfg.action_dim,
      speed_scale_kmh=cfg.speed_scale_kmh,
    )
    base = PilotNet(pilot_cfg)
    # RL 用随机动作；关闭 Dropout，使 BC 特征保持稳定。
    for m in base.features.modules():
      if isinstance(m, nn.Dropout):
        m.p = 0.0
    self.features = base.features
    self.trunks = base.trunks
    self.bc_features = copy.deepcopy(self.features)
    self.bc_trunks = copy.deepcopy(self.trunks)
    self._freeze_bc_encoder()
    self.mus = nn.ModuleList()
    self.bc_mus = nn.ModuleList()
    n_cmd = max(1, int(cfg.num_commands))
    mu_in = TRUNK_HIDDEN
    for i in range(n_cmd):
      mu = nn.Linear(mu_in, cfg.action_dim)
      bc_mu = nn.Linear(mu_in, cfg.action_dim)
      _orthogonal_(mu.weight, gain=0.01)
      nn.init.zeros_(mu.bias)
      nn.init.zeros_(bc_mu.weight)
      nn.init.zeros_(bc_mu.bias)
      copy_linear_expand_in(mu, base.outs[i].weight, base.outs[i].bias)
      if cfg.action_dim >= 2:
        with torch.no_grad():
          if mu.bias[1].abs() < 1e-8:
            mu.bias[1] = np_clip_atanh(cfg.throttle_prior)
      with torch.no_grad():
        bc_mu.weight.copy_(mu.weight)
        bc_mu.bias.copy_(mu.bias)
      for p in bc_mu.parameters():
        p.requires_grad_(False)
      self.mus.append(mu)
      self.bc_mus.append(bc_mu)
    self.value = nn.Linear(mu_in, 1)
    steer_std = float(cfg.log_std_init_steer)
    init_std = [steer_std]
    if cfg.action_dim >= 2:
      init_std.append(float(cfg.log_std_init_throttle))
      init_std.extend([steer_std] * (cfg.action_dim - 2))
    self.log_std = nn.Parameter(torch.tensor(init_std, dtype=torch.float32))
    _orthogonal_(self.value.weight, gain=1.0)
    nn.init.zeros_(self.value.bias)

  def _explore_throttle(self) -> bool:
    return bool(getattr(self.config, "explore_throttle", False))

  def _reduce_action_terms(self, per_dim: torch.Tensor) -> torch.Tensor:
    """对角高斯：默认只把转向计入 log π / 熵，避免油门探索学成刹停。"""
    if per_dim.shape[-1] >= 2 and not self._explore_throttle():
      return per_dim[..., 0]
    return per_dim.sum(dim=-1)

  def _freeze_bc_encoder(self) -> None:
    self.bc_features.eval()
    for p in self.bc_features.parameters():
      p.requires_grad_(False)
    for trunk in self.bc_trunks:
      trunk.eval()
      for p in trunk.parameters():
        p.requires_grad_(False)

  def snapshot_bc_encoder(self) -> None:
    """把当前编码器钉成空路用的冻结 BC（加载 BC 权重之后调用）。"""
    self.bc_features.load_state_dict(self.features.state_dict())
    for dst, src in zip(self.bc_trunks, self.trunks):
      dst.load_state_dict(src.state_dict())
    self._freeze_bc_encoder()

  def _joint(self, images: torch.Tensor, speeds: torch.Tensor) -> torch.Tensor:
    if any(p.requires_grad for p in self.features.parameters()):
      feat = self.features(images).view(images.size(0), -1)
    else:
      with torch.no_grad():
        feat = self.features(images).view(images.size(0), -1)
    spd = speed_token(speeds, self.config.speed_scale_kmh)
    return torch.cat([feat, spd], dim=1)

  def _trunk_all(self, joint: torch.Tensor) -> torch.Tensor:
    return torch.stack([trunk(joint) for trunk in self.trunks], dim=1)

  def _bc_h_all(self, images: torch.Tensor, speeds: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      feat = self.bc_features(images).view(images.size(0), -1)
      spd = speed_token(speeds, self.config.speed_scale_kmh)
      joint = torch.cat([feat, spd], dim=1)
      return torch.stack([trunk(joint) for trunk in self.bc_trunks], dim=1)

  def _apply_linears(
    self,
    h_all: torch.Tensor,
    linears: nn.ModuleList,
  ) -> torch.Tensor:
    return torch.stack(
      [linear(h_all[:, i]) for i, linear in enumerate(linears)], dim=1
    )

  def encode(
    self,
    images: torch.Tensor,
    commands: torch.Tensor,
    speeds: torch.Tensor,
  ) -> torch.Tensor:
    h_all = self._trunk_all(self._joint(images, speeds))
    return select_by_command(h_all, commands)

  def forward(
    self,
    images: torch.Tensor,
    commands: torch.Tensor,
    speeds: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_all = self._trunk_all(self._joint(images, speeds))
    h = select_by_command(h_all, commands)
    mu = select_by_command(self._apply_linears(h_all, self.mus), commands)
    log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
    std = log_std.exp().expand_as(mu)
    # value 不回传 CNN/主干，避免空路出界把跟路特征冲掉。
    value = self.value(h.detach()).squeeze(-1)
    return mu, std, value

  def bc_action(
    self,
    images: torch.Tensor,
    commands: torch.Tensor,
    speeds: torch.Tensor,
  ) -> torch.Tensor:
    """完整冻结 BC 的确定性 (steer, throttle)，按指令选头。"""
    h_all = self._bc_h_all(images, speeds)
    mu = select_by_command(self._apply_linears(h_all, self.bc_mus), commands)
    return torch.tanh(mu)

  def act(
    self,
    images: torch.Tensor,
    commands: torch.Tensor,
    speeds: torch.Tensor,
    deterministic: bool = False,
    pin_bc_gate: float | torch.Tensor | None = None,
    pin_bc_min: float = 0.2,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_all = self._trunk_all(self._joint(images, speeds))
    h = select_by_command(h_all, commands)
    mu = select_by_command(self._apply_linears(h_all, self.mus), commands)
    log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
    std = log_std.exp().expand_as(mu)
    value = self.value(h.detach()).squeeze(-1)
    gate_t: torch.Tensor | None = None
    if pin_bc_gate is not None:
      gate_t = torch.as_tensor(
        pin_bc_gate, dtype=torch.float32, device=images.device
      ).reshape(-1)
      # 单环境保持原来的早退，避免多采一次随机数。
      if gate_t.numel() == 1 and float(gate_t.item()) < float(pin_bc_min):
        action = self.bc_action(images, commands, speeds)
        log_prob = torch.zeros(images.shape[0], device=images.device)
        return action, log_prob, value
    dist = Normal(mu, std)
    if deterministic:
      pre_tanh = mu
    else:
      pre_tanh = dist.sample()
      if pre_tanh.shape[-1] >= 2 and not self._explore_throttle():
        pre_tanh = pre_tanh.clone()
        pre_tanh[..., 1] = mu[..., 1]
    action = torch.tanh(pre_tanh)
    log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
    log_prob = self._reduce_action_terms(log_prob)
    if gate_t is not None and gate_t.numel() > 1:
      pin_mask = gate_t < float(pin_bc_min)
      if bool(pin_mask.any()):
        bc_action = self.bc_action(images, commands, speeds)
        action = torch.where(pin_mask.unsqueeze(-1), bc_action, action)
        log_prob = torch.where(pin_mask, torch.zeros_like(log_prob), log_prob)
    return action, log_prob, value

  def evaluate(
    self,
    images: torch.Tensor,
    commands: torch.Tensor,
    speeds: torch.Tensor,
    action: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, std, value = self.forward(images, commands, speeds)
    eps = 1e-6
    pre_tanh = torch.atanh(action.clamp(-1 + eps, 1 - eps))
    dist = Normal(mu, std)
    log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + eps)
    log_prob = self._reduce_action_terms(log_prob)
    entropy = self._reduce_action_terms(dist.entropy())
    return log_prob, entropy, value

  def load_pilotnet_checkpoint(
    self, path: str | Path, device: torch.device | str = "cpu"
  ) -> None:
    """把 BC PilotNet 权重拷进 features + 各指令 trunk；用最后一层初始化 μ。"""
    device = torch.device(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["model"]
    if is_late_fusion_state_dict(sd):
      raise RuntimeError(
        "PilotNet checkpoint is the old late-fusion single head. "
        "Re-train BC after the branched-head change: "
        "python -m drive_agent.train --data data/driving --checkpoint "
        "checkpoints/pilotnet.pt"
      )
    pilot_cfg = PilotNetConfig(
      image_height=self.config.image_height,
      image_width=self.config.image_width,
      num_commands=self.config.num_commands,
      action_dim=self.config.action_dim,
      speed_scale_kmh=self.config.speed_scale_kmh,
    )
    if "config" in ckpt:
      fields = set(PilotNetConfig.__dataclass_fields__)
      merged = {**PilotNetConfig().__dict__, **ckpt["config"]}
      pilot_cfg = PilotNetConfig(**{k: merged[k] for k in fields})
    self.config.speed_scale_kmh = float(pilot_cfg.speed_scale_kmh)
    self.config.num_commands = int(pilot_cfg.num_commands)
    base = PilotNet(pilot_cfg).to(device)
    try:
      load_pilotnet_weights(base, sd)
    except RuntimeError as exc:
      raise RuntimeError(
        "PilotNet checkpoint does not match branched "
        "(image, command, speed) → command-selected (steer, throttle). "
        "Re-collect driving data and re-train BC."
      ) from exc
    self.features.load_state_dict(base.features.state_dict())
    n_cmd = min(len(self.trunks), len(base.trunks))
    for i in range(n_cmd):
      self.trunks[i].load_state_dict(base.trunks[i].state_dict())
      last = base.outs[i]
      copy_linear_expand_in(self.mus[i], last.weight, last.bias)
      with torch.no_grad():
        self.bc_mus[i].weight.copy_(self.mus[i].weight)
        self.bc_mus[i].bias.copy_(self.mus[i].bias)
        if self.config.action_dim >= 2 and last.out_features < 2:
          self.mus[i].bias[1] = np_clip_atanh(self.config.throttle_prior)
          self.bc_mus[i].bias[1] = self.mus[i].bias[1]
    self.snapshot_bc_encoder()


def copy_linear_expand_in(
  dst: nn.Linear, src_weight: torch.Tensor, src_bias: torch.Tensor
) -> None:
  """拷贝线性层；输入维不同时按重叠列拷贝，多出来的目标列保持 0。"""
  with torch.no_grad():
    dst.weight.zero_()
    dst.bias.zero_()
    rows = min(int(dst.weight.shape[0]), int(src_weight.shape[0]))
    cols = min(int(dst.weight.shape[1]), int(src_weight.shape[1]))
    dst.weight[:rows, :cols].copy_(src_weight[:rows, :cols])
    dst.bias[:rows].copy_(src_bias[:rows])


def _orthogonal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
  """``nn.init.orthogonal_``；CPU PyTorch 未编 LAPACK 时用 numpy QR。"""
  try:
    return nn.init.orthogonal_(tensor, gain=gain)
  except RuntimeError as exc:
    msg = str(exc)
    if "LAPACK" not in msg and "geqrf" not in msg.lower():
      raise
  if tensor.ndim < 2:
    raise ValueError("orthogonal init needs a tensor with 2+ dimensions")
  rows = int(tensor.size(0))
  cols = int(tensor.numel() // rows)
  flat = np.random.randn(rows, cols)
  if rows < cols:
    q, r = np.linalg.qr(flat.T)
  else:
    q, r = np.linalg.qr(flat)
  diag = np.sign(np.diag(r))
  diag[diag == 0.0] = 1.0
  q = q * diag
  if rows < cols:
    q = q.T
  with torch.no_grad():
    tensor.copy_(
      torch.from_numpy(np.ascontiguousarray(q)).to(
        device=tensor.device, dtype=tensor.dtype
      ).view_as(tensor)
    )
    tensor.mul_(float(gain))
  return tensor


def np_clip_atanh(x: float) -> float:
  x = float(max(-0.999, min(0.999, x)))
  return 0.5 * math.log((1.0 + x) / (1.0 - x))


def save_pilot_rl(
  path: str | Path,
  model: PilotActorCritic,
  *,
  step: int = 0,
  extra: dict | None = None,
) -> None:
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "model": model.state_dict(),
    "config": model.config.__dict__,
    "step": step,
    "kind": "pilot_rl",
  }
  if extra:
    payload.update(extra)
  torch.save(payload, path)


def load_pilot_rl(
  path: str | Path,
  device: torch.device | str = "cpu",
) -> PilotActorCritic:
  device = torch.device(device)
  ckpt = torch.load(path, map_location=device, weights_only=False)
  cfg = PilotRLConfig(
    **{
      k: v
      for k, v in (ckpt.get("config") or {}).items()
      if k in PilotRLConfig.__dataclass_fields__
    }
  )
  model = PilotActorCritic(cfg).to(device)
  sd = ckpt["model"]
  if "trunk.0.weight" in sd or "mu.weight" in sd:
    raise RuntimeError(
      f"{path} is a pre-branching Pilot-RL checkpoint. "
      "Re-train BC with branched heads, then run train_pilot_rl again."
    )
  own = model.state_dict()
  compatible = {
    k: v
    for k, v in sd.items()
    if k in own and tuple(own[k].shape) == tuple(v.shape)
  }
  missing, _unexpected = model.load_state_dict(compatible, strict=False)
  for i, mu in enumerate(model.mus):
    wk = f"mus.{i}.weight"
    bk = f"mus.{i}.bias"
    if wk in sd and tuple(sd[wk].shape) != tuple(mu.weight.shape):
      copy_linear_expand_in(mu, sd[wk], sd[bk])
  for i, bc_mu in enumerate(model.bc_mus):
    wk = f"bc_mus.{i}.weight"
    bk = f"bc_mus.{i}.bias"
    if wk in sd and tuple(sd[wk].shape) != tuple(bc_mu.weight.shape):
      copy_linear_expand_in(bc_mu, sd[wk], sd[bk])
  if "value.weight" in sd and tuple(sd["value.weight"].shape) != tuple(
    model.value.weight.shape
  ):
    copy_linear_expand_in(model.value, sd["value.weight"], sd["value.bias"])
  missing_bc = [k for k in missing if k.startswith("bc_mus.")]
  if missing_bc:
    with torch.no_grad():
      for mu, bc_mu in zip(model.mus, model.bc_mus):
        bc_mu.weight.copy_(mu.weight)
        bc_mu.bias.copy_(mu.bias)
  if any(k.startswith("bc_features.") or k.startswith("bc_trunks.") for k in missing):
    model.snapshot_bc_encoder()
  else:
    model._freeze_bc_encoder()
  model.eval()
  return model

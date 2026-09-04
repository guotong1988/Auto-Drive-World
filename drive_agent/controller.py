"""运行时控制：规则导航 + PilotNet / Pilot-RL（转向与油门）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from drive_agent.commands import COMMAND_NAMES, COMMAND_TO_ID
from drive_agent.config import PilotNetConfig
from drive_agent.model import PilotNet, is_late_fusion_state_dict, load_pilotnet_weights
from drive_agent.pilot_rl_model import PilotActorCritic, load_pilot_rl
from drive_agent.ped_safety import (
  residual_gate_from_ped,
  threat_pedestrian,
)
from drive_agent.rule_expert import RuleExpert
from drive_env.maps import MapSpec


class SteeringController:
  """规则导航指令；视觉策略给转向和油门。"""

  def __init__(
    self,
    checkpoint: str | Path | None = None,
    device: str = "auto",
    fallback_to_expert: bool = True,
    throttle: float | None = None,
    map_spec: MapSpec | None = None,
  ):
    self.config = PilotNetConfig()
    self.throttle = throttle if throttle is not None else self.config.throttle
    self.fallback_to_expert = fallback_to_expert
    self.expert = RuleExpert(throttle=self.throttle, map_spec=map_spec)

    if device == "auto":
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = torch.device(device)

    self.model: PilotNet | None = None
    self.pilot_rl: PilotActorCritic | None = None
    self._log_stride = 10
    self._log_counter = 0
    self._load_checkpoint(checkpoint)

  @property
  def uses_model(self) -> bool:
    return self.model is not None or self.pilot_rl is not None

  @property
  def uses_pilot_rl(self) -> bool:
    return self.pilot_rl is not None

  def _load_checkpoint(self, checkpoint: str | Path | None) -> None:
    if checkpoint is None:
      return
    path = Path(checkpoint)
    if not path.is_file():
      return

    ckpt = torch.load(path, map_location=self.device, weights_only=False)
    if ckpt.get("kind") == "pilot_rl" or (
      isinstance(ckpt.get("config"), dict) and "throttle_prior" in ckpt["config"]
    ):
      self.pilot_rl = load_pilot_rl(path, device=self.device)
      self.config = PilotNetConfig(
        image_height=self.pilot_rl.config.image_height,
        image_width=self.pilot_rl.config.image_width,
        num_commands=self.pilot_rl.config.num_commands,
        action_dim=self.pilot_rl.config.action_dim,
        speed_scale_kmh=self.pilot_rl.config.speed_scale_kmh,
        throttle=self.pilot_rl.config.throttle_prior,
      )
      self.throttle = self.config.throttle
      return

    if "config" in ckpt:
      fields = set(PilotNetConfig.__dataclass_fields__)
      merged = {**PilotNetConfig().__dict__, **ckpt["config"]}
      self.config = PilotNetConfig(**{k: merged[k] for k in fields})

    self.model = PilotNet(self.config).to(self.device)
    sd = ckpt["model"]
    if is_late_fusion_state_dict(sd):
      raise RuntimeError(
        f"checkpoint {path} is the old late-fusion PilotNet. "
        "Re-train: python -m drive_agent.train --data data/driving "
        "--checkpoint checkpoints/pilotnet.pt"
      )
    try:
      load_pilotnet_weights(self.model, sd)
    except RuntimeError as exc:
      raise RuntimeError(
        f"checkpoint {path} does not match branched "
        "(image, command, speed) → command-selected (steer, throttle). "
        "Re-train BC."
      ) from exc
    self.model.eval()

  def _speed_tensor(self, speed_kmh: float) -> torch.Tensor:
    return torch.tensor([float(speed_kmh)], dtype=torch.float32, device=self.device)

  def _yield_cfg(self):
    if self.pilot_rl is not None:
      return self.pilot_rl.config
    return self.config

  def _dodge_gate(
    self,
    x: float,
    y: float,
    heading_deg: float,
    pedestrians: list[tuple[float, float]] | None,
    speed_kmh: float = 0.0,
  ) -> float:
    peds = pedestrians or []
    cfg = self._yield_cfg()
    ped = threat_pedestrian(x, y, heading_deg, peds, cfg)
    return float(
      residual_gate_from_ped(
        x, y, heading_deg, ped, cfg, speed_kmh=speed_kmh
      )
    )

  @torch.no_grad()
  def predict_from_image(
    self,
    image_chw: np.ndarray,
    command: int | str = "straight",
    speed_kmh: float = 0.0,
  ) -> tuple[float, float]:
    """由图像 + 指令 + 车速预测 (steer, throttle)，范围 [-1, 1]。"""
    if self.pilot_rl is not None:
      return self.predict_pilot_rl_action(
        image_chw, command, speed_kmh
      )
    if self.model is None:
      raise RuntimeError("No trained model loaded.")

    if isinstance(command, str):
      command = COMMAND_TO_ID[command]

    image_np = image_chw.astype(np.float32, copy=False)
    if image_np.max() > 1.5:
      image_np = image_np / 255.0
    image = torch.from_numpy(image_np).unsqueeze(0).to(self.device)
    cmd = torch.tensor([command], dtype=torch.long, device=self.device)
    out = self.model(
      image, cmd, self._speed_tensor(speed_kmh)
    ).reshape(-1)
    steer = float(out[0].item())
    throttle = float(out[1].item()) if out.numel() > 1 else float(self.throttle)
    return (
      float(max(-1.0, min(1.0, steer))),
      float(max(-1.0, min(1.0, throttle))),
    )

  @torch.no_grad()
  def predict_pilot_rl_action(
    self,
    image_chw: np.ndarray,
    command: int | str = "straight",
    speed_kmh: float = 0.0,
    gate: float | None = None,
  ) -> tuple[float, float]:
    """返回微调后 Pilot RL 的 (steer, throttle)；空路可钉在冻结 BC 头上。"""
    assert self.pilot_rl is not None
    if isinstance(command, str):
      command = COMMAND_TO_ID[command]
    image_np = image_chw.astype(np.float32, copy=False)
    if image_np.max() > 1.5:
      image_np = image_np / 255.0
    image = torch.from_numpy(image_np).unsqueeze(0).to(self.device)
    cmd = torch.tensor([command], dtype=torch.long, device=self.device)
    cfg = self.pilot_rl.config
    pin = bool(getattr(cfg, "pin_bc_empty", True))
    action, _, _ = self.pilot_rl.act(
      image,
      cmd,
      self._speed_tensor(speed_kmh),
      deterministic=True,
      pin_bc_gate=gate if pin else None,
      pin_bc_min=float(getattr(cfg, "explore_gate_min", 0.2)),
    )
    vec = action.reshape(-1)
    steer = float(vec[0].item())
    throttle = float(vec[1].item()) if vec.numel() > 1 else float(self.throttle)
    return (
      float(max(-1.0, min(1.0, steer))),
      float(max(-1.0, min(1.0, throttle))),
    )

  def predict(
    self,
    image_chw: np.ndarray | None,
    x: float,
    y: float,
    heading_deg: float,
    speed_kmh: float | None = None,
    pedestrians: list[tuple[float, float]] | None = None,
  ) -> tuple[float, float]:
    """返回 (油门, 转向)。

    规则规划器始终提供导航指令。Pilot-RL 或 BC PilotNet 提供转向和油门。
    """
    speed = 0.0 if speed_kmh is None else float(speed_kmh)
    expert_throttle, expert_steer = self.expert.predict(x, y, heading_deg, speed)

    if self.pilot_rl is not None and image_chw is not None:
      gate_now = self._dodge_gate(x, y, heading_deg, pedestrians, speed)
      steer, throttle = self.predict_pilot_rl_action(
        image_chw,
        self.expert.command_id,
        speed_kmh=speed,
        gate=float(gate_now),
      )
      self._log_control("pilot-rl", steer, throttle, gate_now)
      return throttle, steer

    if self.model is not None and image_chw is not None:
      steer, throttle = self.predict_from_image(
        image_chw,
        self.expert.command_id,
        speed_kmh=speed,
      )
      gate = self._dodge_gate(x, y, heading_deg, pedestrians, speed)
      self._log_control("in", steer, throttle, gate)
      return throttle, steer

    if self.fallback_to_expert:
      steer = expert_steer
      throttle = expert_throttle
    else:
      steer = 0.0
      throttle = self.throttle
    return throttle, steer

  def _log_control(
    self,
    tag: str,
    steer: float,
    throttle: float,
    gate: float,
  ) -> None:
    self._log_counter += 1
    if self._log_counter % self._log_stride != 0:
      return
    cmd_id = int(self.expert.command_id)
    cmd_name = (
      COMMAND_NAMES[cmd_id]
      if 0 <= cmd_id < len(COMMAND_NAMES)
      else str(cmd_id)
    )
    tx, ty = self.expert.target_pos
    print(
      f"[ctrl] {tag}: command={cmd_id}({cmd_name})  "
      f"nav=({tx:.0f},{ty:.0f})  |  "
      f"out: throttle={throttle:.3f} steer={steer:.4f} "
      f"gate={gate:.2f}"
    )

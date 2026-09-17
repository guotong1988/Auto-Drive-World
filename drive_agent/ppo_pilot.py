"""PilotNet 转向+油门微调的 PPO 训练器（图像 + 导航指令 + 速度 → 动作）。

训练采集从策略高斯 N(μ,σ) 采样；评测/推理取均值。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from drive_agent.config import PilotRLConfig
from drive_agent.pilot_rl_model import PilotActorCritic
from drive_agent.vec_env import VecDrivePilotEnv, VecObs


def compute_gae(
  rewards: np.ndarray,
  values: np.ndarray,
  dones: np.ndarray,
  last_value: float | np.ndarray,
  gamma: float,
  gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
  rewards = np.asarray(rewards, dtype=np.float32)
  values = np.asarray(values, dtype=np.float32)
  dones = np.asarray(dones, dtype=np.float32)
  squeeze = rewards.ndim == 1
  if squeeze:
    rewards = rewards[:, None]
    values = values[:, None]
    dones = dones[:, None]
    last_values = np.asarray([last_value], dtype=np.float32)
  else:
    last_values = np.asarray(last_value, dtype=np.float32).reshape(-1)
  t_len = rewards.shape[0]
  advantages = np.zeros_like(rewards)
  last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
  for t in reversed(range(t_len)):
    next_value = last_values if t == t_len - 1 else values[t + 1]
    next_nonterminal = 1.0 - dones[t]
    delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
    last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
    advantages[t] = last_gae
  returns = advantages + values
  if squeeze:
    return advantages[:, 0], returns[:, 0]
  return advantages, returns


def _diag_gaussian_kl(
  mu: torch.Tensor,
  std: torch.Tensor,
  mu_ref: torch.Tensor,
  std_ref: torch.Tensor,
) -> torch.Tensor:
  """KL(N(mu,std) || N(mu_ref,std_ref))，按 batch 项与动作维度计算。"""
  var = std.pow(2)
  var_ref = std_ref.pow(2).clamp_min(1e-8)
  return (
    torch.log(std_ref.clamp_min(1e-8) / std.clamp_min(1e-8))
    + (var + (mu - mu_ref).pow(2)) / (2.0 * var_ref)
    - 0.5
  )


@dataclass
class PilotRolloutBatch:
  images: torch.Tensor
  commands: torch.Tensor
  actions: torch.Tensor
  log_probs: torch.Tensor
  rewards: torch.Tensor
  dones: torch.Tensor
  values: torch.Tensor
  advantages: torch.Tensor
  returns: torch.Tensor
  gates: torch.Tensor
  speeds: torch.Tensor
  # 采集时该步是否钉在冻结 BC 上（log_prob=0）。钉住的样本 ratio 不是合法
  # 重要性比，策略损失必须靠它把这类样本权重清零——仅靠 gate 阈值区分不了
  # 迟滞带内「放开采样」与「仍在钉」的样本。
  pinned: torch.Tensor


class PilotPPOTrainer:
  def __init__(
    self,
    env: VecDrivePilotEnv,
    model: PilotActorCritic,
    config: PilotRLConfig | None = None,
    device: str | torch.device = "cpu",
  ):
    self.env = env
    self.model = model
    self.config = config or model.config
    self.device = torch.device(device)
    self.model.to(self.device)
    self.num_envs = max(1, int(getattr(env, "num_envs", 1)))
    self._apply_freeze()
    self.optimizer = torch.optim.Adam(self._param_groups())
    self.bc_ref: PilotActorCritic | None = None
    bc_kl_max = max(
      float(self.config.bc_kl_coef),
      float(getattr(self.config, "bc_kl_coef_end", 0.0)),
    )
    if bc_kl_max > 0.0:
      self.bc_ref = copy.deepcopy(self.model).to(self.device)
      self.bc_ref.eval()
      for p in self.bc_ref.parameters():
        p.requires_grad_(False)
    self._obs: VecObs = self.env.reset()
    self.ep_return = np.zeros(self.num_envs, dtype=np.float64)
    self.ep_len = np.zeros(self.num_envs, dtype=np.int32)
    self.ep_max_off = np.zeros(self.num_envs, dtype=np.float64)
    # 钉 BC 的迟滞状态（默认空路钉 BC）：gate 升过 explore_gate_on 才放开
    # 采样，降下 explore_gate_off 才钉回，避免阈值附近来回跳。
    self._pin_bc = np.ones(self.num_envs, dtype=np.bool_)
    self.recent_returns: list[float] = []
    self.recent_success: list[float] = []
    self.recent_hits: list[float] = []
    self.recent_timeouts: list[float] = []
    self.recent_offroads: list[float] = []
    self.recent_lens: list[float] = []
    self.recent_goal_dists: list[float] = []
    self.recent_speeds: list[float] = []
    self.recent_peds: list[float] = []
    self.recent_throttles: list[float] = []
    self.recent_early: list[float] = []
    self._ent_coef = float(self.config.ent_coef)
    self._bc_kl_coef = float(self.config.bc_kl_coef)

  def _apply_freeze(self) -> None:
    self.model._freeze_bc_encoder()
    if self.config.freeze_features:
      for p in self.model.features.parameters():
        p.requires_grad_(False)
      self.model.features.eval()
    if float(self.config.trunk_lr_mult) <= 0.0:
      for trunk in self.model.trunks:
        for p in trunk.parameters():
          p.requires_grad_(False)
        trunk.eval()

  def _param_groups(self) -> list[dict]:
    cfg = self.config
    groups = [
      {
        "params": [p for mu in self.model.mus for p in mu.parameters()]
        + list(self.model.value.parameters())
        + [self.model.log_std],
        "lr": cfg.lr,
        "base_lr": cfg.lr,
      },
    ]
    if float(cfg.trunk_lr_mult) > 0.0:
      groups.append(
        {
          "params": [p for t in self.model.trunks for p in t.parameters()],
          "lr": cfg.lr * float(cfg.trunk_lr_mult),
          "base_lr": cfg.lr * float(cfg.trunk_lr_mult),
        }
      )
    if (not cfg.freeze_features) and float(cfg.features_lr_mult) > 0.0:
      feat_lr = cfg.lr * float(cfg.features_lr_mult)
      groups.append(
        {
          "params": list(self.model.features.parameters()),
          "lr": feat_lr,
          "base_lr": feat_lr,
        }
      )
    return groups

  def _set_train_mode(self) -> None:
    self.model.train()
    self.model._freeze_bc_encoder()
    if self.config.freeze_features:
      self.model.features.eval()
    if float(self.config.trunk_lr_mult) <= 0.0:
      for trunk in self.model.trunks:
        trunk.eval()

  def set_progress(self, steps: int) -> None:
    cfg = self.config
    total = max(1, int(cfg.total_steps))
    t = min(1.0, float(steps) / float(total))
    scale = 1.0 + (cfg.lr_end / max(cfg.lr, 1e-12) - 1.0) * t
    for group in self.optimizer.param_groups:
      group["lr"] = float(group["base_lr"]) * scale
    self._ent_coef = cfg.ent_coef + (cfg.ent_coef_end - cfg.ent_coef) * t
    bc_kl_end = float(getattr(cfg, "bc_kl_coef_end", cfg.bc_kl_coef))
    self._bc_kl_coef = float(cfg.bc_kl_coef) + (bc_kl_end - float(cfg.bc_kl_coef)) * t

  def _policy_mask(self, gates: Any, speeds: Any, pinned: Any) -> Any:
    """进策略损失的样本权重 [0, 1]；``None`` = 不过滤。numpy 与 torch 通用。

    软门控：权重在 policy_gate_min ± policy_gate_soft 之间从 0 线性升到 1，
    避免 gate 在阈值附近抖动时相似样本时而进损失时而不进。钉 BC 的样本
    （log_prob=0）权重恒为 0。advantage 归一化和 ``update`` 的加权必须用
    同一个权重，否则归一化的统计量来自一批不参与损失的样本。
    """
    cfg = self.config
    gate_min = float(cfg.policy_gate_min)
    if gate_min <= 0.0:
      return None
    speed_min = float(getattr(cfg, "policy_min_speed_kmh", 6.0))
    soft = max(float(getattr(cfg, "policy_gate_soft", 0.1)), 1e-6)
    w = (gates - (gate_min - soft)) / (2.0 * soft)
    if isinstance(w, np.ndarray):
      w = np.clip(w, 0.0, 1.0)
    else:
      w = w.clamp(0.0, 1.0)
    return w * (speeds >= speed_min) * (1.0 - pinned)

  def _record_episode(self, env_i: int, info: dict) -> None:
    cfg = self.config
    terminal = str(info.get("terminal", "?"))
    ep_return = float(self.ep_return[env_i])
    ep_len = int(self.ep_len[env_i])
    ep_max_off = float(self.ep_max_off[env_i])
    self.recent_returns.append(ep_return)
    self.recent_success.append(1.0 if info.get("success") else 0.0)
    self.recent_hits.append(1.0 if info.get("hit") else 0.0)
    self.recent_timeouts.append(1.0 if info.get("timeout") else 0.0)
    self.recent_offroads.append(
      1.0 if ep_max_off >= float(cfg.offroad_done_m) else 0.0
    )
    self.recent_lens.append(float(ep_len))
    self.recent_goal_dists.append(float(info.get("goal_dist", 0.0)))
    self.recent_speeds.append(float(info.get("speed_kmh", 0.0)))
    self.recent_peds.append(float(info.get("nearest_ped", 0.0)))
    self.recent_throttles.append(float(info.get("throttle", 0.0)))
    self.recent_early.append(1.0 if terminal == "collision" else 0.0)
    tag = f"[ep {env_i}]" if self.num_envs > 1 else "[ep]"
    print(
      f"{tag} {info.get('map', '?'):12s}  {terminal:10s}  "
      f"len={ep_len:4d}  ret={ep_return:+7.1f}  "
      f"dist={float(info.get('goal_dist', 0.0)):6.1f}  "
      f"spd={float(info.get('speed_kmh', 0.0)):5.1f}  "
      f"ped={float(info.get('nearest_ped', 0.0)):5.1f}  "
      f"off={ep_max_off:4.2f}  "
      f"thr={float(info.get('throttle', 0.0)):+5.2f}  "
      f"str={float(info.get('steer', 0.0)):+5.2f}  "
      f"cmd={int(info.get('cmd', -1))}  "
      f"g={float(info.get('dodge_gate', 0.0)):.2f}"
    )
    self.ep_return[env_i] = 0.0
    self.ep_len[env_i] = 0
    self.ep_max_off[env_i] = 0.0

  def collect_rollout(self) -> PilotRolloutBatch:
    cfg = self.config
    n_envs = self.num_envs
    t_len = cfg.rollout_steps
    h, w = cfg.image_height, cfg.image_width
    img_buf = np.zeros((t_len, n_envs, 3, h, w), dtype=np.float32)
    cmd_buf = np.zeros((t_len, n_envs), dtype=np.int64)
    act_buf = np.zeros((t_len, n_envs, cfg.action_dim), dtype=np.float32)
    logp_buf = np.zeros((t_len, n_envs), dtype=np.float32)
    rew_buf = np.zeros((t_len, n_envs), dtype=np.float32)
    done_buf = np.zeros((t_len, n_envs), dtype=np.float32)
    val_buf = np.zeros((t_len, n_envs), dtype=np.float32)
    gate_buf = np.zeros((t_len, n_envs), dtype=np.float32)
    spd_buf = np.zeros((t_len, n_envs), dtype=np.float32)
    pin_buf = np.zeros((t_len, n_envs), dtype=np.float32)
    clip = float(cfg.reward_clip)

    obs = self._obs
    self.model.eval()
    for t in range(t_len):
      img_t = torch.as_tensor(obs.images, dtype=torch.float32, device=self.device)
      cmd_t = torch.as_tensor(obs.commands, dtype=torch.long, device=self.device)
      spd_t = torch.as_tensor(obs.speeds, dtype=torch.float32, device=self.device)
      pin_bc = bool(getattr(cfg, "pin_bc_empty", True))
      pin_gate: torch.Tensor | None = None
      if pin_bc:
        # 迟滞更新钉 BC 状态，再喂合成门控（0=钉，1=放开）给 act。
        self._pin_bc = np.where(
          obs.gates >= float(cfg.explore_gate_on),
          False,
          np.where(
            obs.gates < float(cfg.explore_gate_off), True, self._pin_bc
          ),
        )
        pin_gate = torch.as_tensor(
          np.where(self._pin_bc, 0.0, 1.0),
          dtype=torch.float32,
          device=self.device,
        )
      # 训练：从 N(μ,σ) 采样；空路可钉冻结 BC。评测仍走确定性 μ。
      with torch.no_grad():
        action, log_prob, value = self.model.act(
          img_t,
          cmd_t,
          spd_t,
          deterministic=False,
          pin_bc_gate=pin_gate,
          pin_bc_min=float(cfg.explore_gate_min),
        )
      action_np = action.detach().cpu().numpy()
      next_obs, rewards, dones, infos = self.env.step(action_np)
      rewards = np.asarray(rewards, dtype=np.float32).reshape(n_envs)
      dones = np.asarray(dones).reshape(n_envs)

      img_buf[t] = obs.images
      cmd_buf[t] = obs.commands
      act_buf[t] = action_np
      logp_buf[t] = log_prob.detach().cpu().numpy()
      rew_buf[t] = np.clip(rewards, -clip, clip)
      done_buf[t] = dones.astype(np.float32)
      val_buf[t] = value.detach().cpu().numpy()
      gate_buf[t] = obs.gates
      spd_buf[t] = obs.speeds
      # pin_bc_empty=False 时不存在钉 BC 样本，全记 0（迟滞状态未启用）。
      pin_buf[t] = self._pin_bc.astype(np.float32) if pin_bc else 0.0

      self.ep_return += rewards.astype(np.float64)
      self.ep_len += 1
      for i in range(n_envs):
        self.ep_max_off[i] = max(
          float(self.ep_max_off[i]), float(infos[i].get("off_road", 0.0))
        )
        if bool(dones[i]):
          self._record_episode(i, infos[i])
          # 新回合从「空路钉 BC」重新起迟滞。
          self._pin_bc[i] = True
      obs = next_obs

    self._obs = obs
    with torch.no_grad():
      img_t = torch.as_tensor(obs.images, dtype=torch.float32, device=self.device)
      cmd_t = torch.as_tensor(obs.commands, dtype=torch.long, device=self.device)
      spd_t = torch.as_tensor(obs.speeds, dtype=torch.float32, device=self.device)
      last_value = self.model.forward(img_t, cmd_t, spd_t)[2].detach().cpu().numpy()

    advantages, returns = compute_gae(
      rew_buf, val_buf, done_buf, last_value, cfg.gamma, cfg.gae_lambda
    )
    flat = t_len * n_envs
    adv = advantages.reshape(flat)
    # 只有 update() 里 w>0 的样本进策略损失（门控开、没钉 BC 且不是低速）。
    # 若按全体样本归一化，这个子集的均值不为 0，偏移会不分好坏地整体抬高
    # 或压低躲人动作；归一化与 update 必须用同一组软权重。
    mask = self._policy_mask(
      gate_buf.reshape(flat), spd_buf.reshape(flat), pin_buf.reshape(flat)
    )
    if mask is not None and float(mask.sum()) >= 2:
      w_sum = float(mask.sum())
      mean = float((adv * mask).sum()) / w_sum
      var = float((mask * (adv - mean) ** 2).sum()) / w_sum
      adv = (adv - mean) / (float(np.sqrt(var)) + 1e-8)
    else:
      adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return PilotRolloutBatch(
      images=torch.as_tensor(img_buf.reshape(flat, 3, h, w), device=self.device),
      commands=torch.as_tensor(cmd_buf.reshape(flat), device=self.device),
      actions=torch.as_tensor(
        act_buf.reshape(flat, cfg.action_dim), device=self.device
      ),
      log_probs=torch.as_tensor(logp_buf.reshape(flat), device=self.device),
      rewards=torch.as_tensor(rew_buf.reshape(flat), device=self.device),
      dones=torch.as_tensor(done_buf.reshape(flat), device=self.device),
      values=torch.as_tensor(val_buf.reshape(flat), device=self.device),
      advantages=torch.as_tensor(adv, device=self.device),
      returns=torch.as_tensor(returns.reshape(flat), device=self.device),
      gates=torch.as_tensor(gate_buf.reshape(flat), device=self.device),
      speeds=torch.as_tensor(spd_buf.reshape(flat), device=self.device),
      pinned=torch.as_tensor(pin_buf.reshape(flat), device=self.device),
    )

  def update(self, batch: PilotRolloutBatch) -> dict[str, float]:
    cfg = self.config
    self._set_train_mode()
    n = batch.images.shape[0]
    idx = np.arange(n)
    metrics = {
      "policy_loss": 0.0,
      "value_loss": 0.0,
      "entropy": 0.0,
      "approx_kl": 0.0,
      "bc_kl": 0.0,
      "ent_coef": float(self._ent_coef),
      "bc_kl_coef": float(self._bc_kl_coef),
      "lr": float(self.optimizer.param_groups[0]["lr"]),
      "abs_action": float(batch.actions.abs().mean().item()),
    }
    batches = 0
    early_stop = False
    trainable = [p for p in self.model.parameters() if p.requires_grad]

    for _ in range(cfg.update_epochs):
      np.random.shuffle(idx)
      for start in range(0, n, cfg.minibatch_size):
        mb = idx[start : start + cfg.minibatch_size]
        images = batch.images[mb]
        commands = batch.commands[mb]
        actions = batch.actions[mb]
        old_logp = batch.log_probs[mb]
        old_values = batch.values[mb]
        adv = batch.advantages[mb]
        ret = batch.returns[mb]
        gates = batch.gates[mb]
        speeds = batch.speeds[mb]
        pinned = batch.pinned[mb]

        logp, entropy, value = self.model.evaluate(
          images,
          commands,
          speeds,
          actions,
        )
        log_ratio = logp - old_logp
        ratio = torch.exp(log_ratio)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
        min_surr = torch.min(surr1, surr2)
        mb_mask = self._policy_mask(gates, speeds, pinned)
        if mb_mask is not None:
          w = mb_mask.to(dtype=min_surr.dtype)
          w_sum = w.sum()
          if float(w_sum.item()) >= 4.0:
            w_sum = w_sum.clamp_min(1.0)
            policy_loss = -(min_surr * w).sum() / w_sum
            entropy_loss = -(entropy * w).sum() / w_sum
            kl_term = ((ratio - 1.0) - log_ratio) * w
            approx_kl = float(kl_term.sum().item() / float(w_sum.item()))
          else:
            policy_loss = min_surr.new_zeros(())
            entropy_loss = min_surr.new_zeros(())
            approx_kl = 0.0
        else:
          policy_loss = -min_surr.mean()
          entropy_loss = -entropy.mean()
          approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())

        v_clip = float(cfg.value_clip)
        if v_clip > 0.0:
          value_clipped = old_values + (value - old_values).clamp(-v_clip, v_clip)
          value_loss = 0.5 * torch.max(
            (value - ret).pow(2),
            (value_clipped - ret).pow(2),
          ).mean()
        else:
          value_loss = 0.5 * (value - ret).pow(2).mean()

        bc_kl = policy_loss.new_zeros(())
        if self.bc_ref is not None and self._bc_kl_coef > 0.0:
          mu, std, _ = self.model.forward(images, commands, speeds)
          with torch.no_grad():
            mu_bc, std_bc, _ = self.bc_ref.forward(images, commands, speeds)
          kl_dim = _diag_gaussian_kl(mu, std, mu_bc, std_bc)
          # 靠近行人时去掉转向 BC KL，允许侧向绕行；油门仍贴 BC，避免刹停。
          w_steer = (1.0 - gates) * float(cfg.bc_kl_steer_weight) + gates * float(
            cfg.bc_kl_dodge_weight
          )
          bc_kl = kl_dim[..., 0] * w_steer
          if kl_dim.shape[-1] >= 2:
            w_thr = (1.0 - gates) * float(
              getattr(cfg, "bc_kl_throttle_weight", 1.0)
            ) + gates * float(getattr(cfg, "bc_kl_throttle_dodge_weight", 0.0))
            bc_kl = bc_kl + kl_dim[..., 1] * w_thr
          bc_kl = bc_kl.mean()

        loss = (
          policy_loss
          + cfg.vf_coef * value_loss
          + self._ent_coef * entropy_loss
          + self._bc_kl_coef * bc_kl
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        self.optimizer.step()

        metrics["policy_loss"] += float(policy_loss.item())
        metrics["value_loss"] += float(value_loss.item())
        metrics["entropy"] += float(entropy.mean().item())
        metrics["approx_kl"] += approx_kl
        metrics["bc_kl"] += float(bc_kl.item())
        batches += 1

        if cfg.target_kl > 0.0 and approx_kl > cfg.target_kl:
          early_stop = True
          break
      if early_stop:
        break

    if batches:
      for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "bc_kl"):
        metrics[k] /= batches
    return metrics

  def stats(self, window: int = 20) -> dict[str, float]:
    rets = self.recent_returns[-window:]
    suc = self.recent_success[-window:]
    hits = self.recent_hits[-window:]
    tos = self.recent_timeouts[-window:]
    offs = self.recent_offroads[-window:]
    lens = self.recent_lens[-window:]
    dists = self.recent_goal_dists[-window:]
    spds = self.recent_speeds[-window:]
    peds = self.recent_peds[-window:]
    thrs = self.recent_throttles[-window:]
    return {
      "ep_return_mean": float(np.mean(rets)) if rets else 0.0,
      "success_rate": float(np.mean(suc)) if suc else 0.0,
      "hit_rate": float(np.mean(hits)) if hits else 0.0,
      "timeout_rate": float(np.mean(tos)) if tos else 0.0,
      "offroad_rate": float(np.mean(offs)) if offs else 0.0,
      "ep_len_mean": float(np.mean(lens)) if lens else 0.0,
      "goal_dist_mean": float(np.mean(dists)) if dists else 0.0,
      "speed_mean": float(np.mean(spds)) if spds else 0.0,
      "ped_mean": float(np.mean(peds)) if peds else 0.0,
      "throttle_mean": float(np.mean(thrs)) if thrs else 0.0,
      "episodes": float(len(self.recent_returns)),
      "early_rate": float(np.mean(self.recent_early[-window:]))
      if self.recent_early
      else 0.0,
    }

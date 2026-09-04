"""转向模型的训练与推理默认配置。"""

from __future__ import annotations

from dataclasses import dataclass

from drive_agent.commands import NUM_COMMANDS


@dataclass
class PilotNetConfig:
  image_height: int = 60
  image_width: int = 80
  num_commands: int = NUM_COMMANDS
  # (steer, throttle)；自动驾驶约 42 km/h 巡航，速度特征按此尺度归一。
  action_dim: int = 2
  speed_scale_kmh: float = 45.0
  throttle: float = 0.65
  lr: float = 1e-4
  batch_size: int = 64
  epochs: int = 12
  weight_decay: float = 1e-5
  val_ratio: float = 0.15
  seed: int = 42


@dataclass
class PilotRLConfig:
  """用 PPO 微调 PilotNet：(图像, 导航指令, 速度) → (转向, 油门)。

  导航指令仍由规则规划器提供。训练采集从策略高斯 N(μ,σ) 采样，
  评测/推理取均值；空路钉完整冻结 BC。靠近行人时策略梯度可进 CNN/主干。
  """

  num_commands: int = NUM_COMMANDS
  action_dim: int = 2
  image_height: int = 60
  image_width: int = 80
  speed_scale_kmh: float = 45.0

  lr: float = 1e-4
  lr_end: float = 2e-5
  # 行人附近策略损失可进 CNN / 主干（小学习率）；空路不进策略损失。
  # value 对特征 stop-grad，避免出界/撞人的价值损失冲掉跟路。
  freeze_features: bool = False
  trunk_lr_mult: float = 0.1
  features_lr_mult: float = 0.05
  # 空旷路面钉在 BC 上（与 main.py --checkpoint 闭环一致）。
  # 靠近行人时 KL 权重下降（见 bc_kl_dodge_weight）。
  bc_kl_coef: float = 0.08
  bc_kl_steer_weight: float = 1.0
  bc_kl_dodge_weight: float = 0.0
  # 空路和走廊都把油门拉回 BC 巡航；躲人靠转向，不靠刹停让行。
  bc_kl_throttle_weight: float = 1.0
  bc_kl_throttle_dodge_weight: float = 1.0
  gamma: float = 0.996
  gae_lambda: float = 0.95
  clip_eps: float = 0.2
  ent_coef: float = 0.002
  ent_coef_end: float = 0.0004
  log_std_init_steer: float = -1.5
  log_std_init_throttle: float = -1.6
  # 行人特征不再进策略。下列字段只为旧 checkpoint 反序列化保留。
  ped_feat_dim: int = 3
  ped_reinit_std: float = 0.0
  ped_reinit_stale_steps: int = 0
  # 空旷路面钉在冻结 BC（与 main.py --checkpoint 闭环一致）。
  # 走廊有人时只对转向从 N(μ,σ) 采样；油门用均值，避免采到刹车学成让行。
  explore_gate_min: float = 0.2
  # 门控迟滞（旧外加噪声用）；现已改为策略分布采样。
  explore_gate_on: float = 0.22
  explore_gate_off: float = 0.06
  policy_gate_min: float = 0.2
  # 停车时打方向几乎不改变轨迹，低速步不进策略损失。
  policy_min_speed_kmh: float = 6.0
  # False = 采集时油门取 μ，不采样；PPO 也不用油门 log π（躲人靠转向）。
  explore_throttle: bool = False
  # 旧外加探索噪声字段，只为 checkpoint 反序列化保留，不再参与动作。
  explore_rho_dodge: float = 0.95
  explore_rho_clear: float = 0.0
  explore_rho_throttle: float = 0.2
  explore_std_boost: float = 0.0
  explore_std_boost_throttle: float = 0.0
  explore_steer_ref_kmh: float = 16.0
  explore_steer_speed_pow: float = 1.0
  explore_steer_speed_floor: float = 0.28
  explore_z_clip_steer: float = 1.2
  explore_adapt: bool = False
  explore_adapt_window: int = 8
  explore_adapt_trigger: float = 0.35
  explore_std_boost_min: float = 0.45
  explore_std_boost_max: float = 1.35
  explore_rho_dodge_min: float = 0.92
  explore_rho_dodge_max: float = 0.97
  explore_adapt_down: float = 0.25
  explore_adapt_up: float = 0.08
  # 空路钉在冻结 BC 最后一层上，避免躲人更新把 90° 弯的跟路带偏。
  pin_bc_empty: bool = True
  target_kl: float = 0.02
  vf_coef: float = 0.5
  max_grad_norm: float = 0.5
  update_epochs: int = 4
  minibatch_size: int = 64
  rollout_steps: int = 2048
  total_steps: int = 200_000
  # 1 = 训练进程内单环境（可 --window）；>1 时多进程并行采集，主进程批量推理。
  num_envs: int = 1
  # success 与 return 都连续 patience 次更新没有新高才停；只看 success 会在还在涨 return 时砍掉。
  # 30 ≈ 6 万步，滚动 20 局里曲线一抖就会提前砍掉后半段；60 ≈ 12 万步，仍能在 20 万步预算内停。
  early_stop_patience: int = 60
  early_stop_slack: float = 0.0
  early_stop_return_slack: float = 1.0
  # 必须大于 |reward_goal| / |reward_collision|，否则终止奖励会被裁掉。
  reward_clip: float = 200.0
  value_clip: float = 0.0
  action_repeat: int = 1

  dt: float = 1.0 / 30.0
  max_episode_seconds: float = 90.0
  autopilot_speed_frac: float = 1.0 / 3.0
  # RuleExpert 巡航先验（专家对照 / 1D 动作回退）；BC 油门从数据学。
  throttle_prior: float = 0.65
  seed: int = 42

  residual_lane_m: float = 5.0
  # 门控距离下限（低速）；巡航按 TTC 拉远，避免 45 km/h 时 16–32 m 才开探索。
  residual_gate_near: float = 16.0
  residual_gate_far: float = 32.0
  residual_gate_ttc_near: float = 2.5
  residual_gate_ttc_far: float = 5.0
  ped_dist_scale: float = 40.0

  # 成功奖励必须压过约 2 秒冲出路面自杀（旧 goal=+20 会被稠密代价淹没）。
  reward_goal: float = 150.0
  # 顺利切到下一导航路点（小地图黄点）；低于终点，避免盖过旗帜。
  reward_waypoint: float = 40.0
  reward_collision: float = -150.0
  reward_timeout: float = -20.0
  reward_offroad_done: float = -40.0
  offroad_done_m: float = 1.8
  reward_progress: float = 0.25
  reward_cte: float = -0.04
  # 0 = 避让门控打开时去掉横向误差项（否则会对抗侧向绕行）。
  cte_dodge_scale: float = 0.0
  cte_clip_m: float = 6.0
  reward_on_road: float = 0.01
  reward_time: float = -0.002
  # 贴身/TTC 会把「前方有人就刹停」学成主解；躲人靠转向和撞人终止。
  reward_proximity: float = 0.0
  proximity_range: float = 4.0
  proximity_power: float = 2.0
  reward_ttc: float = 0.0
  ttc_horizon_s: float = 4.0
  # 边走边把行人甩到车侧（|right| 变大），绕行的稠密信号。
  reward_dodge_lat: float = 0.05
  reward_offroad: float = -0.15
  reward_stall: float = -0.08
  stall_speed_kmh: float = 5.0
  stall_clear_m: float = 8.0
  # 低速重叠仍算撞人；下列让行字段只为旧 checkpoint 反序列化保留。
  yield_speed_kmh: float = 0.0
  yield_brake_gain: float = 1.5
  yield_creep_kmh: float = 8.0
  yield_emergency_ttc_s: float = 1.6
  yield_clear_m: float = 1.8
  yield_hold_m: float = 3.6
  # None = 与 main.py 相同（PedestrianCrowd 默认 10–24）；0 = 不刷人（eval --no-peds）
  rl_ped_max: int | None = None
  # 撞人结束回合；压草坪不结束，只靠 reward_offroad 扣分（可从路缘开回）。
  # --like-main 评测时撞人也不结束。
  terminate_on_hit: bool = True
  terminate_on_offroad: bool = False

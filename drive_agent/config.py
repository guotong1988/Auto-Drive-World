"""转向模型的训练与推理默认配置。"""

from __future__ import annotations

from dataclasses import dataclass

from drive_agent.commands import NUM_COMMANDS

# 策略输入分辨率 (H, W)。采集 / BC / PPO / eval / main 自动驾驶共用。
POLICY_IMAGE_HEIGHT = 120
POLICY_IMAGE_WIDTH = 160


@dataclass
class PilotNetConfig:
  image_height: int = POLICY_IMAGE_HEIGHT
  image_width: int = POLICY_IMAGE_WIDTH
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
  image_height: int = POLICY_IMAGE_HEIGHT
  image_width: int = POLICY_IMAGE_WIDTH
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
  # 随训练进度线性退火到该值：前期锚定防躲人更新带偏跟路，后期放开
  # 释放上限（anchor 同时惩罚 μ 与 σ，全程固定会压住末段精细优化）。
  bc_kl_coef_end: float = 0.0
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
  # 空旷路面钉在冻结 BC（与 main.py --checkpoint 闭环一致）。
  # 走廊有人时只对转向从 N(μ,σ) 采样；油门用均值，避免采到刹车学成让行。
  explore_gate_min: float = 0.2
  # 钉 BC 的迟滞：gate 升过 on 才放开采样，降下 off 才钉回，
  # 避免 gate 在阈值附近抖动时动作分布在 BC 与采样之间来回跳。
  explore_gate_on: float = 0.22
  explore_gate_off: float = 0.06
  policy_gate_min: float = 0.2
  # 策略损失门控的软边宽度：样本权重在 gate_min ± soft 之间从 0 线性升到 1。
  policy_gate_soft: float = 0.1
  # 停车时打方向几乎不改变轨迹，低速步不进策略损失。
  policy_min_speed_kmh: float = 6.0
  # False = 采集时油门取 μ，不采样；PPO 也不用油门 log π（躲人靠转向）。
  explore_throttle: bool = False
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
  # patience 按更新次数算：60 次 × rollout_steps × action_repeat ≈ 18 万环境步
  # （--total-steps 也是环境步）。滚动 50 局里曲线一抖就会提前砍掉后半段，别再调小。
  early_stop_patience: int = 60
  early_stop_slack: float = 0.0
  early_stop_return_slack: float = 1.0
  # 必须大于 |reward_goal| / |reward_collision|，否则终止奖励会被裁掉。
  reward_clip: float = 200.0
  value_clip: float = 0.0
  # 一个动作保持几个物理 tick。dt=1/30 且 repeat=1 时一局有 2700 个决策步，
  # gamma=0.996 的视野只有 250 步（8 秒 / 约 100 m），到旗子的 +150 折回起点
  # 只剩 0.003，等于目标里没有「到终点」。repeat=3 是 10 Hz 决策、一局 900 步，
  # 视野约 25 秒（约 300 m），goal 折扣升到 0.027。
  # BC 是 30 Hz 数据训的，repeat 越大 pin 的空路动作越粗；闭环开始左右摆就降回 2。
  action_repeat: int = 3

  dt: float = 1.0 / 30.0
  max_episode_seconds: float = 90.0
  autopilot_speed_frac: float = 1.0 / 3.0
  # RuleExpert 巡航先验（专家对照 / 1D 动作回退）；BC 油门从数据学。
  throttle_prior: float = 0.65
  seed: int = 42

  residual_lane_m: float = 5.0
  # 规则绕行（RuleDodge）的激活门控：低于此值且最近行人还在前方时跟中线。
  rule_dodge_gate_on: float = 0.15
  # 门控距离下限（低速）；巡航按 TTC 拉远，避免 45 km/h 时 16–32 m 才开探索。
  residual_gate_near: float = 16.0
  residual_gate_far: float = 32.0
  residual_gate_ttc_near: float = 2.5
  residual_gate_ttc_far: float = 5.0
  ped_dist_scale: float = 40.0

  # 成功奖励必须压过约 2 秒冲出路面自杀（旧 goal=+20 会被稠密代价淹没）。
  reward_goal: float = 150.0
  # 顺利切到下一导航路点（小地图黄点）。整条路线的路点总和要明显低于 reward_goal，
  # 否则近处不打折的路点比远处打完折的旗子更划算，最优解变成一路刷黄点。
  # 路线上通常 3–6 个黄点，12 → 总和 36–72，仍远低于 150。
  reward_waypoint: float = 12.0
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

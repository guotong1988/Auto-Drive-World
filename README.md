# Auto Drive World
https://space.bilibili.com/447278957/lists

## 规则自动驾驶
```
~/.pyenv/versions/3.11.13/bin/python3 main.py
```

## 无窗口跑规则驾驶，检查是否会开上草坪
```
~/.pyenv/versions/3.11.13/bin/python3 tools/sim_expert.py          # 全部地图
```

```
~/.pyenv/versions/3.11.13/bin/python3 tools/sim_expert.py l_bend   # 单张地图并打印轨迹
```

```
~/.pyenv/versions/3.11.13/bin/python3 tools/sim_expert.py train_maps
```

## 规则自动驾驶，切换地图
训练地图: crossroads / l_bend / zigzag / t_junction / dual_bend / u_turn

测试地图: ring / grid / chicane / spur / plaza

路上有脚本行人（路口/路段斑马线过街，路边停一下再横穿；偶有未标线路段乱穿）；小地图橙色点；撞到会提示
```
~/.pyenv/versions/3.11.13/bin/python3 main.py --map l_bend
```

```
~/.pyenv/versions/3.11.13/bin/python3 main.py --map chicane
```

## 人工驾驶，采集数据
场景含行人（车头前视画面里会出现）；标签是人开的转向和油门，不要求躲人

同时写入车速（km/h），给后续 `(图像, 指令, 速度) → (转向, 油门)` 用
```
~/.pyenv/versions/3.11.13/bin/python3 main.py --collect --map grid
```

## 规则自动驾驶，采集数据
同样有行人走动，但规则只沿路到旗子，不躲人、撞了也照常（可用 --dodge 采绕行）

--episodes 为每张地图成功 episode 数；可用单图或 train_maps / test_maps / all

每局在图上多条可达终点的简单路径中随机选一条（不固定最短路）；--seed 可复现

默认会短暂扰动方向盘（标签仍是专家纠偏），采集偏离中线后的回正

扰动太密容易把「猛打回正」学进模型，闭环会左右摆；`--no-disturb` 关掉（等价 `--disturb-prob 0`）

可混采：同一 `--output` 先无噪声再有噪声，训练会一起用

--dodge：前方走廊有人时规则打方向绕开（不刹车）；横穿从行人后方绕，不插到行进方向前面；标签是绕行转向；撞人的局丢掉

混入绕行数据后再训 BC，RL 从「已会一点绕」的 μ 附近采样，才容易采到成功轨迹

油门标签是专家过弯限速，不叠噪声；须重新采集（旧 npz 没有 throttle/speed）

压到一点草地（车角擦出路缘）仍保留；冲进草地太深或累计压草过久则丢弃重采
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.collect --no-disturb --episodes 2 --map train_maps
```

```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.collect --episodes 3 --map train_maps
```

```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.collect --no-disturb --dodge --episodes 3 --map train_maps
```

```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.collect --episodes 3 --map zigzag
```

### 无窗口采集
（离屏车头前视，不弹 3D 窗口；标签/扰动/落盘与上面相同，可叠加 --no-disturb / --dodge）

策略观测是挡风玻璃高度前视，不是跟随相机；旧跟随相机 npz 不能接着训，请重新采集
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.collect --headless --no-disturb --episodes 2 --map train_maps
```

```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.collect --headless --dodge --episodes 3 --map all
```

```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.collect --headless --no-disturb --dodge --episodes 5 --map all
```

## 训练模型
共享 CNN，按 straight/left/right/stop 分头；(图像, 导航指令, 车速) → 所选头的 (转向, 油门)

行人只出现在画面里；旧分头最后一层的行人列加载时丢掉

旧跟随相机数据 / 旧单头拼接 checkpoint 都不能用，请按新镜头重新采集再训练
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.train \
  --data data/driving \
  --epochs 15 \
  --batch-size 64 \
  --lr 1e-4 \
  --checkpoint checkpoints/pilotnet.pt
```

## 加载模型
之后按 T 开启自动驾驶（导航指令仍由规则规划器提供，转向和油门由模型预测）

油门由模型直接执行，不再因前方行人强制降速；泛化评测请换测试地图
```
~/.pyenv/versions/3.11.13/bin/python3 main.py --checkpoint checkpoints/pilotnet.pt --map crossroads
```

## 闭环评测
（与 PPO 同一套环境：随机路线、行人、撞人终止；压草只扣分不结束；确定性、不加探索噪声）

先看 BC 基线能不能到终点；--steer expert 是规则转向+油门对照；--no-peds 只测跟路
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.eval_pilot \
  --map train_maps \
  --checkpoint checkpoints/pilotnet.pt \
  --episodes 3
```

规则对照（环境成功率上限）
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.eval_pilot \
  --map train_maps --steer expert --episodes 3
```

关掉行人，只看跟路
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.eval_pilot \
  --map l_bend --checkpoint checkpoints/pilotnet.pt --no-peds --episodes 3
```

与 PPO 同一套终止（撞人结束；这是 RL 该对照的 BC 基线）
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.eval_pilot \
  --map all --checkpoint checkpoints/pilotnet.pt --episodes 3
```

对齐 main.py：撞人/压草不结束，只看能不能到旗子（可换 --map ring）
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.eval_pilot \
  --map all --checkpoint checkpoints/pilotnet.pt --like-main --episodes 3
```

## 整网 PPO 微调 PilotNet：(画面, 导航指令, 速度) → (steer, throttle)
导航指令仍是规则的 straight/left/right；行人只出现在画面里，不另做特征向量

空路钉完整冻结 BC；靠近行人只对转向从高斯采样（油门用均值，避免刹停），策略梯度可进 CNN/主干

绕开行人/到旗子有正奖励；压草只扣分不结束，撞人仍结束 rollout

无窗口离屏渲染
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.train_pilot_rl \
  --map all \
  --pilot-checkpoint checkpoints/pilotnet.pt \
  --total-steps 1000000 \
  --rollout-steps 1024 \
  --checkpoint checkpoints/pilot_rl.pt
```

### 并行采集
多进程仿真 + 主进程批量推理，提高 GPU 利用率（--window 时不能并行）

每个环境采 --rollout-steps 步，一次更新样本量 = num_envs × rollout_steps；总步数仍按环境交互累计
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.train_pilot_rl \
  --map all \
  --pilot-checkpoint checkpoints/pilotnet.pt \
  --total-steps 10000000 \
  --rollout-steps 1024 \
  --num-envs 24 \
  --checkpoint checkpoints/pilot_rl.pt
```

开窗口看训练过程（同一套 PPO；窗口是跟随相机给人看，策略仍吃前视；比离屏慢）

行人在小地图上是橙色点（出生点 28m 内不刷，3D 里一开始常常看不到人）
```
~/.pyenv/versions/3.11.13/bin/python3 -m drive_agent.train_pilot_rl \
  --map train_maps \
  --pilot-checkpoint checkpoints/pilotnet.pt \
  --total-steps 200000 \
  --rollout-steps 1024 \
  --checkpoint checkpoints/pilot_rl.pt \
  --window
```

## Pilot-RL 自动驾驶
（按 T；转向和油门由微调网络，导航指令仍是规则）
```
~/.pyenv/versions/3.11.13/bin/python3 main.py \
  --checkpoint checkpoints/pilot_rl_best.pt \
  --map crossroads
```

无窗口评估（离屏车头前视，自动开自动驾驶，不弹 3D 窗口；与 PPO 同一套终止：撞人/超时，压草不结束）
```
~/.pyenv/versions/3.11.13/bin/python3 main.py \
  --checkpoint checkpoints/pilot_rl_best.pt \
  --map crossroads \
  --headless
```

```
~/.pyenv/versions/3.11.13/bin/python3 main.py \
  --checkpoint checkpoints/pilot_rl_best.pt \
  --map train_maps \
  --headless \
  --episodes 3
```

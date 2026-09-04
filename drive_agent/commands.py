"""条件模仿学习用的高层导航指令。"""

from __future__ import annotations

COMMAND_NAMES: tuple[str, ...] = ("straight", "left", "right", "stop")
COMMAND_TO_ID: dict[str, int] = {name: i for i, name in enumerate(COMMAND_NAMES)}
NUM_COMMANDS: int = len(COMMAND_NAMES)

# 路口转角小于该值（度）视为直行通过。
TURN_COMMAND_DEG: float = 25.0
# 距转弯圆弧超过该距离仍标 straight。专家纯追踪大约在前视(~10m)碰到
# 圆弧才开始打方向；20m 就标 left 时，分头仍会把弯前直行和弯中转向
# 平均成提前打方向、过早收油，闭环抄近路压路缘。
TURN_COMMAND_PREVIEW_M: float = 6.0
# 出弯后航向还没对齐出口时继续标 left/right，避免切点上一头切回 straight
# 把油门拉满、方向盘回正。18° 仍会在车身没回正时加速冲出路面。
TURN_COMMAND_HOLD_DEG: float = 8.0

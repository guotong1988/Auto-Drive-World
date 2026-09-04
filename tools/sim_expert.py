"""无窗口跑规则专家；按地图报告冲出路面情况。

用真实 Bullet 车辆、不开图形窗口，以便检查转弯几何，而不必盯 3D 画面。

    python tools/sim_expert.py              # 全部地图
    python tools/sim_expert.py l_bend       # 一张地图，并导出轨迹
    python tools/sim_expert.py train_maps   # 仅训练划分
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panda3d.core import NodePath, loadPrcFileData  # noqa: E402

loadPrcFileData("", "window-type none")

from drive_agent.rule_expert import RuleExpert  # noqa: E402
from drive_env.maps import MapSpec, get_map, resolve_maps  # noqa: E402
from drive_env.physics import PhysicsWorld  # noqa: E402
from drive_env.terrain import build_barriers, build_terrain  # noqa: E402
from drive_env.vehicle import Vehicle  # noqa: E402

DT = 1.0 / 60.0
MAX_SECONDS = 120.0
AUTOPILOT_MAX_KMH = Vehicle.MAX_SPEED_KMH / 3.0


HALF_LENGTH = 1.75
HALF_WIDTH = 0.88


def _point_off_road(x: float, y: float, spec: MapSpec) -> float:
  best = float("inf")
  for x0, x1, y0, y1 in spec.road_rects:
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    best = min(best, (dx * dx + dy * dy) ** 0.5)
    if best == 0.0:
      return 0.0
  return best


def off_road_distance(x: float, y: float, heading_deg: float, spec: MapSpec) -> float:
  """底盘完全在沥青上时为 0，否则为越过路缘的米数。"""
  h = math.radians(heading_deg)
  # Panda 航向：0 朝向 +Y，正值左转。
  fx, fy = -math.sin(h), math.cos(h)
  rx, ry = fy, -fx
  worst = 0.0
  for sl, sw in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
    cx = x + fx * HALF_LENGTH * sl + rx * HALF_WIDTH * sw
    cy = y + fy * HALF_LENGTH * sl + ry * HALF_WIDTH * sw
    worst = max(worst, _point_off_road(cx, cy, spec))
  return worst


def run_map(map_id: str, route: list[str] | None = None, trace: bool = False) -> dict:
  spec = get_map(map_id)
  render = NodePath("render")
  physics = PhysicsWorld()
  build_terrain(render, physics, spec)
  build_barriers(render, physics, spec)

  sx, sy = spec.spawn_xy
  vehicle = Vehicle(render, physics, (sx, sy, spec.spawn_z), spec.spawn_heading)
  vehicle.max_speed_kmh = AUTOPILOT_MAX_KMH
  # 默认走最短路；传入 ``route`` 时锁定该顶层规划。
  expert = RuleExpert(map_spec=spec, route_policy="shortest")
  if route is not None:
    expert.commit_route(route)

  worst = 0.0
  worst_at = (0.0, 0.0)
  off_frames = 0
  steps = int(MAX_SECONDS / DT)
  rows = []
  i = 0

  for i in range(steps):
    pos = vehicle.node.getPos()
    heading = vehicle.node.getH()
    speed = vehicle.speed_kmh()
    throttle, steer = expert.predict(pos.x, pos.y, heading, speed)
    if expert.arrived:
      vehicle.set_input(0.0, 0.0, brake=1.0)
    else:
      vehicle.set_input(throttle, steer)
    vehicle.update(DT)

    pos = vehicle.node.getPos()
    dist = off_road_distance(pos.x, pos.y, vehicle.node.getH(), spec)
    if dist > 0.5:
      off_frames += 1
    if dist > worst:
      worst = dist
      worst_at = (pos.x, pos.y)
    if trace and i % 15 == 0:
      rows.append(
        f"  t={i * DT:5.1f}s pos=({pos.x:7.1f},{pos.y:7.1f}) h={heading:7.1f} "
        f"steer={steer:+.2f} wheel={vehicle.steering:+6.1f} "
        f"v={speed:5.1f} thr={throttle:+.2f} off={dist:5.1f}"
      )
    if expert.arrived:
      break

  result = {
    "map": map_id,
    "route": " → ".join(expert.route),
    "arrived": expert.arrived,
    "seconds": (i + 1) * DT,
    "worst_off_road": worst,
    "worst_at": worst_at,
    "off_seconds": off_frames * DT,
    "trace": rows,
  }
  return result


def run_all_routes(map_id: str, trace: bool = False) -> list[dict]:
  """跑遍所有可达终点的简单路径（顶层规划变体）。"""
  probe = RuleExpert(map_spec=get_map(map_id), route_policy="shortest")
  routes = probe.enumerate_routes_to_goal(probe.spawn_node)
  results = []
  for route in routes:
    results.append(run_map(map_id, route=route, trace=trace and len(routes) == 1))
  return results


def main():
  selections = sys.argv[1:] or ["all"]
  ids: list[str] = []
  for sel in selections:
    ids.extend(resolve_maps(sel))
  # 去重重叠别名，同时保持原顺序。
  seen: set[str] = set()
  ids = [m for m in ids if not (m in seen or seen.add(m))]
  single = len(sys.argv) == 2 and len(ids) == 1
  for map_id in ids:
    results = run_all_routes(map_id, trace=single)
    for r in results:
      status = "到达" if r["arrived"] else "未到达"
      print(
        f"{r['map']:<12} {status}  用时 {r['seconds']:5.1f}s  "
        f"最大离开路面 {r['worst_off_road']:5.1f}m "
        f"@({r['worst_at'][0]:.0f},{r['worst_at'][1]:.0f})  "
        f"越野时长 {r['off_seconds']:4.1f}s  "
        f"| {r['route']}"
      )
      for row in r["trace"]:
        print(row)


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""无窗口跑规则专家；按地图报告冲出路面情况。

用真实 Bullet 车辆、不开图形窗口，以便检查转弯几何，而不必盯 3D 画面。

    python -m drive_agent.sim_expert              # 全部地图
    python -m drive_agent.sim_expert l_bend       # 一张地图，并导出轨迹
    python -m drive_agent.sim_expert train_maps   # 仅训练划分
"""

from __future__ import annotations

import sys

from panda3d.core import NodePath, loadPrcFileData

loadPrcFileData("", "window-type none")

from drive_agent.ped_safety import off_road_distance
from drive_agent.rule_expert import RuleExpert
from drive_env.maps import get_map, resolve_maps
from drive_env.physics import PhysicsWorld
from drive_env.terrain import build_barriers, build_terrain
from drive_env.vehicle import Vehicle

DT = 1.0 / 60.0
MAX_SECONDS = 120.0
AUTOPILOT_MAX_KMH = Vehicle.MAX_SPEED_KMH / 3.0


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

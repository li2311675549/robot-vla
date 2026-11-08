#!/usr/bin/env python3
"""Actual physics smoke, exits nonzero on invalid motion/time/warnings."""
import argparse
import datetime
import json
import platform

from _bootstrap import ROOT
import mujoco
import numpy as np
from robot_exam.simulation import Simulation, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["minimal", "ur5", "robotiq"], default="minimal")
    args = parser.parse_args()
    sim = Simulation(args.scene)
    initial = sim.joint_positions()
    target = targets(sim, args.scene)
    steps = round(3 / sim.model.opt.timestep)
    sim.step(target, steps)
    final = sim.joint_positions()
    movement = max(abs(final[k] - initial[k]) for k in final)
    assert movement > 0.05, f"No effective joint motion: {movement}"
    assert abs(sim.data.time - 3) < 1e-7
    report = {"status": "PASS", "scene": args.scene,
              "timestamp": datetime.datetime.now().astimezone().isoformat(),
              "python": platform.python_version(), "mujoco": mujoco.__version__,
              "steps": steps, "sim_seconds": sim.data.time,
              "nq": sim.model.nq, "nv": sim.model.nv, "nu": sim.model.nu,
              "initial": initial, "target_ctrl": target.tolist(), "final": final,
              "max_movement_rad": movement, "warnings": sim.data.warning.number.tolist()}
    path = ROOT / f"outputs/smoke/{args.scene}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

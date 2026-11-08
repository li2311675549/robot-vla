#!/usr/bin/env python3
import argparse
import csv
import json
import math
import datetime

from _bootstrap import ROOT
import numpy as np
from robot_exam.simulation import Simulation
from robot_exam.control.base import BaseController, wrap_angle


def run_case(case, output=None, callback=None):
    sim = Simulation(ROOT / "scenes/husky.xml")
    base = BaseController(sim)
    for _ in range(500):
        sim.step(sim.data.ctrl)
        if callback:
            callback(sim, "settle")
    initial = base.pose().copy()
    goal = {"straight": [.6, 0, 0], "reverse": [-.4, 0, 0],
            "turn": [0, 0, 1], "curve": [.6, .30, .5]}[case]
    rows = []
    for step in range(6000):
        if case == "turn":
            base.rotate(initial, goal[2])
        elif case == "reverse":
            x, y, yaw = base.pose()
            base.command(np.clip(1.2*(goal[0]-x), -.2, .2), -6*yaw)
        else:
            base.track(goal)
        sim.step(sim.data.ctrl)
        if callback:
            callback(sim, case)
        if step % 25 == 0:
            rows.append([sim.data.time, *base.pose(), *sim.data.qvel[:6], *sim.data.qvel[6:10], *sim.data.ctrl[:4]])
        pose = base.pose()
        if step > 1000 and np.linalg.norm(pose[:2]-goal[:2]) < (.06 if case == "turn" else .02) and abs(wrap_angle(pose[2]-goal[2])) < .025:
            break
    before_stop = base.pose().copy()
    base.command(0, 0)
    for _ in range(750):
        sim.step(sim.data.ctrl)
        if round(sim.data.time/sim.model.opt.timestep) % 25 == 0:
            rows.append([sim.data.time, *base.pose(), *sim.data.qvel[:6], *sim.data.qvel[6:10], *sim.data.ctrl[:4]])
        if callback:
            callback(sim, "stop")
    final = base.pose()
    result = {"case": case, "initial": initial.tolist(), "goal": goal, "final": final.tolist(),
              "position_error_m": float(np.linalg.norm(final[:2]-goal[:2])),
              "heading_error_rad": float(abs(wrap_angle(final[2]-goal[2]))),
              "stopping_drift_m": float(np.linalg.norm(final[:2]-before_stop[:2])),
              "final_speed_m_s": float(np.linalg.norm(sim.data.qvel[:2])),
              "final_angular_speed_rad_s": float(abs(sim.data.qvel[5])),
              "sim_seconds": sim.data.time, "warnings": sim.data.warning.number.tolist()}
    result["status"] = "PASS" if (result["position_error_m"] < .08 and result["heading_error_rad"] < .06 and result["final_speed_m_s"] < .01 and result["stopping_drift_m"] < .03) else "FAIL"
    if output:
        output.mkdir(parents=True, exist_ok=True)
        with (output / f"{case}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "x", "y", "yaw", "vx", "vy", "vz", "wx", "wy", "wz",
                             "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr", "ctrl_fl", "ctrl_fr", "ctrl_rl", "ctrl_rr",
                             "position_error_m", "heading_error_rad"])
            writer.writerows(row + [float(np.linalg.norm(np.asarray(row[1:3])-goal[:2])), float(abs(wrap_angle(row[3]-goal[2])))] for row in rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    out = ROOT / "outputs/stage2/gate1"
    results = []
    from robot_exam.evaluation.recording import VideoRecorder
    import mujoco
    for case in ("straight", "reverse", "turn", "curve"):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.25, .15, .2]
        camera.distance = 2.5
        camera.azimuth = 135
        camera.elevation = -35
        recorder = VideoRecorder(out / case, camera=camera) if args.record else None
        try:
            results.append(run_case(case, out, recorder))
        finally:
            if recorder:
                recorder.close()
    report = {"status": "PASS" if all(x["status"] == "PASS" for x in results) else "FAIL",
              "timestamp": datetime.datetime.now().astimezone().isoformat(), "cases": results}
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

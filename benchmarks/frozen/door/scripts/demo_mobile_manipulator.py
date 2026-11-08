#!/usr/bin/env python3
"""Gate 4: wheel-drive to table -> RGB-D target -> physical pick and place."""
import argparse
import datetime
import json
import math
import time

from _bootstrap import ROOT
import numpy as np
from robot_exam.simulation import Simulation
from robot_exam.control.base import BaseController
from robot_exam.skills.pick_place import PickPlace
from robot_exam.perception.rgbd import RGBDCamera


def run(output, callback=None):
    output.mkdir(parents=True, exist_ok=True)
    sim = Simulation(ROOT / "scenes/mobile_manipulator.xml")
    base = BaseController(sim)
    stability = []
    start = time.monotonic()

    def observe(simulation, stage):
        if round(simulation.data.time/simulation.model.opt.timestep) % 25 == 0:
            rotation = simulation.data.xmat[base.body].reshape(3, 3)
            stability.append({"time": simulation.data.time, "stage": stage, "base": base.pose().tolist(),
                              "tilt_rad": float(math.acos(np.clip(rotation[2, 2], -1, 1))),
                              "base_speed": float(np.linalg.norm(simulation.data.qvel[:2])),
                              "wheel_velocity": simulation.data.qvel[6:10].tolist()})
        if callback:
            callback(simulation, stage)

    skill = PickPlace(sim, observe)
    try:
        skill.hold(.8, "initial_settle")
        drive_start = base.pose().copy()
        reached = False
        for _ in range(10000):
            distance, angle = base.track([1, 0, 0], max_linear=.20)
            skill.tick("drive_to_table")
            if distance < .022 and angle < .025:
                reached = True
                break
        if not reached:
            raise RuntimeError(f"Base parking timeout: {base.pose()}")
        base.command(0, 0)
        skill.hold(1., "park")
        parked = base.pose().copy()
        if np.linalg.norm(sim.data.qvel[:2]) > .01:
            raise RuntimeError("Base failed to stop before manipulation")
        camera = RGBDCamera(sim)
        try:
            rgb, depth = camera.capture()
            calibration = camera.save(output / "camera", rgb, depth)
            detection = camera.detect_red_fixture(rgb, depth)
        finally:
            camera.close()
        truth = sim.data.xpos[skill.object_body].copy()  # validation/scoring only
        localization_error = float(np.linalg.norm(np.asarray(detection["center"])-truth))
        if localization_error > .01:
            raise RuntimeError(f"RGB-D localization error: {localization_error}")
        # Only image/depth estimate enters the pick skill. Placement is a known fixture.
        grasp = skill.run(detection["center"], [1.55, .22, .750])
        manip = [r for r in stability if r["stage"] not in ("initial_settle", "drive_to_table", "park")]
        drift = max(np.linalg.norm(np.asarray(r["base"][:2])-parked[:2]) for r in manip)
        tilt = max(r["tilt_rad"] for r in stability)
        report = {"status": "PASS" if grasp["status"] == "PASS" and drift < .03 and tilt < .12 else "FAIL",
                  "drive_start": drive_start.tolist(), "parked": parked.tolist(),
                  "parking_position_error_m": float(np.linalg.norm(parked[:2]-[1, 0])),
                  "base_drift_during_manipulation_m": float(drift), "max_base_tilt_rad": tilt,
                  "rgbd_localization_error_m": localization_error, "detection": detection,
                  "grasp": grasp, "sim_seconds": sim.data.time,
                  "wall_seconds": time.monotonic()-start,
                  "observation_mode": "RGB-D fixture localization; simulated base pose feedback for parking; object truth scoring only"}
    except Exception as exc:
        report = {"status": "FAIL", "error": str(exc), "sim_seconds": sim.data.time,
                  "wall_seconds": time.monotonic()-start}
    report["timestamp"] = datetime.datetime.now().astimezone().isoformat()
    (output / "result.json").write_text(json.dumps(report, indent=2))
    (output / "trajectory.json").write_text(json.dumps(skill.rows))
    (output / "base_stability.json").write_text(json.dumps(stability))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "outputs/stage2/gate4"))
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    from pathlib import Path
    from robot_exam.evaluation.recording import VideoRecorder
    recorder = VideoRecorder(Path(args.output) / "video") if args.record else None
    try:
        report = run(Path(args.output), callback=recorder)
    finally:
        if recorder:
            recorder.close()
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

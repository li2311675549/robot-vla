#!/usr/bin/env python3
import json
from _bootstrap import ROOT
import numpy as np
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController
from robot_exam.perception.rgbd import RGBDCamera


def main():
    out = ROOT / "outputs/stage2/gate3"
    sim = Simulation(ROOT / "scenes/ur5_gripper.xml")
    arm = ArmController(sim)
    for _ in range(500):
        arm.command(arm.reference)
        sim.step(sim.data.ctrl)
    camera = RGBDCamera(sim)
    try:
        rgb, depth = camera.capture()
        calibration = camera.save(out, rgb, depth)
        detection = camera.detect_red_fixture(rgb, depth)
        truth = sim.data.xpos[sim.model.body("object").id].copy()  # scoring only
        error = float(np.linalg.norm(np.asarray(detection["center"])-truth))
        report = {"status": "PASS" if error < .01 else "FAIL", "detection": detection,
                  "scoring_ground_truth": truth.tolist(), "localization_error_m": error,
                  "calibration": calibration}
        (out / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if report["status"] != "PASS":
            raise SystemExit(1)
    finally:
        camera.close()


if __name__ == "__main__":
    main()

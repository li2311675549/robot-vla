#!/usr/bin/env python3
import json
import datetime
import argparse
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.skills.pick_place import PickPlace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["box", "small", "wide", "cylinder"], default="box")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    out = ROOT / "outputs/stage2/gate2" / args.variant
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.variant == "box" else "_" + args.variant
    sim = Simulation(ROOT / f"scenes/ur5_gripper{suffix}.xml")
    from robot_exam.evaluation.recording import VideoRecorder
    recorder = VideoRecorder(out / "video") if args.record else None
    skill = PickPlace(sim, recorder)
    try:
        halfheight = sim.model.geom("test_object").size[1 if args.variant == "cylinder" else 2]
        report = skill.run([.55, 0, .725 + halfheight], [.55, .22, .725 + halfheight])
    except Exception as exc:
        report = {"status": "FAIL", "error": str(exc), "sim_seconds": sim.data.time}
    finally:
        if recorder:
            recorder.close()
    report["timestamp"] = datetime.datetime.now().astimezone().isoformat()
    (out / "result.json").write_text(json.dumps(report, indent=2))
    (out / "trajectory.json").write_text(json.dumps(skill.rows))
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bounded interactive GUI check with real physics; closes automatically."""
import argparse
import json
import threading
import time

from _bootstrap import ROOT
import mujoco
import mujoco.viewer
import numpy as np
from robot_exam.simulation import Simulation, camera_for, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["minimal", "ur5", "robotiq"], default="ur5")
    parser.add_argument("--seconds", type=float, default=5)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error("seconds must be in (0, 60]")
    sim = Simulation(args.scene)
    initial = sim.data.ctrl.copy()
    target = targets(sim, args.scene)
    sync_count = 0
    start = time.monotonic()
    existing_threads = set(threading.enumerate())
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer_threads = set(threading.enumerate()) - existing_threads
        camera = camera_for(args.scene)
        with viewer.lock():
            if isinstance(camera, str):
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = sim.model.camera(camera).id
            else:
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance = camera.distance
                viewer.cam.azimuth = camera.azimuth
                viewer.cam.elevation = camera.elevation
        while viewer.is_running() and sim.data.time < args.seconds:
            # The viewer's mutex also protects model/data access by its render thread.
            with viewer.lock():
                fraction = min(1, sim.data.time)
                sim.step(initial + fraction * (target - initial))
            viewer.sync()
            sync_count += 1
            remaining = start + sim.data.time - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    # close() only requests asynchronous shutdown in MuJoCo 3.6.0. Waiting
    # prevents GLFW teardown racing a live rendering thread at interpreter exit.
    for thread in viewer_threads:
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("Viewer thread failed to stop cleanly")
    assert sim.data.time >= args.seconds - sim.model.opt.timestep
    assert sync_count > 0
    result = {"status": "PASS", "scene": args.scene, "sync_count": sync_count,
              "sim_seconds": sim.data.time, "wall_seconds": time.monotonic() - start,
              "render_threads_joined": len(viewer_threads),
              "evidence": "launch_passive opened; is_running and sync exercised; GUI threads joined"}
    path = ROOT / f"outputs/viewer/{args.scene}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

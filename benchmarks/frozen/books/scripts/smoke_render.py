#!/usr/bin/env python3
"""RGB/depth + actual controlled motion + bounded MP4, no model downloads."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

from _bootstrap import ROOT
import imageio.v3 as iio
import mujoco
import numpy as np
from robot_exam.simulation import Simulation, camera_for, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["minimal", "ur5", "robotiq"], default="minimal")
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/render")
    args = parser.parse_args()
    if not 0 < args.seconds <= 30:
        parser.error("seconds must be in (0, 30]")
    out = args.output_root.resolve() / args.scene
    out.mkdir(parents=True, exist_ok=True)
    sim = Simulation(args.scene)
    initial_ctrl = sim.data.ctrl.copy()
    target = targets(sim, args.scene)
    fps, width, height = 20, 640, 480
    frame_count = max(1, round(fps * args.seconds))
    camera = camera_for(args.scene)
    encoder = None
    ffmpeg_log = (out / "ffmpeg.log").open("w")
    try:
        if args.video:
            if not shutil.which("ffmpeg"):
                raise RuntimeError("ffmpeg missing; install separately or omit --video")
            encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                                        "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                        "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264",
                                        "-pix_fmt", "yuv420p", str(out / "demo.mp4")],
                                       stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=ffmpeg_log)
        with mujoco.Renderer(sim.model, height=height, width=width) as renderer:
            from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
            gl_info = {name: glGetString(enum).decode() for name, enum in
                       (("vendor", GL_VENDOR), ("renderer", GL_RENDERER), ("version", GL_VERSION))}
            renderer.update_scene(sim.data, camera=camera)
            first = renderer.render().copy()
            iio.imwrite(out / "initial.png", first)
            with (out / "trajectory.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sim_seconds", *sim.joint_positions()])
                for frame in range(frame_count):
                    end_step = round((frame + 1) / fps / sim.model.opt.timestep)
                    while sim.data.time < end_step * sim.model.opt.timestep - 1e-9:
                        fraction = min(1.0, (sim.data.time + sim.model.opt.timestep) / 1.0)
                        sim.step(initial_ctrl + fraction * (target - initial_ctrl))
                    renderer.update_scene(sim.data, camera=camera)
                    rgb = renderer.render().copy()
                    writer.writerow([sim.data.time, *sim.joint_positions().values()])
                    if encoder:
                        encoder.stdin.write(rgb.tobytes())
            iio.imwrite(out / "rgb.png", rgb)
            assert rgb.shape == (height, width, 3) and rgb.dtype == np.uint8
            assert rgb.std() > 8, "Blank/constant RGB"
            assert np.array_equal(iio.imread(out / "rgb.png"), rgb), "PNG round trip failed"
            assert np.mean(np.abs(rgb.astype(float) - first.astype(float))) > 0.1, "No visible motion"
            renderer.enable_depth_rendering()
            renderer.update_scene(sim.data, camera=camera)
            depth = renderer.render().copy()
            assert depth.shape == (height, width) and np.isfinite(depth).all()
            assert depth.min() > 0 and np.ptp(depth) > 0.01
            np.save(out / "depth_m.npy", depth)
            low, high = np.percentile(depth, [2, 98])
            preview = np.clip((depth - low) / max(float(high - low), 1e-6), 0, 1)
            iio.imwrite(out / "depth_preview.png", (preview * 255).astype(np.uint8))
        if encoder:
            encoder.stdin.close()
            if encoder.wait(timeout=30) != 0:
                raise RuntimeError("Video encoding failed; inspect ffmpeg.log")
            probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,nb_read_frames,duration", "-of", "json",
                                    str(out / "demo.mp4")], check=True, capture_output=True, text=True)
            video = json.loads(probe.stdout)["streams"][0]
            assert int(video["nb_read_frames"]) == frame_count
            assert (video["width"], video["height"]) == (width, height)
        else:
            video = None
        report = {"status": "PASS", "scene": args.scene,
                  "timestamp": datetime.datetime.now().astimezone().isoformat(),
                  "backend": os.environ.get("MUJOCO_GL", "glfw"), "opengl": gl_info,
                  "rgb_shape": list(rgb.shape), "rgb_std": float(rgb.std()),
                  "visible_change_mean": float(np.mean(np.abs(rgb.astype(float) - first.astype(float)))),
                  "depth_min_m": float(depth.min()), "depth_max_m": float(depth.max()),
                  "sim_seconds": sim.data.time, "frames": frame_count, "video": video,
                  "warnings": sim.data.warning.number.tolist()}
        (out / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        if encoder and encoder.poll() is None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.terminate()
            encoder.wait(timeout=10)
        ffmpeg_log.close()


if __name__ == "__main__":
    main()

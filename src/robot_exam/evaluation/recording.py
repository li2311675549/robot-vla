"""Continuous simulation-time recording, streamed to existing system ffmpeg."""
import json
from pathlib import Path
import subprocess

import imageio.v3 as iio
import mujoco
import numpy as np


class VideoRecorder:
    def __init__(self, output, camera=None, fps=20):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.camera = camera
        self.renderer = None
        self.encoder = None
        self.error_log = None
        self.frames = []
        self.next_time = None
        self.last_stage = None
        self.last_rgb = None
        self.gl = None

    def __call__(self, sim, stage):
        if self.next_time is None:
            # A checkpoint can start hundreds of simulated seconds after
            # zero. Never emit a burst of frames to catch an obsolete clock.
            self.next_time = float(sim.data.time)
        if sim.data.time + 1e-9 < self.next_time:
            return
        if self.renderer is None:
            self.renderer = mujoco.Renderer(sim.model, height=480, width=640)
            if self.camera is None:
                self.camera = mujoco.MjvCamera()
                self.camera.lookat[:] = [.85, 0, .65]
                self.camera.distance = 3.6
                self.camera.azimuth = 135
                self.camera.elevation = -25
            from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER
            self.gl = {"vendor": glGetString(GL_VENDOR).decode(), "renderer": glGetString(GL_RENDERER).decode()}
            self.error_log = (self.output / "ffmpeg.log").open("w")
            self.encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                             "-s", "640x480", "-r", str(self.fps), "-i", "-", "-an", "-c:v", "libx264",
                                             "-pix_fmt", "yuv420p", str(self.output / "demo.mp4")],
                                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.error_log)
        self.renderer.update_scene(sim.data, camera=self.camera)
        rgb = self.renderer.render().copy()
        if not rgb.std() > 8:
            raise RuntimeError("Blank video frame")
        self.encoder.stdin.write(rgb.tobytes())
        if stage != self.last_stage:
            iio.imwrite(self.output / f"{len(self.frames):05d}_{stage}.png", rgb)
        self.frames.append({"frame": len(self.frames), "sim_seconds": sim.data.time, "stage": stage})
        self.next_time += 1/self.fps
        self.last_stage = stage
        self.last_rgb = rgb

    def close(self):
        try:
            if self.encoder:
                self.encoder.stdin.close()
                if self.encoder.wait(timeout=30) != 0:
                    raise RuntimeError("Video encoding failed")
                probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                        "-show_entries", "stream=width,height,nb_read_frames,duration", "-of", "json",
                                        str(self.output / "demo.mp4")], capture_output=True, text=True, check=True)
                details = json.loads(probe.stdout)["streams"][0]
                if int(details["nb_read_frames"]) != len(self.frames):
                    raise RuntimeError("Video frame count mismatch")
                # Decode every encoded frame, so ffprobe metadata alone cannot pass.
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(self.output / "demo.mp4"),
                                "-f", "null", "-"], check=True, capture_output=True)
                iio.imwrite(self.output / "rgb.png", self.last_rgb)
                intervals=np.diff([f['sim_seconds'] for f in self.frames])
                cadence=bool(not len(intervals) or np.all(np.abs(intervals-1/self.fps)<.01))
                result = {"status": "PASS" if cadence else "FAIL", "fps": self.fps, "frames": len(self.frames), "stream": details,
                          "opengl": self.gl, "first_sim_seconds": self.frames[0]["sim_seconds"],
                          "last_sim_seconds": self.frames[-1]["sim_seconds"], "continuous": cadence,
                          "cadence_validated":cadence,"full_decode": "PASS"}
                (self.output / "video_result.json").write_text(json.dumps(result, indent=2))
                (self.output / "timeline.json").write_text(json.dumps(self.frames))
                return result
        finally:
            if self.encoder and self.encoder.poll() is None:
                self.encoder.terminate()
                self.encoder.wait(timeout=10)
            if self.renderer:
                self.renderer.close()
            if self.error_log:
                self.error_log.close()

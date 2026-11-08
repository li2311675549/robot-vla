"""Continuous H.264 recording with existing PyAV; no external ffmpeg executable.

Only the encoder backend differs from VideoRecorder. Render size, camera,
simulation-time cadence, screenshots, and full-frame decode checks are kept.
"""
import json

import imageio.v3 as iio
import mujoco
import numpy as np

from robot_exam.evaluation.recording import VideoRecorder


class PyAVVideoRecorder(VideoRecorder):
    def __init__(self, output, camera=None, fps=20):
        super().__init__(output,camera,fps)
        self.container=None
        self.stream=None

    def encode_rgb(self, rgb, sim_time, stage):
        import av
        if self.container is None:
            self.container=av.open(str(self.output/'demo.mp4'),'w')
            self.stream=self.container.add_stream('libx264',rate=int(self.fps))
            self.stream.width=640;self.stream.height=480;self.stream.pix_fmt='yuv420p'
        frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
        for packet in self.stream.encode(frame):self.container.mux(packet)
        if stage!=self.last_stage:iio.imwrite(self.output/f'{len(self.frames):05d}_{stage}.png',rgb)
        self.frames.append(dict(frame=len(self.frames),sim_seconds=float(sim_time),stage=stage))
        self.last_rgb=rgb;self.last_stage=stage

    def __call__(self, sim, stage):
        if self.next_time is None:self.next_time=float(sim.data.time)
        if sim.data.time+1e-9<self.next_time:return
        if self.renderer is None:
            self.renderer=mujoco.Renderer(sim.model,height=480,width=640)
            if self.camera is None:
                self.camera=mujoco.MjvCamera();self.camera.lookat[:]=[.85,0,.65]
                self.camera.distance=3.6;self.camera.azimuth=135;self.camera.elevation=-25
            from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER
            self.gl=dict(vendor=glGetString(GL_VENDOR).decode(),renderer=glGetString(GL_RENDERER).decode())
        self.renderer.update_scene(sim.data,camera=self.camera)
        rgb=self.renderer.render().copy()
        if not rgb.std()>8:raise RuntimeError('Blank video frame')
        self.encode_rgb(rgb,sim.data.time,stage)
        self.next_time+=1/self.fps

    def close(self):
        import av
        try:
            if self.container is None:return None
            for packet in self.stream.encode():self.container.mux(packet)
            self.container.close();self.container=None
            with av.open(str(self.output/'demo.mp4')) as container:
                stream=container.streams.video[0]
                count=sum(1 for _ in container.decode(stream))
                details=dict(width=stream.width,height=stream.height,nb_read_frames=str(count),
                             duration=count/self.fps)
            if count!=len(self.frames):raise RuntimeError('Video full-decode frame count mismatch')
            intervals=np.diff([f['sim_seconds'] for f in self.frames])
            cadence=bool(not len(intervals) or np.all(np.abs(intervals-1/self.fps)<.01))
            iio.imwrite(self.output/'rgb.png',self.last_rgb)
            result=dict(status='PASS' if cadence else 'FAIL',fps=self.fps,frames=count,stream=details,
                opengl=self.gl,first_sim_seconds=self.frames[0]['sim_seconds'],
                last_sim_seconds=self.frames[-1]['sim_seconds'],continuous=cadence,
                cadence_validated=cadence,full_decode='PASS',encoder_backend='pyav_libx264')
            (self.output/'video_result.json').write_text(json.dumps(result,indent=2)+'\n')
            (self.output/'timeline.json').write_text(json.dumps(self.frames)+'\n')
            return result
        finally:
            if self.container is not None:self.container.close()
            if self.renderer:self.renderer.close()

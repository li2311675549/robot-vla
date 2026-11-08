#!/usr/bin/env python3
"""Read-only rendering comparison at a stored physical initial state."""
import argparse,json
from pathlib import Path
import imageio.v3 as iio
import mujoco,numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.perception.rgbd import RGBDCamera

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--camera-near-m',type=float);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve renderer audits')
    a.output.mkdir(parents=True);raw=np.load(a.episode/'replay.npz');row=json.loads((a.episode/'frames.jsonl').read_text().splitlines()[0])
    sim=Simulation.__new__(Simulation);sim.model=mujoco.MjModel.from_binary_path(str(a.episode/'model.mjb'));sim.data=mujoco.MjData(sim.model);sim.step_controllers=[]
    mujoco.mj_setState(sim.model,sim.data,raw['initial_state'],int(raw['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    original_near=float(sim.model.vis.map.znear*sim.model.stat.extent)
    if a.camera_near_m is not None:
        if not .001<=a.camera_near_m<=.02:raise ValueError('Versioned virtual-camera near plane must be1–20mm')
        sim.model.vis.map.znear=a.camera_near_m/sim.model.stat.extent
    result={'scope':'same saved initial state, renderer comparison only; no task operation','source_episode':str(a.episode),'views':{},
            'source_near_m':original_near,'render_near_m':float(sim.model.vis.map.znear*sim.model.stat.extent)}
    for view,name,key in [('front','d435i_rgbd','rgb'),('wrist','wrist_rgb','rgb_wrist')]:
        camera=RGBDCamera(sim,camera_name=name)
        rgb=camera.capture_rgb();original=iio.imread(a.episode/row[key]);delta=abs(rgb.astype(float)-original.astype(float))
        from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER
        result['opengl']={'vendor':glGetString(GL_VENDOR).decode(),'renderer':glGetString(GL_RENDERER).decode()}
        iio.imwrite(a.output/(view+'.png'),rgb)
        result['views'][view]={'mean_absolute_byte_error':float(delta.mean()),'fraction_channels_gt10':float((delta>10).mean()),'max_byte_error':float(delta.max())}
        camera.close()
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

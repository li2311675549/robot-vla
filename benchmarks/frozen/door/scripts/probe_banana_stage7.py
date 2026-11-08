#!/usr/bin/env python3
"""Local physical banana grasp/hold/reverse probe, independent of A* dock."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from build_tasks import build
from robot_exam.perception.objects import observe_objects
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.simulation import Simulation
from robot_exam.tasks.manipulation import pick
from robot_exam.tasks.runtime import State, TaskRuntime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=412)
    parser.add_argument('--speed',type=float,default=.06)
    parser.add_argument('--acceleration',type=float,default=.35)
    parser.add_argument('--distance',type=float,default=.35)
    parser.add_argument('--candidate-index',type=int,default=0)
    parser.add_argument('--grasp-height-offset',type=float,default=0.)
    parser.add_argument('--pad-sliding-friction',type=float,default=.7)
    parser.add_argument('--holding-closure',type=float,default=1.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    scene,metadata=build('fruit',args.seed,output/'scene.xml')
    sim=Simulation(scene)
    for side in ('left','right'):
        for part in (1,2):
            geom=sim.model.geom(f'grip_{side}_pad{part}').id
            sim.model.geom_friction[geom,0]=args.pad_sliding_friction
    # Fixture initialization only, before first physics step. No free-body
    # state is assigned during grasp, hold or transport.
    sim.data.qpos[:2]=[0.,0.]
    mujoco.mj_forward(sim.model,sim.data)
    runtime=TaskRuntime(sim,output=output,wall_limit=1800,sim_limit=1800)
    result={'seed':args.seed,'speed_m_s':args.speed,'acceleration_rad_s2':args.acceleration,
            'candidate_index':args.candidate_index,
            'grasp_height_offset_m':args.grasp_height_offset,
            'pad_sliding_friction':args.pad_sliding_friction,
            'holding_closure':args.holding_closure,
            'distance_m':args.distance,'body':None,'grasp_verified':False,
            'static_hold_verified':False,'transport_success':False,'failure':None}
    try:
        runtime.hold(.6)
        camera=RGBDCamera(sim)
        try:
            rgb,depth=camera.capture()
            detections=[d for d in observe_objects(camera,rgb,depth) if d['category']=='banana']
        finally:camera.close()
        if not detections:raise RuntimeError('No banana detected from the work pose')
        detection=detections[0].copy()
        center=np.asarray(detection['center'],dtype=float).copy()
        center[2]+=args.grasp_height_offset
        detection['center']=center.tolist()
        bananas=[o for o in metadata['objects'] if o['category']=='banana']
        selected=min(bananas,key=lambda o:np.linalg.norm(
            sim.data.xpos[sim.model.body(o['body']).id]-detection['center']))
        body=selected['body'];result['body']=body
        grasp=pick(runtime,body,detection,candidate_index=args.candidate_index)
        result['grasp']=grasp
        result['grasp_verified']=bool(runtime.stable_grasp(body,seconds=.5))
        if not result['grasp_verified']:raise RuntimeError('Physical banana grasp not stable after lift')
        runtime.gripper.command(args.holding_closure)
        runtime.transition(State.NAVIGATE,'static_banana_hold',timeout=15)
        runtime.base.command(0,0);runtime.begin_payload(body,np.asarray(grasp['rotation']))
        runtime.payload_motion_profile='banana'
        runtime.hold(3.)
        result['static_hold_verified']=bool(runtime.stable_grasp(body,seconds=.5))
        if not result['static_hold_verified']:raise RuntimeError('Banana lost during static hold')
        runtime.config['payload_banana_linear_speed_m_s']=args.speed
        runtime.config['payload_banana_wheel_acceleration_rad_s2']=args.acceleration
        runtime.transition(State.NAVIGATE,'physical_reverse_transport',timeout=60)
        initial_x=float(runtime.base.pose()[0]);start=float(sim.data.time)
        while float(runtime.base.pose()[0])>initial_x-args.distance:
            if sim.data.time-start>45:raise RuntimeError('Reverse motion timed out')
            runtime.base.command(-args.speed,0)
            runtime.tick()
        runtime.base.command(0,0);runtime.hold(2.)
        result['transport_success']=bool(runtime.stable_grasp(body,seconds=.5))
        result['transport_sim_seconds']=float(sim.data.time-start)
        result['transported_distance_m']=float(initial_x-runtime.base.pose()[0])
    except Exception as exc:
        result['failure']={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
    finally:
        if runtime.payload:runtime.end_payload()
        runtime.save()
        result['final_base_pose']=runtime.base.pose().tolist()
        result['warnings']=sim.data.warning.number.tolist()
        (output/'result.json').write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x))
        print(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x))


if __name__=='__main__':main()

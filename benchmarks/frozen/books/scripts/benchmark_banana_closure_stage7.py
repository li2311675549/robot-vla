#!/usr/bin/env python3
"""Paired fruit transport from physical grasp checkpoints, varying hold command."""
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
from robot_exam.control.base import wrap_angle
from robot_exam.evaluation.recording import VideoRecorder


def fixture(seed,folder,category='banana',candidate_index=0):
    folder.mkdir(parents=True,exist_ok=True)
    scene,metadata=build('fruit',seed,folder/'scene.xml')
    sim=Simulation(scene)
    sim.data.qpos[:2]=[0.,0.]  # Fixture initialization before first step.
    mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=folder/'physical_grasp',wall_limit=1800,sim_limit=1800)
    r.hold(.6)
    camera=RGBDCamera(sim)
    try:
        rgb,depth=camera.capture()
        detections=[d for d in observe_objects(camera,rgb,depth) if d['category']==category]
    finally:camera.close()
    if not detections:raise RuntimeError(f'No visible {category}')
    detection=detections[0]
    selected=min(metadata['objects'],key=lambda o:np.linalg.norm(
        sim.data.xpos[sim.model.body(o['body']).id]-detection['center']))
    if selected['category']!=category:
        raise RuntimeError(f'Vision {category} mapped to {selected["category"]}; fixture rejected')
    body=selected['body']
    grasp=pick(r,body,detection,candidate_index=candidate_index)
    if not r.stable_grasp(body,seconds=.5):raise RuntimeError('Physical grasp did not stabilize')
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(sim.model,spec))
    mujoco.mj_getState(sim.model,sim.data,state,spec)
    np.savez_compressed(folder/'initial_state.npz',integration_state=state,state_spec=int(spec),
                        ctrl=sim.data.ctrl.copy(),qgoal=r.qgoal.copy(),
                        arm_reference=r.arm.reference.copy(),wheel_applied=r.wheel_applied.copy())
    info={'seed':seed,'category':category,'candidate_index':candidate_index,
          'body':body,'rotation':grasp['rotation'].tolist(),
          'arm_last_successful':[q.tolist() for q in r.arm.last_successful],
          'physical_lift_m':grasp['lift_m'],'initial_base':r.base.pose().tolist(),
          'initial_bilateral_contact':r.bilateral(body)}
    (folder/'fixture.json').write_text(json.dumps(info,indent=2))
    r.save()
    return scene,info


def trial(scene,folder,info,closure,output,turn_radians=0.,record=False,turn_controller='direct'):
    output.mkdir(parents=True,exist_ok=True)
    arrays=np.load(folder/'initial_state.npz')
    sim=Simulation(scene)
    mujoco.mj_setState(sim.model,sim.data,arrays['integration_state'],int(arrays['state_spec']))
    sim.data.ctrl[:]=arrays['ctrl']
    mujoco.mj_forward(sim.model,sim.data)
    recorder=None
    if record:
        camera=mujoco.MjvCamera();camera.lookat[:]=[-.12,0,.78]
        camera.distance=2.1;camera.azimuth=145;camera.elevation=-24
        recorder=VideoRecorder(output/'video',camera=camera,fps=5)
        recorder.next_time=float(sim.data.time)
    r=TaskRuntime(sim,output=output,callback=recorder,wall_limit=1800,sim_limit=1800)
    r.qgoal=arrays['qgoal'].copy()
    r.arm.reference=arrays['arm_reference'].copy()
    r.arm.last_successful=[np.asarray(q) for q in info['arm_last_successful']]
    r.wheel_applied=arrays['wheel_applied'].copy()
    category=info.get('category','banana')
    if category=='banana':
        r.config['payload_banana_linear_speed_m_s']=.06
        r.config['payload_banana_wheel_acceleration_rad_s2']=.35
    body=info['body'];result={'seed':info['seed'],'body':body,'holding_closure':closure,
                              'turn_target_rad':turn_radians,'turn_controller':turn_controller,
                              'transport_success':False,'failure':None}
    start=float(sim.data.time);initial_x=float(r.base.pose()[0])
    try:
        r.gripper.command(closure)
        r.transition(State.NAVIGATE,'paired_static_hold',timeout=60)
        r.base.command(0,0)
        r.payload_motion_profile=category
        r.begin_payload(body,np.asarray(info['rotation']))
        r.hold(3.)
        if not r.stable_grasp(body,seconds=.5):raise RuntimeError('Static hold lost physical grasp')
        r.transition(State.NAVIGATE,'paired_reverse_transport',timeout=60)
        while r.base.pose()[0]>initial_x-.35:
            if sim.data.time-start>45:raise RuntimeError('Reverse transport timeout')
            r.base.command(-.06,0);r.tick()
        if turn_radians:
            yaw_start=float(r.base.pose()[2]);turn_start=float(sim.data.time)
            direction=float(np.sign(turn_radians))
            anchor=r.base.pose()[:2].copy()
            while direction*wrap_angle(float(r.base.pose()[2])-yaw_start)<abs(turn_radians):
                if sim.data.time-turn_start>35:raise RuntimeError('Loaded turn timed out')
                if turn_controller=='rotate':r.base.rotate(anchor,yaw_start+turn_radians)
                else:r.base.command(0,direction*.08)
                r.tick()
        r.base.command(0,0);r.hold(2.)
        result['transport_success']=bool(r.stable_grasp(body,seconds=.5))
        if not result['transport_success']:raise RuntimeError('Terminal bilateral grasp unstable')
    except Exception as exc:
        result['failure']={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
    finally:
        result['elapsed_sim_seconds']=float(sim.data.time-start)
        result['travel_m']=float(initial_x-r.base.pose()[0])
        result['terminal_base_pose']=r.base.pose().tolist()
        result['terminal_bilateral']=bool(r.bilateral(body))
        result['warnings']=sim.data.warning.number.tolist()
        result['collision_events']=len(r.collision_events)
        if r.payload_trace:
            result['max_relative_drift_m']=max(x['relative_error_m'] for x in r.payload_trace)
            result['max_relative_rotation_rad']=max(x['relative_rotation_error_rad'] for x in r.payload_trace)
        if r.payload:r.end_payload()
        r.save()
        if recorder:
            try:result['video']=recorder.close()
            except Exception as exc:result['video_error']=str(exc)
        (output/'result.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed-start',type=int,default=420)
    parser.add_argument('--cases',type=int,default=10)
    parser.add_argument('--turn',type=float,default=0.)
    parser.add_argument('--turn-controller',choices=['direct','rotate'],default='direct')
    parser.add_argument('--closures',type=float,nargs='+',default=[1.,.7])
    parser.add_argument('--category',choices=['banana','apple','orange'],default='banana')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in range(args.seed_start,args.seed_start+args.cases):
        folder=root/f'seed_{seed}'
        try:
            if args.category=='apple':
                candidate_errors=[]
                for candidate_index in range(6):
                    trial_folder=folder/f'candidate_{candidate_index}'
                    try:
                        scene,info=fixture(seed,trial_folder,category=args.category,
                                           candidate_index=candidate_index)
                        folder=trial_folder
                        break
                    except Exception as candidate_error:
                        candidate_errors.append(str(candidate_error))
                else:raise RuntimeError(f'No stable apple grasp among 6 candidates: {candidate_errors}')
            else:scene,info=fixture(seed,folder,category=args.category)
            for closure in args.closures:
                label='full' if closure==1. else 'regulated'
                item=trial(scene,folder,info,closure,folder/label,
                           turn_radians=args.turn,turn_controller=args.turn_controller)
                records.append(item)
                print(seed,label,item['transport_success'],round(item['travel_m'],3),item['failure'],flush=True)
        except Exception as exc:
            entry={'seed':seed,'fixture_failure':str(exc)}
            records.append(entry)
            (folder/'fixture_failure.json').write_text(json.dumps(entry,indent=2))
            print(seed,'fixture failure',str(exc),flush=True)
        (root/'episodes.json').write_text(json.dumps(records,indent=2))
    summary={}
    for closure in args.closures:
        subset=[x for x in records if x.get('holding_closure')==closure]
        label='full' if closure==1. else 'regulated'
        summary[label]={'episodes':len(subset),'physical_transport_successes':sum(x['transport_success'] for x in subset),
                        'mean_max_drift_m':float(np.mean([x.get('max_relative_drift_m',float('nan')) for x in subset])) if subset else None}
    summary['fixture_failures']=sum('fixture_failure' in x for x in records)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Paired physical dock trials from identical MuJoCo and controller states.

The only qpos assignment sets up an independent empty/pick fixture before its
first physics step. Loaded fixtures perform a real finger grasp and lift. Each
controller trial restores the same saved post-settle integration state; during
the trial only wheel/arm/gripper actuators move the system.
"""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from build_tasks import build
from robot_exam.perception.objects import observe_objects
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.simulation import Simulation
from robot_exam.tasks.manipulation import pick
from robot_exam.tasks.runtime import State, TaskFailure, TaskRuntime
from robot_exam.control.base import wrap_angle


def fixture(index, kind, output):
    output.mkdir(parents=True, exist_ok=True)
    seed=700+index
    scene,meta=build('fruit',seed,output/'scene.xml')
    sim=Simulation(scene)
    if kind=='empty':
        # Initial condition only: no robot or fruit pose is assigned during a
        # dock trial. The robot starts behind the table work position.
        # Existing A* task logs hand control to dock near x=-0.45 m, with
        # lateral offsets around 0.00-0.04 m and heading within ~0.03 rad.
        # This matrix spans that measured handoff envelope rather than an
        # unrelated sideways start already beside the target.
        sim.data.qpos[:2]=[-.48+.008*index,-.032+.007*index]
        yaw=-.07+.015*index
        sim.data.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        mujoco.mj_forward(sim.model,sim.data)
        target=[0.,0.,0.]
        body=None;rotation=None;grasp=None
    else:
        # Match the loaded goal matrix at base pose zero. For loaded trials
        # this initial placement permits a physical grasp without repeating
        # the five metre drive; empty_lateral uses the same geometric targets
        # without a payload to isolate reachability from payload control.
        sim.data.qpos[:2]=[0.,0.]
        mujoco.mj_forward(sim.model,sim.data)
        target=[-.14-.01*(index%3),-.06+.015*index,-.18+.04*index]
        body=None;rotation=None;grasp=None
    runtime=TaskRuntime(sim,output=output/'fixture',wall_limit=1800,sim_limit=1800)
    runtime.hold(.6)
    if kind=='loaded':
        camera=RGBDCamera(sim)
        try:
            rgb,depth=camera.capture()
            detections=[d for d in observe_objects(camera,rgb,depth) if d['category']=='orange']
        finally:camera.close()
        if not detections:raise RuntimeError('No visible orange for physical load fixture')
        detection=detections[0]
        objects=[o for o in meta['objects'] if o['category']=='orange']
        selected=min(objects,key=lambda o:np.linalg.norm(
            sim.data.xpos[sim.model.body(o['body']).id]-detection['center']))
        body=selected['body']
        grasp=pick(runtime,body,detection)
        if not runtime.stable_grasp(body,seconds=.3):
            raise RuntimeError('Loaded fixture did not physically retain bilateral contact')
        rotation=grasp['rotation']
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(sim.model,spec))
    mujoco.mj_getState(sim.model,sim.data,state,spec)
    snapshot={'integration_state':state,'state_spec':int(spec),
              'ctrl':sim.data.ctrl.copy(),'qgoal':runtime.qgoal.copy(),
              'arm_reference':runtime.arm.reference.copy(),
              'wheel_applied':runtime.wheel_applied.copy(),
              'arm_last_successful':[q.tolist() for q in runtime.arm.last_successful],
              'body':body,'rotation':None if rotation is None else rotation.tolist(),
              'target':target,'seed':seed,'grasp':None if grasp is None else {
                  'lift_m':grasp['lift_m'],'candidate_count':grasp['candidate_count']},
              'initial_base':runtime.base.pose().tolist(),
              'initial_bilateral_contact':None if body is None else runtime.bilateral(body)}
    np.savez_compressed(output/'initial_state.npz',integration_state=state,
                        state_spec=int(spec),ctrl=snapshot['ctrl'],qgoal=snapshot['qgoal'],
                        arm_reference=snapshot['arm_reference'],wheel_applied=snapshot['wheel_applied'])
    (output/'fixture.json').write_text(json.dumps({k:v for k,v in snapshot.items()
        if k not in ('integration_state','ctrl','qgoal','arm_reference','wheel_applied')},indent=2))
    return scene,snapshot


def trial(scene,snapshot,strategy,output,kind,timeout=25.,config_overrides=None):
    output.mkdir(parents=True,exist_ok=True)
    sim=Simulation(scene)
    mujoco.mj_setState(sim.model,sim.data,snapshot['integration_state'],snapshot['state_spec'])
    sim.data.ctrl[:]=snapshot['ctrl']
    mujoco.mj_forward(sim.model,sim.data)
    runtime=TaskRuntime(sim,output=output,wall_limit=1800,sim_limit=1800)
    runtime.config=json.loads((ROOT/'configs/task_defaults.json').read_text())
    if config_overrides:runtime.config.update(config_overrides)
    if strategy=='residual':runtime.config['dock_residual_reentry_seconds']=.6
    if strategy=='signed':
        runtime.config['dock_residual_reentry_seconds']=.6
        runtime.config['dock_reverse_lateral_sign']=True
    runtime.qgoal=snapshot['qgoal'].copy()
    runtime.arm.reference=snapshot['arm_reference'].copy()
    runtime.arm.last_successful=[np.asarray(q) for q in snapshot['arm_last_successful']]
    runtime.wheel_applied=snapshot['wheel_applied'].copy()
    runtime.base.command(0,0)
    if kind=='loaded':
        runtime.payload_motion_profile='orange'
        runtime.begin_payload(snapshot['body'],np.asarray(snapshot['rotation']))
    runtime.transition(State.NAVIGATE,'paired_stage7_dock',timeout=max(45.,timeout+5.))
    start=sim.data.time;error=None;returned=False
    settled_samples=[]
    try:
        if strategy=='legacy':runtime.dock_legacy(snapshot['target'],timeout=timeout,tolerance=.023)
        else:runtime.dock(snapshot['target'],timeout=timeout,tolerance=.023)
        returned=True
        # The same independent post-command observation is applied to both
        # strategies. The new controller additionally checks stability inside.
        runtime.base.command(0,0)
        for _ in range(100):
            runtime.hold(.02)
            observed=runtime.base.pose()
            velocity=sim.data.qvel[runtime.base_velocity_dof:runtime.base_velocity_dof+6]
            settled_samples.append({'time':float(sim.data.time),
                'position_error_m':float(np.linalg.norm(observed[:2]-snapshot['target'][:2])),
                'heading_error_rad':float(abs(wrap_angle(observed[2]-snapshot['target'][2]))),
                'linear_speed_m_s':float(np.linalg.norm(velocity[:2])),
                'yaw_rate_rad_s':float(abs(velocity[5]))})
    except Exception as exc:
        error={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
    finally:
        runtime.base.command(0,0)
        if kind=='loaded':runtime.end_payload()
        runtime.save()
    pose=runtime.base.pose();goal=np.asarray(snapshot['target'])
    v=sim.data.qvel[runtime.base_velocity_dof:runtime.base_velocity_dof+6]
    position_error=float(np.linalg.norm(pose[:2]-goal[:2]))
    heading_error=float(abs(wrap_angle(pose[2]-goal[2])))
    linear_speed=float(np.linalg.norm(v[:2]));yaw_rate=float(abs(v[5]))
    bilateral=None if snapshot['body'] is None else runtime.bilateral(snapshot['body'])
    warnings=sim.data.warning.number.tolist()
    contacts=[]
    for contact in sim.data.contact:
        if contact.dist>0:continue
        contacts.append({'geoms':[sim.model.geom(int(g)).name for g in contact.geom],
            'distance_m':float(contact.dist)})
    stable_2s=(len(settled_samples)==100 and all(
        x['position_error_m']<.023 and x['heading_error_rad']<.035
        and x['linear_speed_m_s']<.02 and x['yaw_rate_rad_s']<.03
        for x in settled_samples))
    strict=bool(returned and stable_2s and position_error<.023 and heading_error<.035
        and linear_speed<.02 and yaw_rate<.03 and not runtime.collision_events
        and not any(warnings) and bilateral is not False)
    result={'kind':kind,'strategy':strategy,'seed':snapshot['seed'],
            'controller_returned':returned,'strict_success_after_2s':strict,
            'continuous_stable_2s':stable_2s,'post_command_samples':settled_samples,
            'position_error_m':position_error,'heading_error_rad':heading_error,
            'linear_speed_m_s':linear_speed,'yaw_rate_rad_s':yaw_rate,
            'elapsed_sim_seconds':float(sim.data.time-start),'bilateral_contact':bilateral,
            'collision_events':len(runtime.collision_events),'warnings':warnings,'error':error,
            'terminal_contacts':contacts,'body_height_m':float(sim.data.xpos[sim.model.body('husky').id,2]),
            'initial_base':snapshot['initial_base'],'goal':snapshot['target']}
    (output/'result.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/stage7/dock_pairs')
    parser.add_argument('--cases',type=int,default=10)
    parser.add_argument('--timeout',type=float,default=25.)
    parser.add_argument('--kinds',nargs='+',choices=['empty','empty_lateral','loaded'],default=['empty','loaded'])
    parser.add_argument('--strategies',nargs='+',choices=['legacy','hysteresis','residual','signed'],default=['legacy','hysteresis'])
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for kind in args.kinds:
        for index in range(args.cases):
            folder=args.output/kind/f'case_{index:02d}'
            try:
                scene,snapshot=fixture(index,kind,folder)
                for strategy in args.strategies:
                    result=trial(scene,snapshot,strategy,folder/strategy,kind,timeout=args.timeout)
                    result['case_index']=index;records.append(result)
                    print(kind,index,strategy,result['strict_success_after_2s'],
                          result['position_error_m'],result['heading_error_rad'],
                          result['error'],flush=True)
            except Exception as exc:
                entry={'kind':kind,'case_index':index,'fixture_failure':str(exc)}
                records.append(entry);(folder/'fixture_failure.json').write_text(json.dumps(entry,indent=2))
                print(kind,index,'fixture failure',exc,flush=True)
            (args.output/'episodes.json').write_text(json.dumps(records,indent=2))
    summary={}
    for kind in args.kinds:
        for strategy in args.strategies:
            subset=[r for r in records if r.get('kind')==kind and r.get('strategy')==strategy]
            summary[f'{kind}/{strategy}']={'episodes':len(subset),
                'strict_successes':sum(r['strict_success_after_2s'] for r in subset),
                'mean_position_error_m':float(np.mean([r['position_error_m'] for r in subset])) if subset else None,
                'mean_heading_error_rad':float(np.mean([r['heading_error_rad'] for r in subset])) if subset else None,
                'mean_elapsed_sim_seconds':float(np.mean([r['elapsed_sim_seconds'] for r in subset])) if subset else None}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

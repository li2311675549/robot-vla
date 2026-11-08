#!/usr/bin/env python3
"""Pair stored expert actions under exact deployment prefix/score semantics.

Reset is only episode initialization. Every subsequent motion uses ctrl.
Writes independent evidence; never alters historical replay/acceptance files.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import _bootstrap
from _bootstrap import ROOT


def run_case(entry,mode,output,live_start=False):
    import mujoco
    import numpy as np
    from robot_exam.simulation import Simulation
    from robot_exam.control.arm import ArmController,GripperController
    from robot_exam.control.parking import WheelParkingHold
    from robot_exam.vla.interface import ActionChunkExecutor
    folder=Path(entry['path']);case=Path(output)/'cases'/f"{entry['seed']}_{entry['skill']}_{mode}"
    case.mkdir(parents=True);os.chdir(case)
    raw=np.load(folder/'replay.npz');model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
    data=mujoco.MjData(model)
    if live_start:mujoco.mj_resetDataKeyframe(model,data,0)
    else:mujoco.mj_setState(model,data,raw['initial_state'],int(raw['state_spec']))
    mujoco.mj_forward(model,data)
    sim=Simulation.__new__(Simulation);sim.model=model;sim.data=data;sim.step_controllers=[]
    if live_start:
        # EXACT evaluation admission: 500 default-control steps, then1100
        # encoder-hold steps; no extra gravity-compensated arm warm-start.
        sim.step(data.ctrl,500);hold=WheelParkingHold(sim);hold.begin()
        for _ in range(1100):sim.step(data.ctrl)
        arm=ArmController(sim);grip=GripperController(sim)
    else:
        arm=ArmController(sim);grip=GripperController(sim)
        if 'initial_arm_reference' in raw:arm.reference[:]=raw['initial_arm_reference']
        hold=WheelParkingHold(sim);hold.begin()
        for _ in range(1100):arm.command(arm.reference);sim.step(data.ctrl)
    assert hold.ready,'Parking admission failed'
    executor=ActionChunkExecutor(sim,arm,grip,base_hold=hold)
    rows=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
    actions=np.asarray([r['action'] for r in rows]);controls=raw['controls']
    recorded_qpos=raw['qpos'];initial_difference=float(np.max(np.abs(data.qpos-recorded_qpos[0])))
    if live_start and initial_difference>1e-9:raise RuntimeError(f'Live initial state differs from demonstration:{initial_difference}')
    body=model.body('object').id;dof=model.jnt_dofadr[model.joint('object_free').id]
    initial=float(data.xpos[body,2]);start=float(data.time);wall=time.monotonic()
    best=current=maximum=0.;last=-1;error=None;collisions=[]
    tracking_squared=np.zeros(6);tracking_max=np.zeros(6);tracking_steps=0
    robot=set();base=model.body('husky').id
    for b in range(model.nbody):
        parent=b
        while parent and parent!=base:parent=int(model.body_parentid[parent])
        if parent==base:robot.add(b)
    body_names=[model.body(b).name or '' for b in range(model.nbody)]
    geom_names=[model.geom(g).name or '' for g in range(model.ngeom)]
    try:
        for i,ctrl in enumerate(controls):
            if i%100==0 and time.monotonic()-wall>180:raise TimeoutError('Bounded physical replay')
            frame=min(int((data.time-start+1e-8)*5),len(actions)-1)
            if frame!=last:
                executor.submit(actions[frame:frame+(1 if mode=='sliced1' else 10)],data.time);last=frame
            delta=data.qpos[arm.qadr]-recorded_qpos[i,arm.qadr]
            tracking_squared+=delta**2;tracking_max=np.maximum(tracking_max,np.abs(delta));tracking_steps+=1
            data.ctrl[:]=ctrl  # Executor overwrites arm/gripper; encoder parking overwrites wheels.
            executor.step()
            lift=float(data.xpos[body,2]-initial);maximum=max(maximum,lift)
            current=current+model.opt.timestep if lift>.05 and grip.bilateral() else 0.;best=max(best,current)
            for contact in data.contact:
                if contact.dist>=-.003:continue
                g1,g2=map(int,contact.geom);b1,b2=map(int,model.geom_bodyid[[g1,g2]])
                if b1 not in robot and b2 not in robot:continue
                names=[body_names[b1],body_names[b2]]
                if 'floor' in (geom_names[g1],geom_names[g2]) or b1==b2:continue
                if all(n.startswith('grip_') for n in names):continue
                if 'object' in names and any(n.startswith('grip_') for n in names):continue
                collisions.append({'time':float(data.time),'bodies':names,'depth_m':-float(contact.dist)})
                raise RuntimeError('Unexpected nonfinger robot penetration')
    except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
    goal=np.asarray(json.loads((folder.parent/'skill_result.json').read_text())['drop'])
    pos=data.xpos[body].copy();speed=float(np.linalg.norm(data.qvel[dof:dof+3]))
    xy=float(np.linalg.norm(pos[:2]-goal[:2]));z=abs(float(pos[2]-goal[2]));released=not grip.contacts()
    success=bool(error is None and maximum>.055 and best>=1.5 and xy<.04 and z<.01 and speed<.01
                 and released and not data.warning.number.any() and not hold.faulted)
    result={'success':success,'mode':mode,'seed':entry['seed'],'skill':entry['skill'],'split':entry['split'],
        'data_origin':entry.get('data_origin'),'source':str(folder),'source_action_frames':len(rows),
        'maximum_lift_m':maximum,'continuous_bilateral_hold_seconds':best,'placement_xy_m':xy,'placement_z_m':z,
        'final_speed_m_s':speed,'released':released,'parking_fault':hold.faulted,'warnings':data.warning.number.tolist(),
        'unexpected_contacts':collisions,'error':error,'wall_seconds':time.monotonic()-wall,'sim_seconds':float(data.time-start),
        'initialization':'exact_live_keyframe500_then_encoder1100' if live_start else 'historical_replay_extra_arm_gravity_settle1100',
        'initial_qpos_difference_from_expert':initial_difference,
        'joint_state_rms_error_from_expert_rad':np.sqrt(tracking_squared/max(tracking_steps,1)).tolist(),
        'joint_state_max_error_from_expert_rad':tracking_max.tolist(),
        'scope':'Recorded expert targets through real actuators; exact current stationary physical score; not learned VLA inference.'}
    (case/'result.json').write_text(json.dumps(result,indent=2)+'\n');hold.release();raw.close();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--modes',nargs='+',choices=('sliced1','lookahead10'),default=['sliced1','lookahead10'])
    p.add_argument('--live-start',action='store_true',help='Match actual evaluation admission and require identical demonstration initial qpos')
    a=p.parse_args();a.output=a.output.resolve()
    if a.output.exists():raise FileExistsError('Preserve every attempted audit')
    if not 1<=a.workers<=2:p.error('Bound local CPU workers to1..2')
    a.output.mkdir(parents=True)
    entries=json.loads(a.manifest.read_text())['accepted']
    files=[ROOT/'src/robot_exam'/n for n in ('simulation.py','control/arm.py','control/parking.py','vla/interface.py')]+[Path(__file__).resolve(),a.manifest]
    hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    (a.output/'freeze.json').write_text(json.dumps({'source_hashes':hashes,'modes':a.modes,'live_start':a.live_start,'cases':len(entries)*len(a.modes)},indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_case,e,mode,str(a.output),a.live_start):(e,mode) for e in entries for mode in a.modes}
        for future in as_completed(futures):
            entry,mode=futures[future]
            try:result=future.result()
            except Exception as exc:result={'success':False,'seed':entry['seed'],'skill':entry['skill'],'mode':mode,'data_origin':entry.get('data_origin'),'error':{'type':type(exc).__name__,'message':str(exc)}}
            results.append(result)
            summary={m:{'attempts':sum(r['mode']==m for r in results),'successes':sum(r['mode']==m and r['success'] for r in results)} for m in a.modes}
            (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            (a.output/'summary.json').write_text(json.dumps({'completed':len(results),'expected':len(futures),'modes':summary},indent=2)+'\n')
            print(len(results),result['seed'],result['skill'],mode,result['success'],result.get('error'),flush=True)
    mismatches=[str(f) for f in files if hashlib.sha256(f.read_bytes()).hexdigest()!=hashes[str(f)]]
    (a.output/'integrity.json').write_text(json.dumps({'source_mismatches':mismatches,'all_cases_recorded':len(results)==len(entries)*len(a.modes)},indent=2)+'\n')


if __name__=='__main__':main()

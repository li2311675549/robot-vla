#!/usr/bin/env python3
"""Replay recorded 5Hz goals with the SAME acceleration-limited VLA executor."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.vla.interface import ActionChunkExecutor
from robot_exam.simulation import Simulation


def replay_episode(path):
    meta=json.loads((path/'episode.json').read_text())
    frames=[json.loads(x) for x in (path/'frames.jsonl').read_text().splitlines()]
    raw=np.load(path/'replay.npz');controls=raw['controls'];initial=raw['initial_state']
    model=mujoco.MjModel.from_binary_path(str(path/'model.mjb'));data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,initial,int(raw['state_spec']));mujoco.mj_forward(model,data)
    sim=Simulation.__new__(Simulation);sim.model=model;sim.data=data;sim.step_controllers=[]
    arm=ArmController(sim);gripper=GripperController(sim)
    if 'initial_arm_reference' in raw:arm.reference[:]=raw['initial_arm_reference']
    hold=None
    if meta.get('base_control')=='encoder_parking_v1':
        from robot_exam.control.parking import WheelParkingHold
        hold=WheelParkingHold(sim);hold.begin()
        for _ in range(1100):arm.command(arm.reference);sim.step(data.ctrl)
    executor=ActionChunkExecutor(sim,arm,gripper,base_hold=hold)
    actions=np.asarray([x['action'] for x in frames]);start=float(data.time)
    body=model.body('object').id;initial_height=float(data.xpos[body,2])
    maximum_lift=0.;bilateral_lift_samples=0;last_frame=-1
    for index,ctrl in enumerate(controls):
        frame=min(int((data.time-start+1e-8)*5),len(actions)-1)
        if frame!=last_frame:
            executor.submit(actions[frame:frame+10],data.time);last_frame=frame
        data.ctrl[:]=ctrl # recorded non-policy channels only; executor overwrites all arm/gripper channels
        executor.step()
        if index%25==0:
            lift=float(data.xpos[body,2]-initial_height);maximum_lift=max(maximum_lift,lift)
            if lift>.05 and gripper.bilateral():bilateral_lift_samples+=1
    expected=json.loads((path.parent/'skill_result.json').read_text())
    dof=model.jnt_dofadr[model.joint('object_free').id]
    position=data.xpos[body].copy();goal=np.asarray(expected['drop'])
    error=float(np.linalg.norm(position[:2]-goal[:2]));speed=float(np.linalg.norm(data.qvel[dof:dof+3]))
    passed=bool(error<.04 and abs(position[2]-goal[2])<.01 and speed<.01
                and maximum_lift>.055 and bilateral_lift_samples>=5 and not gripper.contacts()
                and not data.warning.number.any())
    result={'success':passed,'seed':meta['seed'],'skill':meta['skill'],'path':str(path),
            'scope':'recorded expert goals, real VLA executor replay; not model inference',
            'placement_error_m':error,'final_speed_m_s':speed,'final_position':position.tolist(),
            'maximum_lift_m':maximum_lift,'bilateral_lift_samples_20hz':bilateral_lift_samples,
            'max_speed_rad_s':.7,'max_acceleration_rad_s2':1.4,'warnings':data.warning.number.tolist()}
    if hold:
        result['base_control']='live encoder parking, not recorded wheel targets'
        result['maximum_base_drift_m']=max(x['position_error_m'] for x in hold.samples)
        hold.release()
    (path/'vla_executor_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new summary path')
    manifest=json.loads(a.manifest.read_text());results=[]
    for entry in manifest['accepted']:
        path=Path(entry['path']);existing=path/'vla_executor_replay.json'
        result=json.loads(existing.read_text()) if existing.exists() else replay_episode(path)
        results.append(result);print(result['skill'],result['seed'],result['success'],flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'episodes':len(results),'successes':sum(r['success'] for r in results),'results':results},indent=2)+'\n')


if __name__=='__main__':main()

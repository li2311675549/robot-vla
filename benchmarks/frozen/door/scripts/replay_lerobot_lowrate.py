#!/usr/bin/env python3
"""Physically replay 5Hz LeRobot action frames through the normal arm API.

The action samples are linearly interpolated at the 500Hz physics rate; the
ArmController applies its normal slew limit/gravity feedforward and the
GripperController applies normalized closure. No recorded qpos is injected
after the initial episode reset.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.simulation import Simulation


def replay(path):
    path=Path(path).resolve()
    if path.is_file():path=path.parent
    meta=json.loads((path/'episode.json').read_text())
    frames=[json.loads(line) for line in (path/'frames.jsonl').read_text().splitlines()]
    raw=np.load(path/'replay.npz');times=np.asarray(raw['timestamps']);expected_qpos=raw['qpos'];targets=np.asarray([f['action'] for f in frames],dtype=float)
    frame_times=np.asarray([f['timestamp'] for f in frames],dtype=float)
    if targets.ndim!=2 or targets.shape[1]!=7 or len(frame_times)<2:raise ValueError('Expected at least two 7D low-rate action frames')
    if not np.isfinite(targets).all() or np.any(targets[:,6]<0) or np.any(targets[:,6]>1):raise ValueError('Invalid LeRobot action values')
    model=mujoco.MjModel.from_binary_path(str(path/'model.mjb'));data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,raw['initial_state'],int(raw['state_spec']));mujoco.mj_forward(model,data)
    adapter=SimpleNamespace(model=model,data=data);arm=ArmController(adapter);gripper=GripperController(adapter)
    parked=None;parking_sim=None
    if meta.get('base_control')=='encoder_parking_v1':
        from robot_exam.control.parking import WheelParkingHold
        parking_sim=Simulation.__new__(Simulation);parking_sim.model=model;parking_sim.data=data;parking_sim.step_controllers=[]
        parked=WheelParkingHold(parking_sim);parked.begin()
    max_joint_error=0.
    for index,timestamp in enumerate(times):
        target=np.array([np.interp(timestamp,frame_times,targets[:,j]) for j in range(7)])
        arm.command(target[:6]);gripper.command(float(target[6]))
        if parking_sim:parking_sim.step(data.ctrl)
        else:mujoco.mj_step(model,data)
        max_joint_error=max(max_joint_error,float(np.max(np.abs(data.qpos[arm.qadr]-expected_qpos[index,arm.qadr]))))
    result={'status':'PASS' if not data.warning.number.any() else 'FAIL','replay':'5Hz interpolated targets through ArmController/GripperController at 500Hz',
            'frames':len(frames),'physics_steps':len(times),'max_joint_tracking_difference_vs_500Hz_expert_rad':max_joint_error,
            'warnings':data.warning.number.tolist(),'task_success_label':meta['success'],'policy_input':'RGB at 5Hz; state/action channels remain 7D',
            'low_rate_success':'not equivalent to original expert success; inspect final physical placement and task score'}
    skill=path.parent/'skill_result.json'
    if skill.exists():
        expected=json.loads(skill.read_text())
        if all(k in expected for k in ('drop','status')) and 'object' in [model.body(i).name for i in range(model.nbody)]:
            body=model.body('object').id;dof=model.jnt_dofadr[model.joint('object_free').id]
            position=data.xpos[body].copy();speed=float(np.linalg.norm(data.qvel[dof:dof+3]))
            goal=np.asarray(expected['drop']);error=float(np.linalg.norm(position[:2]-goal[:2]))
            result.update({'final_object_position_m':position.tolist(),'final_object_speed_m_s':speed,
                           'placement_error_m':error,'low_rate_physical_placement_pass':bool(error<.04 and abs(position[2]-goal[2])<.01 and speed<.01)})
    (path/'low_rate_replay.json').write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('episodes',nargs='+',type=Path);args=p.parse_args();failed=False
    for path in args.episodes:
        result=replay(path);print(path,result,flush=True);failed|=result['status']!='PASS'
    if failed:raise SystemExit(1)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Replay a recorded physical fruit grasp, then drive to the selected bin dock.

This is a local transport checkpoint test. It does not count as a new autonomous
grasp or a complete fruit task. After restoration, only normal actuators move.
"""
import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import State,TaskRuntime


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',default='0021.npz')
    p.add_argument('--body',default='fruit_4')
    p.add_argument('--profile',choices=['banana','apple','orange'],default='banana')
    p.add_argument('--closure',type=float,default=None)
    p.add_argument('--linear-speed',type=float,default=None)
    p.add_argument('--banana-staging-pure-yaw',action='store_true')
    p.add_argument('--local-dock',choices=['preapproach','lattice'])
    p.add_argument('--dock-angular',type=float)
    p.add_argument('--joint-carry',action='store_true')
    p.add_argument('--twist-slew',action='store_true')
    p.add_argument('--sim-limit',type=float,default=1600.)
    a=p.parse_args();source=a.source.resolve();output=a.output.resolve()
    if output.exists():raise FileExistsError('Use a new checkpoint experiment')
    output.mkdir(parents=True);os.chdir(output)
    meta=json.loads((source/'scene.json').read_text())
    sim=Simulation(source/'scene.xml')
    checkpoint=np.load(source/'checkpoints'/a.checkpoint)
    mujoco.mj_setState(sim.model,sim.data,checkpoint['integration_state'],int(checkpoint['state_spec']))
    rows=json.loads((source/'trajectory.json').read_text())
    row=min(rows,key=lambda x:abs(x['time']-float(checkpoint['time'])))
    sim.data.ctrl[:]=row['ctrl']
    mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=output,wall_limit=1800,sim_limit=a.sim_limit)
    r.config=json.loads((source/'config.json').read_text())
    if a.local_dock:
        r.config.update(local_dock_strategy=a.local_dock,local_dock_payload_only=True,local_dock_timeout_seconds=60)
    if a.dock_angular is not None:r.config['payload_dock_angular_speed_rad_s']=a.dock_angular
    if a.twist_slew:r.config['payload_precision_twist_slew']=True
    if a.banana_staging_pure_yaw:r.config['banana_staging_pure_yaw']=True
    if a.linear_speed is not None:
        r.config['payload_banana_linear_speed_m_s' if a.profile=='banana'
                 else 'payload_linear_speed_m_s']=a.linear_speed
    r.qgoal=sim.data.qpos[r.arm.qadr].copy()
    r.arm.reference=r.qgoal.copy()
    r.wheel_applied=sim.data.ctrl[r.base.ids].copy()
    r.payload_motion_profile=a.profile
    body=a.body;rotation=r.arm.pose()[1]
    grid=OccupancyGrid();r.grid=grid
    for box in meta['obstacles']:grid.rectangle(*box)
    selected=json.loads((source/'base_pose_trials.json').read_text())['selected']['pose']
    stage=np.asarray(selected).copy()
    stage[0]-=.40*math.cos(stage[2]);stage[1]-=.40*math.sin(stage[2])
    result={'source':str(source),'source_checkpoint':a.checkpoint,'body':body,
            'profile':a.profile,'closure':a.closure,'linear_speed_m_s':a.linear_speed,
            'physical_grasp_restored':bool(r.bilateral(body)),
            'success':False,'failure':None,'target_dock':selected}
    try:
        if not result['physical_grasp_restored']:raise RuntimeError('Checkpoint has no bilateral fruit grip')
        if a.closure is not None:
            r.gripper.command(a.closure)
            r.hold(.5)
            if not r.bilateral(body):raise RuntimeError('Closure adjustment lost bilateral contact')
        r.transition(State.NAVIGATE,'checkpoint_fruit_transport',
                     timeout=float(r.config['banana_transport_state_timeout_seconds']) if a.profile=='banana' else 120.)
        r.begin_payload(body,rotation,world_stabilize=not a.joint_carry)
        result['carry_mode']='fixed joint pose' if a.joint_carry else 'world orientation stabilization'
        r.backout(-.85)
        r.navigate(stage,grid,timeout=float(r.config['banana_navigation_timeout_seconds']) if a.profile=='banana' else 120.)
        r.dock(selected,timeout=45,tolerance=.03)
        result['success']=bool(r.stable_grasp(body))
        if not result['success']:raise RuntimeError('Fruit lost after dock')
    except Exception as exc:
        result['failure']={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
    finally:
        result['sim_seconds']=float(sim.data.time)
        result['final_base_pose']=r.base.pose().tolist()
        result['terminal_bilateral']=bool(r.bilateral(body))
        result['warnings']=sim.data.warning.number.tolist()
        r.base.command(0,0)
        if r.payload:r.end_payload()
        r.save()
        (output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()

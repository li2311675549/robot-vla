#!/usr/bin/env python3
"""Distinguish endpoint IK/collision from blocked straight joint approaches.

All candidate base and arm assignments are restricted to separate planning
MjData. The live checkpoint is not moved by the search. This diagnostic is
not a robot execution success claim.
"""
import argparse
import json
import itertools
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.control.arm import ArmController
from robot_exam.control.joint_rrt import plan_joint_rrt
from robot_exam.control.local_dock import PoseClearance


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--body',default='book_1');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--x',nargs='+',type=float,default=[.065,.08,.11,.14,.17]);p.add_argument('--y',nargs='+',type=float,default=[-.8]);p.add_argument('--yaw',nargs='+',type=float,default=[0.])
    p.add_argument('--tool-yaws',nargs='+',type=float,default=[0.],help='degrees')
    p.add_argument('--tool-tilts',nargs='+',type=float,default=[0.],help='degrees forward from vertical')
    p.add_argument('--close-gripper',action='store_true')
    p.add_argument('--top-drag',action='store_true',help='Plan above the book centre instead of reaching behind its far edge')
    p.add_argument('--above-height',type=float,default=.84)
    p.add_argument('--contact-height',type=float,default=.731)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve all planning diagnostics')
    a.output.mkdir(parents=True);meta=json.loads((a.episode/'scene.json').read_text());events=json.loads((a.episode/'fsm.json').read_text())
    checkpoint=a.episode/'checkpoints'/f'{len(events):04d}.npz';saved=np.load(checkpoint)
    sim=Simulation(a.episode/'scene.xml');mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim);r.config=json.loads((a.episode/'config.json').read_text());r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
    if a.close_gripper:
        r.base.command(0,0);r.gripper.command(1);r.hold(.8)
    model=sim.model;original=sim.data.qpos.copy();obj=next(o for o in meta['objects'] if o['body']==a.body);body=model.body(a.body).id
    center=sim.data.xpos[body].copy();rotation=sim.data.xmat[body].reshape(3,3)
    half=float(np.sum(np.abs(rotation[0,:2])*np.array(obj['size'][:2])/2));far=float(center[0]+half+.035)
    if a.top_drag:far=float(center[0])
    goals=[('above_push',np.array([far,center[1],a.above_height])),('push_entry',np.array([far,center[1],a.contact_height]))]
    base_adr=int(model.jnt_qposadr[model.body_jntadr[r.base.body]]);clearance=PoseClearance(r);rows=[]
    seeds=[r.qgoal,r.STOW,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]]
    for x,y,yaw,tool_yaw,tool_tilt in itertools.product(a.x,a.y,a.yaw,a.tool_yaws,a.tool_tilts):
        angle=np.deg2rad(tool_yaw);tilt=-np.deg2rad(tool_tilt)
        rz=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        ry=np.array([[np.cos(tilt),0,np.sin(tilt)],[0,1,0],[-np.sin(tilt),0,np.cos(tilt)]])
        push_rotation=rz@ry@np.diag([1.,-1.,-1.])
        pose=[x,y,yaw];row={'pose':pose,'tool_yaw_degrees':tool_yaw,'tool_tilt_degrees':tool_tilt,'base_collision_free':clearance(pose),'feasible':False,'targets':[]}
        if not row['base_collision_free']:rows.append(row);continue
        data=mujoco.MjData(model);mujoco.mj_copyData(data,model,sim.data)
        data.qpos[base_adr:base_adr+2]=pose[:2];data.qpos[base_adr+3:base_adr+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        mujoco.mj_forward(model,data);arm=ArmController(SimpleNamespace(model=model,data=data));arm.reference=data.qpos[arm.qadr].copy()
        for label,position in goals:
            target={'label':label,'position':position.tolist(),'branches':[],'planned':False}
            for seed in [data.qpos[arm.qadr].copy(),*seeds]:
                try:
                    q,metrics=arm.solve(position,push_rotation,seed=seed,max_iterations=500)
                    branch={'ik':metrics,'q':q.tolist()}
                    try:
                        path,report=plan_joint_rrt(arm,q,wall_seconds=1.5,max_nodes=1000,allowed_bodies=[a.body])
                        branch.update(path=[v.tolist() for v in path],path_report=report)
                        target['planned']=True
                    except RuntimeError as exc:branch['path_error']=str(exc)
                    target['branches'].append(branch)
                    if target['planned']:
                        data.qpos[arm.qadr]=q;mujoco.mj_forward(model,data);break
                except RuntimeError as exc:target['branches'].append({'ik_error':str(exc)})
            row['targets'].append(target)
            if not target['planned']:break
        row['feasible']=len(row['targets'])==2 and all(t['planned'] for t in row['targets']);rows.append(row)
        (a.output/'candidates.json').write_text(json.dumps(rows,indent=2)+'\n');print(pose,row['feasible'],flush=True)
    np.testing.assert_array_equal(sim.data.qpos,original)
    summary={'scope':'Static planning only, not physical push or complete task success','source_checkpoint':str(checkpoint),'candidate_count':len(rows),'live_state_unchanged_by_search':True,'preparation_closed_gripper_with_real_actuator':a.close_gripper,'feasible_candidates':[{k:x[k] for k in ('pose','tool_yaw_degrees','tool_tilt_degrees')} for x in rows if x['feasible']]}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

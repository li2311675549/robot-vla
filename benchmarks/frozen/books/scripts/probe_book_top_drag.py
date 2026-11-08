#!/usr/bin/env python3
"""Physical closed-finger top drag when the far-edge pusher is unreachable.

Independent checkpoint experiment, not a complete book-storage success claim.
No live state assignment after the initial checkpoint restore.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State,TaskFailure
from robot_exam.tasks.scoring import body_bounds
from robot_exam.control.joint_rrt import plan_joint_rrt
from robot_exam.tasks.manipulation import side_rotation


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--body',default='book_1')
    p.add_argument('--edge-push',action='store_true',help='Use raised30degree far-edge push instead of friction-only top drag')
    p.add_argument('--side-grasp',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Keep all push attempts')
    args.output.mkdir(parents=True)
    events=json.loads((args.episode/'fsm.json').read_text())
    saved=np.load(args.episode/'checkpoints'/f'{len(events):04d}.npz')
    sim=Simulation(args.episode/'scene.xml')
    mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']))
    mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=args.output,sim_limit=sim.data.time+180)
    r.config=json.loads((args.episode/'config.json').read_text())
    r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
    body=sim.model.body(args.body).id;initial=sim.data.xpos[body].copy();rotation=np.diag([1.,-1.,-1.])
    if args.edge_push:
        t=-np.pi/6;rotation=np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]])@rotation
    result={'success':False,'scope':'local physical top drag only; not grasp/storage/full-task success',
            'source_checkpoint':str(args.episode/'checkpoints'/f'{len(events):04d}.npz'),
            'initial_book_position':initial.tolist()};trace=[]
    def inspect():
        unsafe=[]
        for c in sim.data.contact:
            a,b=map(int,c.geom);ba,bb=map(int,sim.model.geom_bodyid[[a,b]])
            if not(ba in r._robot or bb in r._robot):continue
            names=[sim.model.geom(g).name or '' for g in (a,b)]
            depth=-float(c.dist)
            limit=.003 if body in (ba,bb) else .002
            if depth>limit:unsafe.append({'geoms':names,'penetration_m':depth})
        if unsafe:raise TaskFailure('physical_collision',json.dumps(unsafe))
    try:
        r.transition(State.NAVIGATE,'local_top_drag_bay',timeout=50)
        r.dock([.065 if args.edge_push else .045,-.8,0],timeout=45)
        r.gripper.command(1);r.hold(.5)
        r.transition(State.MANIPULATE,'collision_planned_top_drag_pregrasp',timeout=120)
        pre=np.array([initial[0],initial[1],.84]);path=None
        if args.edge_push:
            _,upper=body_bounds(sim,args.body);pre[0]=upper[0]+.035
        for seed in [r.qgoal,r.STOW,[0,-1.5,2,-2,-1.57,0]]:
            try:
                q,_=r.arm.solve(pre,rotation,seed=seed,max_iterations=500)
                path,planning=plan_joint_rrt(r.arm,q,allowed_bodies=[args.body]);break
            except RuntimeError:continue
        if path is None:raise TaskFailure('arm_collision_path','No checked top-drag approach')
        for q in path:r.joint_move(q,seconds=2.5,_collision_checked=True)
        if np.linalg.norm(r.arm.pose()[0]-pre)>.018:raise TaskFailure('arm_tracking','Top-drag pregrasp endpoint')
        inspect()
        lower=min(body_bounds(sim,sim.model.body(i).name)[0][2] for i in r._robot
                  if (sim.model.body(i).name or '').startswith('grip_'))
        offset=float(r.arm.pose()[0][2]-lower)
        _,book_upper=body_bounds(sim,args.body)
        first=float(book_upper[2]+offset+.004)
        contact=args.edge_push;press_z=.755 if args.edge_push else None
        if args.edge_push:
            r.move([pre[0],pre[1],press_z],rotation,seconds=2);inspect()
        for dz in ([] if args.edge_push else np.arange(0.,.013,.001)):
            p=pre.copy();p[2]=first-dz;r.move(p,rotation,seconds=.15);inspect()
            normal=sum(c['normal_force_n'] for c in r.contacts(args.body))
            trace.append({'time':float(sim.data.time),'phase':'descend','pinch':r.arm.pose()[0].tolist(),
                          'normal_force_n':normal,'book_position':sim.data.xpos[body].tolist()})
            if normal>=2.:
                contact=True;press_z=p[2];break
        if not contact:raise TaskFailure('push_contact','No top contact within bounded12mm search')
        for step in range(65):
            lo,hi=body_bounds(sim,args.body)
            if lo[0]<=.252:
                result['success']=bool(sim.data.xpos[body,0]>.282 and lo[2]>.72);break
            goal=np.array([pre[0]-.005*(step+1),pre[1],press_z])
            r.move(goal,rotation,seconds=.25);inspect()
            trace.append({'time':float(sim.data.time),'phase':'drag','pinch':r.arm.pose()[0].tolist(),
                          'normal_force_n':sum(c['normal_force_n'] for c in r.contacts(args.body)),
                          'book_position':sim.data.xpos[body].tolist(),'book_lower':lo.tolist()})
        if not result['success']:raise TaskFailure('push_timeout','Top drag did not create safe edge clearance')
        result['push_success']=True
        r.move(r.arm.pose()[0]+[0,0,.08],rotation,seconds=1.5)
        if args.side_grasp:
            result['success']=False
            r.stow();r.transition(State.NAVIGATE,'edge_grasp_base_retreat',timeout=30);r.backout(-.30)
            center=sim.data.xpos[body].copy();lo,hi=body_bounds(sim,args.body)
            grasp=center.copy();grasp[0]=lo[0]+.015;pregrasp=grasp-[.075,0,0];side=side_rotation()
            if grasp[0]>.27:raise TaskFailure('edge_clearance','No lower-finger clearance after actual push')
            r.transition(State.APPROACH,'book_side_grasp_after_raised_push',timeout=90)
            r.gripper.command(0);r.move([pregrasp[0],pregrasp[1],.90],side,seconds=3)
            path=None
            for seed in [r.qgoal,r.STOW,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]]:
                try:
                    q,_=r.arm.solve(pregrasp,side,seed=seed,max_iterations=500)
                    path,planning=plan_joint_rrt(r.arm,q,allowed_bodies=[args.body]);break
                except RuntimeError:continue
            if path is None:raise TaskFailure('arm_collision_path','No checked side-grasp path')
            for q in path:r.joint_move(q,seconds=2,_collision_checked=True)
            r.move(grasp,side,seconds=1.5);inspect()
            r.gripper.command(1);r.hold(.8)
            if not r.bilateral(args.body):raise TaskFailure('grasp_contact','No actual bilateral side grip')
            r.move(grasp+[0,0,.10],side,seconds=2);inspect()
            lift=float(sim.data.xpos[body,2]-center[2]);stable=r.stable_grasp(args.body,seconds=.8)
            result.update(lift_m=lift,final_bilateral=bool(r.bilateral(args.body)))
            result['success']=bool(lift>.055 and stable)
    except Exception as exc:result.update(error=str(exc),category=getattr(exc,'category','simulation'))
    r.save();(args.output/'top_drag_trace.json').write_text(json.dumps(trace,indent=2))
    result.update(strategy='raised_tilt_edge_push' if args.edge_push else 'top_friction_drag',
                  final_book_position=sim.data.xpos[body].tolist(),displacement=(sim.data.xpos[body]-initial).tolist(),
                  sim_seconds=float(sim.data.time)-float(saved['time']),warning_counts=sim.data.warning.number.tolist(),
                  collision_events=r.collision_events)
    if sim.data.warning.number.any() or r.collision_events:result['success']=False
    result['scope']='local real push plus side-grasp/lift' if args.side_grasp else 'local real book push only; not storage/full-task success'
    state=np.empty(mujoco.mj_stateSize(sim.model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(sim.model,sim.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(args.output/'final_state.npz',integration_state=state,state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
                        time=sim.data.time,arm_reference=r.arm.reference,qgoal=r.qgoal,wheel_applied=r.wheel_applied)
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Static placement feasibility after a real above-drawer orientation restore.

Checkpoint restore is initialization only. The restore trajectory uses real
actuators. Candidate base/arm/held-object poses are then written ONLY into a
separate planning MjData, never the live rollout. Feasible endpoints are not
declared physically executed placements.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController
from robot_exam.control.joint_rrt import plan_joint_rrt
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.tasks.manipulation import side_rotation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rank',type=int,default=0)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Keep previous diagnostics')
    args.output.mkdir(parents=True)
    meta=json.loads((args.episode/'scene.json').read_text())
    events=json.loads((args.episode/'fsm.json').read_text())
    index=[i for i,e in enumerate(events) if e['reason']=='side_book_drawer_insertion'][args.rank]
    checkpoint=args.episode/'checkpoints'/f'{index+1:04d}.npz';saved=np.load(checkpoint)
    sim=Simulation(args.episode/'scene.xml')
    mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']))
    mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=args.output/'orientation_restore',sim_limit=sim.data.time+120)
    r.config={**json.loads((args.episode/'config.json').read_text()),'compress_bulk_logs':True}
    r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
    obj=next(o for o in meta['objects'] if o['rank']==args.rank);name=obj['body']
    r.transition(State.PLACE,'restore_book_orientation_for_static_feasibility',timeout=90)
    rotation=side_rotation();errors=[]
    for seed in (r.qgoal,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]):
        try:
            q,_=r.arm.solve([.36,.5,.9],rotation,seed=seed,max_iterations=500)
            path,_=plan_joint_rrt(r.arm,q,allowed_bodies=[name])
            for waypoint in path:r.joint_move(waypoint,seconds=2.)
            break
        except RuntimeError as exc:errors.append(str(exc))
    else:raise RuntimeError('Real orientation restore failed: '+' | '.join(errors))
    if not r.bilateral(name):raise RuntimeError('Restored book not bilaterally held')
    r.save()
    model=sim.model;live=sim.data.qpos.copy();body=model.body(name).id
    position,current_r=r.arm.pose()
    relative_p=current_r.T@(sim.data.xpos[body]-position)
    relative_r=current_r.T@sim.data.xmat[body].reshape(3,3)
    book_adr=int(model.jnt_qposadr[model.body_jntadr[body]])
    base_adr=int(model.jnt_qposadr[model.body_jntadr[model.body('husky').id]])
    planning=SimpleNamespace(model=model,data=mujoco.MjData(model))
    planning.data.qpos[:]=live;mujoco.mj_forward(model,planning.data)
    arm=ArmController(planning)
    links={'shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link','tool0','wrist_rgb_mount',name}
    rows=[]
    for base_x in (-.35,-.20,-.05):
        for center_x in (.36,.40,.45):
            for clearance in (.025,.05,.10,.23):
                desired_center=np.array([center_x,.5,meta['drawer_bottoms'][args.rank]+.008+obj['size'][2]/2+clearance])
                desired_tip=desired_center-current_r@relative_p
                row={'base_x':base_x,'book_center':desired_center.tolist(),'release_clearance_m':clearance,'feasible':False,'branches':[]}
                for seed in (r.qgoal,[0,-1.5,2,-2,1.57,0],[0,-1.5,-2,1,-1.57,0]):
                    planning.data.qpos[:]=live;planning.data.qpos[base_adr]=base_x
                    try:q,metrics=arm.solve(desired_tip,current_r,seed=seed,max_iterations=350)
                    except RuntimeError as exc:
                        row['branches'].append({'error':str(exc)});continue
                    planning.data.qpos[arm.qadr]=q;mujoco.mj_forward(model,planning.data)
                    tip=planning.data.site_xpos[arm.site];rot=planning.data.site_xmat[arm.site].reshape(3,3)
                    planning.data.qpos[book_adr:book_adr+3]=tip+rot@relative_p
                    quat=np.empty(4);mujoco.mju_mat2Quat(quat,(rot@relative_r).ravel())
                    planning.data.qpos[book_adr+3:book_adr+7]=quat;mujoco.mj_forward(model,planning.data)
                    collisions=[]
                    for contact in planning.data.contact:
                        if contact.dist>=-.0003:continue
                        names=[model.body(int(model.geom_bodyid[g])).name or '' for g in contact.geom]
                        if not any(n in links or n.startswith('grip_') for n in names):continue
                        if all(n.startswith('grip_') for n in names):continue
                        if name in names and any(n.startswith('grip_') for n in names):continue
                        collisions.append({'bodies':names,'penetration_m':-float(contact.dist)})
                    row['branches'].append({'q':q.tolist(),'ik':metrics,'collisions':collisions})
                    if not collisions:row['feasible']=True;break
                rows.append(row)
                (args.output/'candidates.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.testing.assert_array_equal(sim.data.qpos,live)
    summary={'scope':'Static scratch endpoints only; no placement success claim','source_checkpoint':str(checkpoint),
             'live_state_unchanged_by_search':True,'candidate_count':len(rows),
             'feasible_endpoints':[dict(base_x=x['base_x'],book_center=x['book_center'],release_clearance_m=x['release_clearance_m']) for x in rows if x['feasible']]}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

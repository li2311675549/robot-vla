#!/usr/bin/env python3
"""Independent post-push kinematic audit, not physical task success."""
import argparse,json,math
from pathlib import Path
import mujoco,numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController
from robot_exam.tasks.manipulation import side_rotation
from robot_exam.control.joint_rrt import plan_joint_rrt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new audit folder')
    a.output.mkdir(parents=True)
    source=ROOT/'outputs/stage8/book_push_edge_seed811_rank2_collision_branch'
    raw=np.load(source/'final_state.npz');rows=[];safe=[]
    for dx in (0.,-.10,.10):
      for dy in (0.,-.12,.12):
       sim=Simulation(source/'scene.xml');mujoco.mj_setState(sim.model,sim.data,raw['integration_state'],int(raw['state_spec']))
       # Independent initialization/kinematic audit, no claim of navigation.
       sim.data.qpos[0]+=dx;sim.data.qpos[1]+=dy;mujoco.mj_forward(sim.model,sim.data)
       arm=ArmController(sim);body=sim.model.body('book_2').id;center=sim.data.xpos[body].copy()
       R=sim.data.xmat[body].reshape(3,3);halfx=float(np.sum(np.abs(R[0,:2])*[.12,.08]))
       grasp=center.copy();grasp[0]-=halfx-.015;pre=grasp-[.075,0,0]
       for roll in (0.,math.pi):
        c,s=math.cos(roll),math.sin(roll);rotation=side_rotation()@np.array([[c,-s,0],[s,c,0],[0,0,1]])
        for initial in (sim.data.qpos[arm.qadr],np.array([0,-1.5,2,-2,1.57,0]),np.array([0,-1.5,-2,1,-1.57,0])):
         row={'dx':dx,'dy':dy,'roll':roll,'pre':pre.tolist(),'grasp':grasp.tolist()}
         try:
          q,metrics=arm.solve(pre,rotation,seed=initial,max_iterations=500)
          arm.scratch.qpos[:]=sim.data.qpos;arm.scratch.qpos[arm.qadr]=q;mujoco.mj_forward(sim.model,arm.scratch)
          collisions=[]
          for contact in arm.scratch.contact:
           if contact.dist>=-.0003:continue
           names=[sim.model.body(int(sim.model.geom_bodyid[g])).name or '' for g in contact.geom]
           if any(n.startswith(('grip_','wrist_','upper_arm','forearm','shoulder','tool0')) for n in names) and not all(n.startswith('grip_') for n in names):
            collisions.append({'bodies':names,'penetration_m':-float(contact.dist)})
          row.update(ik=metrics,q=q.tolist(),collisions=collisions)
          if not collisions:
           safe.append({**row,'rotation':rotation.tolist(),'base':sim.data.qpos[:2].tolist()})
         except RuntimeError as exc:row['error']=str(exc)
         rows.append(row)
    result={'scope':'scratch IK/endpoint audit of initialized base offsets; not physical grasp success','rows':rows,'safe_endpoints':safe}
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('trials',len(rows),'safe',len(safe))
    for row in safe[:4]:print({k:row[k] for k in ('dx','dy','roll','q')})


if __name__=='__main__':main()

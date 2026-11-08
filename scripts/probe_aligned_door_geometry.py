#!/usr/bin/env python3
"""Independent initial-state IK fixtures, never a physical task success claim."""
import argparse
import json
from pathlib import Path

from build_tasks import build
import mujoco
import numpy as np
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController
from robot_exam.control.joint_path import clear_joint_path
from robot_exam.tasks.door import handle_position
from robot_exam.tasks.door_latch import closing_station


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Keep previous probes')
    args.output.mkdir(parents=True)
    path,meta=build('door',1901,args.output/'scene.xml',door_profile='aligned_handles_v3')
    rows=[]
    for angle in (1.56,1.30,1.,.70,.40,.15,0.):
        for margin in (.04,.08):
            s=Simulation(path)
            target=handle_position(meta,angle,True)
            base=closing_station(meta,angle,margin=margin);yaw=base[2]
            # This is an isolated time-zero geometry fixture, before mj_step.
            s.data.qpos[:2]=base[:2];s.data.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
            j=s.model.joint('door_hinge').id
            s.data.qpos[s.model.jnt_qposadr[j]]=angle
            mujoco.mj_forward(s.model,s.data)
            arm=ArmController(s)
            forward=np.r_[target[:2]-base[:2],0.];forward/=np.linalg.norm(forward)
            rotation=np.column_stack([[0,0,1],[forward[1],-forward[0],0],forward])
            row={'angle':angle,'base':base.tolist(),'target':target.tolist(),'margin':margin}
            try:
                q,metrics=arm.solve_multi(target-.09*rotation[:,2],rotation,max_iterations=350,allowed_bodies=('door_leaf',))
                row.update(ik=True,metrics=metrics,q=q.tolist())
                try:
                    contact_q,contact_metrics=arm.solve(target,rotation,seed=q,max_iterations=400)
                    row.update(grasp_ik=True,grasp_q=contact_q.tolist(),grasp_metrics=contact_metrics)
                except RuntimeError as error:row.update(grasp_ik=False,grasp_error=str(error))
                try:
                    waypoints=clear_joint_path(arm,q)
                    row.update(path=True,waypoints=[q.tolist() for q in waypoints])
                except RuntimeError as error:row.update(path=False,path_error=str(error))
            except RuntimeError as error:row.update(ik=False,error=str(error))
            rows.append(row)
            print(angle,margin,row.get('ik'),row.get('path'),row.get('grasp_ik'),flush=True)
            (args.output/'result.json').write_text(json.dumps({'scope':'time-zero IK and collision fixtures only; no physical motion','rows':rows},indent=2)+'\n')


if __name__=='__main__':main()

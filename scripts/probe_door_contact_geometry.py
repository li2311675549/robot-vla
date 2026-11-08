#!/usr/bin/env python3
"""Time-zero kinematic/collision fixtures for nonholonomic door contact arcs."""
import argparse
import itertools
import json
import math
from pathlib import Path

from build_tasks import build
import mujoco
import numpy as np
from robot_exam.control.arm import ArmController
from robot_exam.control.joint_path import clear_joint_path
from robot_exam.tasks.door import handle_position, rotation_z
from robot_exam.simulation import Simulation


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--side-grasp',action='store_true');p.add_argument('--grasp-angle-deg',type=float,default=90.);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    path,meta=build('door',2101,a.output/'scene.xml',door_profile='aligned_handles_v3')
    rows=[]
    candidates=[('open',-.815,1.4)]+[('close',n,t) for n,t in itertools.product((.65,.7,.75,.8),(.9,1.,1.1,1.2))]
    if a.side_grasp:
        candidates=[('open',n,t) for n,t in itertools.product((-.65,-.70,-.75,-.815),(1.2,1.3,1.4))]+[('close',n,t) for n,t in itertools.product((.60,.65,.70),(.9,1.0))]
    for phase,n,t in candidates:
        sim=Simulation(path);m,d=sim.model,sim.data;arm=ArmController(sim)
        hinge_adr=m.jnt_qposadr[m.joint('door_hinge').id]
        theta=0. if phase=='open' else 1.56
        yaw_offset=math.atan2(n,-t)+(math.pi if phase=='open' else 0.)
        def pose(angle):return np.r_[np.asarray(meta['hinge'])[:2]+(rotation_z(angle)@np.array([n,t,0]))[:2],angle+yaw_offset]
        def fixture(angle,q=None):
            base=pose(angle);d.qpos[:2]=base[:2];d.qpos[3:7]=[math.cos(base[2]/2),0,0,math.sin(base[2]/2)];d.qpos[hinge_adr]=angle
            if q is not None:d.qpos[arm.qadr]=q
            mujoco.mj_forward(m,d)
        fixture(theta);target=handle_position(meta,theta,phase=='close');delta=np.r_[target[:2]-pose(theta)[:2],0.];delta/=np.linalg.norm(delta)
        if a.side_grasp:
            angle=math.radians(a.grasp_angle_deg)
            delta=rotation_z(theta)@np.array([math.cos(angle)*(1. if phase=='open' else -1.),math.sin(angle),0.])
        rotation=np.column_stack([[0,0,1],[delta[1],-delta[0],0],delta]);row={'phase':phase,'normal':n,'tangent':t,'yaw_offset':yaw_offset,'side_grasp':a.side_grasp}
        try:
            pre,metrics=arm.solve_multi(target-.09*rotation[:,2],rotation,max_iterations=350,allowed_bodies=('door_leaf',))
            q,grasp_metrics=arm.solve(target,rotation,seed=pre,max_iterations=350)
            waypoints=clear_joint_path(arm,pre);row.update(pre_ik=metrics,grasp_ik=grasp_metrics,q=q.tolist(),approach_waypoints=len(waypoints))
            robot=set()
            for i in range(m.nbody):
                b=i
                while b and b!=m.body('husky').id:b=int(m.body_parentid[b])
                if b==m.body('husky').id:robot.add(i)
            failures=[]
            for angle in np.linspace(0,1.56,17):
                fixture(angle,q)
                for c in d.contact:
                    if c.dist>=-.001:continue
                    g1,g2=map(int,c.geom);b1,b2=map(int,m.geom_bodyid[[g1,g2]]);names=[m.body(b).name or '' for b in (b1,b2)]
                    if b1 not in robot and b2 not in robot:continue
                    if 'floor' in [m.geom(g1).name,m.geom(g2).name]:continue
                    if all(x.startswith('grip_') for x in names):continue
                    if 'door_leaf' in names and any(x.startswith('grip_') for x in names):continue
                    failures.append({'angle':float(angle),'bodies':names,'depth':-float(c.dist)})
            row.update(success=not failures,collisions=failures[:10])
        except RuntimeError as e:row.update(success=False,error=str(e))
        rows.append(row);print(phase,n,t,row['success'],row.get('error',row.get('collisions')),flush=True)
        (a.output/'result.json').write_text(json.dumps({'scope':'read-only time-zero geometry, not physical operation success','rows':rows},indent=2)+'\n')


if __name__=='__main__':main()

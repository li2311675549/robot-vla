"""A geometric clearance check does not by itself claim physical task success."""
import sys
import numpy as np
import mujoco
from robot_exam.simulation import ROOT,Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.door import handle_position
from robot_exam.control.joint_path import clear_joint_path
sys.path.insert(0,str(ROOT/'scripts'))
from build_tasks import build


def test_inside_door_alternate_base_heading_clearance(tmp_path):
    path,meta=build('door',0,tmp_path/'door.xml');s=Simulation(path)
    # Independent kinematic fixture, before any simulation step.
    s.data.qpos[:2]=meta['closing_base'][:2];s.data.qpos[3:7]=[np.cos(-np.pi/4),0,0,np.sin(-np.pi/4)]
    s.data.qpos[-1]=1.56;mujoco.mj_forward(s.model,s.data)
    r=TaskRuntime(s);p=handle_position(meta,1.56,True);f=p-r.arm.pose()[0];f[2]=0;f/=np.linalg.norm(f)
    R=np.column_stack([[0,0,1],[f[1],-f[0],0],f])
    original=s.data.qpos.copy()
    q,_=r.arm.solve(p-R[:,2]*.09,R,seed=[0,-1.5,2,-2,1.57,0])
    path=clear_joint_path(r.arm,q)
    assert path and np.allclose(path[-1],q)
    assert s.data.time==0 and np.array_equal(original,s.data.qpos)

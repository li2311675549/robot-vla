#!/usr/bin/env python3
"""Resume an actual saved contact state for local diagnosis, never full scoring."""
import argparse,json
from pathlib import Path
from _bootstrap import ROOT
import mujoco,numpy as np
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.tasks.door import handle_position


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint_dir',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    fsm=json.loads((a.checkpoint_dir/'fsm.json').read_text());index=next(i for i,t in enumerate(fsm) if t['reason']=='close_from_inside')
    s=Simulation(a.checkpoint_dir/'scene.xml');data=np.load(a.checkpoint_dir/'checkpoints'/f'{index+1:04d}.npz')
    mujoco.mj_setState(s.model,s.data,data['integration_state'],int(data['state_spec']));mujoco.mj_forward(s.model,s.data)
    r=TaskRuntime(s,output=a.output);r.arm.reference=s.data.qpos[r.arm.qadr].copy();r.qgoal=r.arm.reference.copy();r.base_hold_anchor=r.base.pose().copy()
    meta=json.loads((a.checkpoint_dir/'scene.json').read_text());angle=float(s.data.qpos[-1]);r.transition(State.MANIPULATE,'resumed_actual_checkpoint_close')
    result={'scope':'local resumed diagnostic, excluded from complete-task rates','checkpoint':str(a.checkpoint_dir),'initial_angle':angle}
    try:
        for theta in np.linspace(angle,0,100)[1:]:
            pos=handle_position(meta,theta,True);f=pos-np.r_[r.base.pose()[:2],pos[2]];f[2]=0;f/=np.linalg.norm(f)
            R=np.column_stack([[0,0,1],[f[1],-f[0],0],f]);r.move(pos,R,seconds=.2,tolerance=.025)
        result['arc_tracking_completed']=True
    except Exception as exc:result.update(arc_tracking_completed=False,error=str(exc))
    actual=float(s.data.qpos[-1]);result.update(actual_angle=actual,door_closed=abs(actual)<.035,base=r.base.pose().tolist())
    r.save();(a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

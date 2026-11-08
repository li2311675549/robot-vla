#!/usr/bin/env python3
"""Replay one physical fruit release with alternative collision-checked exits."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.manipulation import release
from robot_exam.tasks.runtime import State,TaskRuntime
from robot_exam.tasks.scoring import fruit_score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',default='0083.npz')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((source/'scene.json').read_text())
    events=json.loads((source/'fsm.json').read_text())
    event=next(x for x in reversed(events) if x['reason']=='release')
    target=np.asarray(event['target'],dtype=float)
    snapshot=np.load(source/'checkpoints'/args.checkpoint)
    offsets=[(0,0,.06),(-.03,0,.06),(.03,0,.06),(0,-.03,.06),(0,.03,.06),None]
    results=[]
    for index,offset in enumerate(offsets):
        case=output/f'case_{index:02d}';case.mkdir(exist_ok=True)
        sim=Simulation(source/'scene.xml')
        mujoco.mj_setState(sim.model,sim.data,snapshot['integration_state'],int(snapshot['state_spec']))
        trajectory=json.loads((source/'trajectory.json').read_text())
        row=min(trajectory,key=lambda x:abs(x['time']-float(snapshot['time'])))
        sim.data.ctrl[:]=row['ctrl'];mujoco.mj_forward(sim.model,sim.data)
        r=TaskRuntime(sim,output=case,wall_limit=300,sim_limit=1800)
        r.config=json.loads((source/'config.json').read_text())
        r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
        rotation=np.diag([1.,-1.,-1.])
        record={'offset_m':offset,'integrated_fallback':offset is None,
                'checkpoint_time':float(snapshot['time']),'success':False}
        try:
            if offset is None:
                release(r,target,rotation,retreat_offsets=[(0,0,.06),(-.03,0,.06),(.03,0,.06)])
            else:
                r.transition(State.PLACE,'checkpoint_fruit_release',target=target.tolist())
                r.move(target,rotation,seconds=1.8)
                r.gripper.command(0);r.hold(.6)
                r.move(target+offset,rotation,seconds=1.3)
                r.hold(.6)
            score=fruit_score(sim,metadata)
            record['correct']=score['correct'];record['score_success']=score['success']
            record['success']=bool(score['correct']>=8 and not r.collision_events)
        except Exception as exc:
            record['error']={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
        finally:
            record['tool_pose']=r.arm.pose()[0].tolist()
            record['warnings']=sim.data.warning.number.tolist()
            record['collision_events']=len(r.collision_events)
            record['sim_seconds']=float(sim.data.time)
            r.save();(case/'result.json').write_text(json.dumps(record,indent=2))
            results.append(record)
    (output/'summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()

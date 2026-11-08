#!/usr/bin/env python3
"""Local real push/grasp from an archived pre-push state; no task success claim."""
import argparse,json,os
from pathlib import Path
import mujoco,numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.skills.book_edge import pick_pushed_book


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    source=a.source.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a new probe directory')
    out.mkdir(parents=True);os.chdir(out)
    meta=json.load(open(source/'scene.json'));obj=next(o for o in meta['objects'] if o['rank']==a.rank)
    events=json.load(open(source/'fsm.json'));index=next(i for i,e in enumerate(events,1) if e['reason']=='oracle_push_book_edge' and e['target']==obj['body'])
    raw=np.load(source/'checkpoints'/f'{index:04d}.npz');sim=Simulation(source/'scene.xml')
    mujoco.mj_setState(sim.model,sim.data,raw['integration_state'],int(raw['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=out,wall_limit=300,sim_limit=3000);r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
    r.config=json.load(open(source/'config.json'));r.config['book_feedback_push']=True
    grid=OccupancyGrid()
    for box in meta['obstacles']:grid.rectangle(*box)
    result={'success':False,'scope':'restored pre-push local skill, not a whole task','rank':a.rank,'source':str(source)}
    try:
        grasp=pick_pushed_book(r,obj,grid)
        result.update(success=grasp['lift_m']>.055 and r.bilateral(obj['body']) and not r.collision_events,
                      grasp={k:v for k,v in grasp.items() if k!='rotation'})
    except Exception as exc:result['failure']={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
    finally:
        r.base.command(0,0);r.save();result.update(sim_seconds=float(sim.data.time),warnings=sim.data.warning.number.tolist())
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

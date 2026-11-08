#!/usr/bin/env python3
"""Initialize at a real pre-toppling checkpoint and test live contact settling."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.drawer import _settle_released_book
from robot_exam.tasks.manipulation import drawer_operation
from robot_exam.tasks.scoring import drawer_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,default=2);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve probes')
    a.output.mkdir(parents=True);record=json.loads((a.probe/'result.json').read_text())
    episode=Path(record['source_episode']);meta=json.loads((episode/'scene.json').read_text());events=json.loads((a.probe/'fsm.json').read_text())
    index=next(i for i,e in enumerate(events) if e['reason']=='settle_released_book_flat')
    checkpoint=a.probe/'checkpoints'/f'{index+1:04d}.npz';saved=np.load(checkpoint)
    sim=Simulation(episode/'scene.xml');mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=a.output,sim_limit=sim.data.time+180);r.config=json.loads(a.config.read_text());r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
    target=next(o for o in meta['objects'] if o['rank']==a.rank)
    result={'scope':'Checkpoint local settling/closing, not complete task','source_probe':str(a.probe),'source_episode':str(episode),'checkpoint':str(checkpoint)}
    try:
        _settle_released_book(r,target,meta['drawer_bottoms'][a.rank])
        drawer_operation(r,a.rank,0.,meta)
        score=drawer_score(sim,meta,np.array(meta['start']));item=next(x for x in score['books'] if x['book']==target['body'])
        result.update(success=bool(item['inside_correct_layer'] and abs(item['opening'])<.012 and not r.collision_events and not sim.data.warning.number.any()),item_score=item)
    except Exception as exc:result.update(success=False,error=str(exc),error_type=type(exc).__name__)
    r.save();result.update(sim_seconds=float(sim.data.time)-float(saved['time']),warnings=sim.data.warning.number.tolist(),collision_events=r.collision_events)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

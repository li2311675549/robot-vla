#!/usr/bin/env python3
"""Local full carry/place/close fork from a real held-book transport entry."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.drawer import _place_side_book
from robot_exam.tasks.manipulation import side_rotation
from robot_exam.tasks.scoring import drawer_score
from robot_exam.navigation.grid import OccupancyGrid


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,default=2)
    p.add_argument('--initial-state',type=Path,help='Actual actuator-executed held-state checkpoint from a separately logged pick probe')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep previous probes')
    a.output.mkdir(parents=True);settings=json.loads(a.config.read_text());events=json.loads((a.episode/'fsm.json').read_text())
    if a.initial_state:
        checkpoint=a.initial_state;pick_result=json.loads((checkpoint.parent/'result.json').read_text())
        if not pick_result['success']:raise ValueError('Only a physically successful pick checkpoint may initialize this carry probe')
        rotation=np.asarray(pick_result['grasp_rotation'])
    else:
        index=[i for i,e in enumerate(events) if e['reason']=='side_book_transport'][a.rank]
        checkpoint=a.episode/'checkpoints'/f'{index+1:04d}.npz';rotation=side_rotation()
    saved=np.load(checkpoint)
    sim=Simulation(a.episode/'scene.xml');mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=a.output,sim_limit=sim.data.time+400);r.config=settings
    r.qgoal=saved['qgoal'].copy() if 'qgoal' in saved else sim.data.qpos[r.arm.qadr].copy()
    r.arm.reference=saved['arm_reference'].copy() if 'arm_reference' in saved else r.qgoal.copy()
    if 'wheel_applied' in saved:r.wheel_applied=saved['wheel_applied'].copy()
    metadata=json.loads((a.episode/'scene.json').read_text());target=next(o for o in metadata['objects'] if o['rank']==a.rank)
    grid=OccupancyGrid()
    for obstacle in metadata['obstacles']:grid.rectangle(*obstacle)
    r.grid=grid;operations=[]
    result={'scope':'Local checkpoint carry/place/close; not autonomous complete task from5m','source_episode':str(a.episode),
            'checkpoint':str(checkpoint),'controller_initialization':'Same restored integration state; recorded arm/wheel references when present, otherwise measured arm reference. Ordinary controls only; no execution-time pose writes.'}
    try:
        _place_side_book(r,metadata,a.rank,{'area':target['area']},target,{'rotation':rotation,'lift_m':0.,'scope':'Local real-control carry/place probe'},grid,{},operations)
        score=drawer_score(sim,metadata,np.array(metadata['start']));item=next(x for x in score['books'] if x['book']==target['body'])
        result.update(success=bool(item['inside_correct_layer'] and abs(item['opening'])<.012 and not r.collision_events and not sim.data.warning.number.any()),item_score=item)
    except Exception as exc:result.update(success=False,error=str(exc),error_type=type(exc).__name__)
    result.update(sim_seconds=float(sim.data.time)-float(saved['time']),warnings=sim.data.warning.number.tolist(),collision_events=r.collision_events)
    r.save();(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Paired wheel-driven route tests from the actual last-fruit grasp state."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.navigation.grid import OccupancyGrid


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--strategies',nargs='+',choices=('legacy','strict_pivot','direct_visible','direct_twist_slew'),default=('legacy','strict_pivot'));a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep paired evidence')
    a.output.mkdir(parents=True);events=json.loads((a.episode/'fsm.json').read_text())
    index=max(i for i,e in enumerate(events) if e['reason']=='carry_global_navigation')
    body=next(e['target'] for e in reversed(events[:index]) if e['reason']=='lift_and_hold')
    saved=np.load(a.episode/'checkpoints'/f'{index+1:04d}.npz')
    selected=json.loads((a.episode/'base_pose_trials.json').read_text())['selected'];goal=np.array(selected['pose']);staging=goal.copy();staging[:2]-=.4*np.array([np.cos(goal[2]),np.sin(goal[2])])
    meta=json.loads((a.episode/'scene.json').read_text());category=next(o['category'] for o in meta['objects'] if o['body']==body)
    settings=json.loads((a.episode/'config.json').read_text());results=[]
    for strategy in a.strategies:
        enabled=strategy!='legacy';folder=a.output/strategy;folder.mkdir()
        sim=Simulation(a.episode/'scene.xml');mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
        r=TaskRuntime(sim,output=folder,sim_limit=sim.data.time+200)
        r.config={**settings,'payload_strict_pivot_turn':enabled,'maximum_navigation_replans':0,'compress_bulk_logs':True}
        if strategy in ('direct_visible','direct_twist_slew'):
            r.config.update(payload_direct_visible_categories=['banana','orange'],payload_visible_handoff_categories=['banana','orange'])
        if strategy=='direct_twist_slew':r.config['payload_precision_twist_slew']=True
        r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
        r.payload_motion_profile=category;r.begin_payload(body,r.arm.pose()[1],world_stabilize=True)
        grid=OccupancyGrid()
        for obstacle in meta['obstacles']:grid.rectangle(*obstacle)
        r.grid=grid;result={'strategy':strategy,'strict_pivot':enabled,'success':False}
        try:
            r.transition(State.NAVIGATE,'paired_payload_route_first_attempt',timeout=120)
            nav=r.navigate(staging,grid,timeout=120,max_replans=0)
            result['navigation_success']=True;result['navigation']=nav
            r.transition(State.NAVIGATE,'paired_precision_dock',timeout=50)
            r.dock(goal,timeout=45,tolerance=.03)
            result.update(success=bool(r.bilateral(body) and not r.collision_events and not sim.data.warning.number.any()),final_bilateral=r.bilateral(body))
        except Exception as exc:result.update(error=str(exc),error_type=type(exc).__name__,failure_stage=r.stage)
        result.update(sim_seconds=float(sim.data.time)-float(saved['time']),final_base=r.base.pose().tolist(),warnings=sim.data.warning.number.tolist(),collision_events=r.collision_events)
        r.end_payload();r.save();(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        results.append(result);print(json.dumps(result),flush=True)
    protocol={'scope':'Checkpoint paired route test, not autonomous full10-fruit success','source_episode':str(a.episode),'checkpoint':f'{index+1:04d}.npz',
              'body':body,'staging':staging.tolist(),'final_goal':goal.tolist(),
              'controller_initialization':'Same restored integration state; measured joint reference and measured carry rotation, no live pose assignment. Single navigation attempt per variant for bounded diagnosis.',
              'results':results}
    (a.output/'summary.json').write_text(json.dumps(protocol,indent=2)+'\n')


if __name__=='__main__':main()

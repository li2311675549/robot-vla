#!/usr/bin/env python3
"""Paired physical passive-drawer handle reacquisition at an actual failure."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.manipulation import drawer_operation


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--layer',type=int,default=1);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous paired tests')
    a.output.mkdir(parents=True);events=json.loads((a.episode/'fsm.json').read_text());index=max(i for i,e in enumerate(events) if e['reason']==f'drawer_handle_{a.layer}')
    checkpoint=a.episode/'checkpoints'/f'{index+1:04d}.npz';saved=np.load(checkpoint);meta=json.loads((a.episode/'scene.json').read_text());settings=json.loads((a.episode/'config.json').read_text());results=[]
    for enabled in (False,True):
        folder=a.output/('feedback' if enabled else 'legacy');folder.mkdir();sim=Simulation(a.episode/'scene.xml')
        mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
        r=TaskRuntime(sim,output=folder,sim_limit=sim.data.time+160);r.config={**settings,'drawer_handle_feedback':enabled,'drawer_handle_geometry_trace':True,'compress_bulk_logs':True}
        r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy();result={'feedback_enabled':enabled}
        try:
            operation=drawer_operation(r,a.layer,0.,meta)
            result.update(success=bool(operation['actual']<.012 and not r.collision_events and not sim.data.warning.number.any()),operation=operation)
        except Exception as exc:result.update(success=False,error=str(exc),error_type=type(exc).__name__)
        result.update(sim_seconds=float(sim.data.time)-float(saved['time']),warnings=sim.data.warning.number.tolist(),collision_events=r.collision_events)
        r.save();(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result);print(json.dumps(result),flush=True)
    (a.output/'summary.json').write_text(json.dumps({'scope':'Paired checkpoint handle/closure tests, not autonomous complete three-book task','source_episode':str(a.episode),'checkpoint':str(checkpoint),'results':results},indent=2)+'\n')


if __name__=='__main__':main()

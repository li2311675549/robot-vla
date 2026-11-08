#!/usr/bin/env python3
"""Robot stays5m away: test long-duration fruit support without manipulation."""
import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=180.);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve passive-control evidence')
    a.output.mkdir(parents=True);sim=Simulation(a.scene);meta=json.loads(a.scene.with_suffix('.json').read_text())
    r=TaskRuntime(sim,output=a.output,sim_limit=a.seconds+2);r.config={'compress_bulk_logs':True,'trajectory_log_interval_seconds':1.}
    start=r.base.pose().copy();r.transition(State.OBSERVE,'passive_fruit_stability_robot_five_metres_away',timeout=a.seconds+1)
    r.hold(a.seconds);rows=[{'body':x['body'],'category':x['category'],'initial':x['initial'],
                           'final':sim.data.xpos[sim.model.body(x['body']).id].tolist()} for x in meta['objects']]
    result={'scope':'passive scene validation, no approach/grasp/base motion tasks',
            'seconds':a.seconds,'robot_initial':start.tolist(),'robot_final':r.base.pose().tolist(),
            'robot_xy_drift_m':float(np.linalg.norm(r.base.pose()[:2]-start[:2])),
            'fallen':[x['body'] for x in rows if x['final'][2]<.65],'objects':rows,
            'data_warning':sim.data.warning.number.tolist(),'collision_events':r.collision_events}
    r.save();(a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

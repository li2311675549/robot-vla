#!/usr/bin/env python3
"""Physical recovery diagnostic initialized before the original blind release."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.control.grasp_recovery import stop_and_revalidate_grasp


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous probes')
    a.output.mkdir(parents=True)
    fsm=json.loads((a.episode/'fsm.json').read_text());index=max(i for i,x in enumerate(fsm) if x['reason']=='stop_reclamp_and_replan_after_payload_slip')
    body=fsm[index]['target'];scene=json.loads((a.episode/'scene.json').read_text());obj=next(o for o in scene['objects'] if o['body']==body)
    saved=np.load(a.episode/'checkpoints'/f'{index+1:04d}.npz');sim=Simulation(a.episode/'scene.xml')
    mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=a.output,sim_limit=sim.data.time+20)
    r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
    r.transition(State.RECOVER,'local_stopped_grasp_probe',timeout=10)
    report=stop_and_revalidate_grasp(r,body,float(obj['initial'][2])+.055)
    report.update(scope='Restored actual pre-release checkpoint; local diagnostic only, not full task success',
                  source_episode=str(a.episode),checkpoint=f'{index+1:04d}.npz',
                  controller_initialization='Measured-joint reference initialization, original gripper ctrl retained; same normal physical controller thereafter.')
    r.save();(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Physical release from a saved held-fruit timeout; not a full task trial."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.tasks.manipulation import release
from robot_exam.tasks.scoring import fruit_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior attempts')
    a.output.mkdir(parents=True);events=json.loads((a.episode/'fsm.json').read_text())
    saved=np.load(a.episode/'checkpoints'/f'{len(events):04d}.npz')
    body=next(e['target'] for e in reversed(events) if e['reason']=='lift_and_hold')
    sim=Simulation(a.episode/'scene.xml');m,d=sim.model,sim.data
    mujoco.mj_setState(m,d,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(m,d)
    r=TaskRuntime(sim,output=a.output,sim_limit=d.time+80);r.config={**json.loads((a.episode/'config.json').read_text()),'compress_bulk_logs':True}
    r.qgoal=d.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy();r.base.command(0,0)
    # Recover the original observed release target from its recorded IK
    # witness in isolated data. No true object/bin pose enters this target.
    report=json.loads((a.episode/'base_pose_trials.json').read_text());selected=report['selected']
    scratch=mujoco.MjData(m);mujoco.mj_copyData(scratch,m,d)
    base_adr=int(m.jnt_qposadr[m.body_jntadr[m.body('husky').id]])
    scratch.qpos[base_adr:base_adr+2]=selected['pose'][:2]
    yaw=selected['pose'][2];scratch.qpos[base_adr+3:base_adr+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    goal=next(x['q'] for x in selected['ik'] if x['target']=='release')
    scratch.qpos[r.arm.qadr]=goal;mujoco.mj_forward(m,scratch)
    target=scratch.site_xpos[r.arm.site].copy();rotation=scratch.site_xmat[r.arm.site].reshape(3,3).copy()
    metadata=json.loads((a.episode/'scene.json').read_text());result={'scope':'Local real-control placement after checkpoint initialization; not autonomous5m or full10-fruit task success',
        'source_episode':str(a.episode),'checkpoint':f'{len(events):04d}.npz','body':body,'target_from_recorded_observed_ik':target.tolist()}
    try:
        r.transition(State.APPROACH,'approach_observed_bin_after_transport',target=target.tolist(),timeout=30)
        if not r.bilateral(body):raise RuntimeError('Checkpoint fruit was not physically held')
        r.move(target+[0,0,.045],rotation,seconds=2.4)
        release(r,target,rotation,retreat_offsets=[(0,0,.06),(-.03,0,.06),(.03,0,.06)])
        r.transition(State.VERIFY,'local_physical_placement_check');r.hold(.5)
        score=fruit_score(sim,metadata);item=next(x for x in score['objects'] if x['body']==body)
        result.update(success=bool(item['correct_bin'] and not r.collision_events and not d.warning.number.any()),item_score=item,total_contained_for_context=score['correct'])
    except Exception as exc:result.update(success=False,error=str(exc),error_type=type(exc).__name__)
    r.save();result.update(sim_seconds=float(d.time)-float(saved['time']),warnings=d.warning.number.tolist(),collision_events=r.collision_events)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Compare real release controls at an identical recorded pre-release state."""
import argparse
import json
from pathlib import Path
import shutil
import traceback

from _bootstrap import ROOT  # noqa: F401
import mujoco
import numpy as np
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.tasks.door_contact import release_and_stow
from robot_exam.tasks.door_latch import door_angle


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--release',choices=['legacy','normal_withdraw'],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    transitions=json.loads((a.source/'fsm.json').read_text());index=next(i for i,t in enumerate(transitions) if t['reason']=='contact_close_release_stow')
    checkpoint=a.source/'checkpoints'/f'{index+1:04d}.npz';state=np.load(checkpoint)
    shutil.copyfile(a.source/'scene.xml',a.output/'scene.xml');sim=Simulation(a.output/'scene.xml')
    mujoco.mj_setState(sim.model,sim.data,state['integration_state'],int(state['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=a.output,sim_limit=sim.data.time+180,wall_limit=600)
    result_source=json.loads((a.source/'result.json').read_text());path=Path(result_source['closing_contact']['attempts'][-1]['evidence'])
    contact=json.loads(path.read_text());q=np.array(contact['samples'][-1]['joint_target']);r.qgoal=q.copy();r.arm.reference=q.copy()
    r.config=json.loads((a.source/'config.json').read_text());r.config['door_contact_release']=a.release
    meta=json.loads((a.source/'scene.json').read_text());result={'scope':'LOCAL_RELEASE_ONLY','source_checkpoint':str(checkpoint),
        'release':a.release,'initial_angle':door_angle(r),'restored_arm_reference':q.tolist()}
    try:
        r.transition(State.APPROACH,'paired_release_probe',timeout=120);release_and_stow(r,meta,True);r.hold(2.)
        result.update(success=bool(abs(door_angle(r))<.035 and not r.collision_events),final_angle=door_angle(r))
    except Exception as error:result.update(success=False,error=str(error),traceback=traceback.format_exc())
    finally:r.base.command(0,0);r.save()
    result.update(warnings=sim.data.warning.number.tolist(),collisions=r.collision_events)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()

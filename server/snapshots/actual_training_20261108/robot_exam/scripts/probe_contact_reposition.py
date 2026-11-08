#!/usr/bin/env python3
"""Physical local dock from a recorded pre-dock integration state; not full task."""
import argparse
import json
from pathlib import Path
import shutil
import traceback

from _bootstrap import ROOT
import mujoco
import numpy as np
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.tasks.door_latch import dock_station


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tracking-version',choices=['anchored_v4','axis_bounded_v5'],default='axis_bounded_v5');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    transitions=json.loads((a.source/'fsm.json').read_text());index=next(i for i in range(len(transitions)-1,-1,-1) if transitions[i]['reason']=='contact_safety_reposition')
    checkpoint=a.source/'checkpoints'/f'{index+1:04d}.npz';saved=np.load(checkpoint)
    shutil.copyfile(a.source/'scene.xml',a.output/'scene.xml');sim=Simulation(a.output/'scene.xml')
    mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=a.output,sim_limit=sim.data.time+180,wall_limit=600)
    r.qgoal=r.STOW.copy();r.arm.reference=r.STOW.copy()
    r.config=json.loads((ROOT/'configs/door_continuous_contact_dev_v4.json').read_text())
    r.config.update(local_dock_tracking_version=a.tracking_version,aligned_door_dock_strategy='direct')
    result={'scope':'LOCAL_REPOSITION_ONLY','checkpoint':str(checkpoint),'tracking_version':a.tracking_version,'controller_internal_restore':{'qgoal':'STOW','arm_reference':'STOW','payload':False,'parking':False},'target':transitions[index]['target']}
    start=float(sim.data.time)
    try:
        r.base.command(0,0);r.hold(.3);r.transition(State.NAVIGATE,'direct_contact_recovery_probe',timeout=100)
        dock_station(r,result['target'],timeout=60,tolerance=.025)
        r.hold(2.);pose=r.base.pose();velocity=sim.data.qvel[r.base_velocity_dof:r.base_velocity_dof+6]
        error=np.linalg.norm(pose[:2]-np.array(result['target'])[:2]);yaw=abs((pose[2]-result['target'][2]+np.pi)%(2*np.pi)-np.pi)
        result.update(success=bool(error<.025 and yaw<.035 and np.linalg.norm(velocity[:2])<.02 and abs(velocity[5])<.03 and not r.collision_events),position_error_m=float(error),heading_error_rad=float(yaw),final_pose=pose.tolist())
    except Exception as error:result.update(success=False,error=str(error),traceback=traceback.format_exc())
    finally:r.base.command(0,0);r.save()
    result.update(sim_seconds_elapsed=sim.data.time-start,warnings=sim.data.warning.number.tolist(),collisions=r.collision_events)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()

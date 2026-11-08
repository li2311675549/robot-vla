#!/usr/bin/env python3
"""Resume a recorded physical post-close checkpoint to test carry height IK.

This is a local diagnostic. Restoring a saved state does not count as an
autonomous task run; after restoration the arm and gripper use normal actuators.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import State,TaskRuntime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--body',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    sim=Simulation(source/'scene.xml')
    checkpoint=np.load(source/'checkpoints'/args.checkpoint)
    mujoco.mj_setState(sim.model,sim.data,checkpoint['integration_state'],int(checkpoint['state_spec']))
    rows=json.loads((source/'trajectory.json').read_text())
    row=min(rows,key=lambda x:abs(x['time']-float(checkpoint['time'])))
    sim.data.ctrl[:]=row['ctrl'];mujoco.mj_forward(sim.model,sim.data)
    runtime=TaskRuntime(sim,output=output,wall_limit=900,sim_limit=3000)
    runtime.qgoal=sim.data.qpos[runtime.arm.qadr].copy()
    runtime.arm.reference=runtime.qgoal.copy()
    runtime.config=json.loads((source/'config.json').read_text())
    fsm=json.loads((source/'fsm.json').read_text())
    before=[x for x in fsm if x['time']<=float(checkpoint['time'])+.001]
    approach=next(x for x in reversed(before) if x['state']=='APPROACH')
    position=np.asarray(approach['target'])
    diagnostics=json.loads((source/'ik_diagnostics.json').read_text())
    rotation=np.asarray(diagnostics[-1]['target_rotation'])
    result={'source':str(source),'checkpoint':args.checkpoint,'body':args.body,
            'physical_grasp_restored':bool(runtime.bilateral(args.body)),
            'candidate_results':[],'selected_extra_lift_m':None,'success':False}
    try:
        if not result['physical_grasp_restored']:raise RuntimeError('Checkpoint lacks physical bilateral grasp')
        runtime.transition(State.MANIPULATE,'resume_saved_physical_lift',timeout=60)
        runtime.move(position+[0,0,.095],rotation,seconds=1.8)
        runtime.hold(.6)
        if not runtime.bilateral(args.body):raise RuntimeError('Physical lift lost contact')
        start=runtime.arm.pose()[0].copy()
        for extra in (.04,.02,0.):
            target=start+[0,0,extra]
            try:
                runtime.solve_ik(target,rotation,seed=runtime.qgoal,context='carry_probe')
                result['candidate_results'].append({'extra_lift_m':extra,'ik':True})
                if extra>0:runtime.move(target,rotation,seconds=1.2)
                result['selected_extra_lift_m']=extra
                break
            except RuntimeError as exc:
                result['candidate_results'].append({'extra_lift_m':extra,'ik':False,'reason':str(exc)})
        result['success']=result['selected_extra_lift_m'] is not None and runtime.stable_grasp(args.body,seconds=.2)
    except Exception as exc:
        result['error']={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
    finally:
        result['sim_seconds']=float(sim.data.time)
        result['fruit_height_m']=float(sim.data.xpos[sim.model.body(args.body).id,2])
        result['tip_height_m']=float(runtime.arm.pose()[0][2])
        result['bilateral_contact']=bool(runtime.bilateral(args.body))
        result['warnings']=sim.data.warning.number.tolist()
        runtime.save()
        (output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()

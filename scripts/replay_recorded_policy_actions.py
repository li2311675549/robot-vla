#!/usr/bin/env python3
"""Reproduce saved learned action chunks through the original real executor.

No fresh inference, no pose resets after normal initialization, no render or
network. This is a reproducibility diagnostic, never an extra policy trial.
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.control.parking import WheelParkingHold
from robot_exam.vla.interface import ActionChunkExecutor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--contacts',action='store_true',help='20Hz diagnostic contact-frame wrenches and tool-relative object pose')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous diagnostics')
    original=json.loads((a.case/'result.json').read_text())
    rows=json.loads((a.case/'queries.json').read_text())
    assert rows and original['interpolate_lookahead']
    sim=Simulation(a.fixture/'scene.xml');sim.step(sim.data.ctrl,500)
    hold=WheelParkingHold(sim);hold.begin()
    for _ in range(1100):sim.step(sim.data.ctrl)
    assert hold.ready
    arm=ArmController(sim);grip=GripperController(sim)
    executor=ActionChunkExecutor(sim,arm,grip,base_hold=hold)
    body=sim.model.body('object').id;initial=float(sim.data.xpos[body,2])
    stats=dict(maximum_lift=0.,current_hold=0.,best_hold=0.);trace=[]
    class Observe:
        def before_step(self):pass
        def after_step(self):
            lift=float(sim.data.xpos[body,2]-initial)
            stats['maximum_lift']=max(stats['maximum_lift'],lift)
            witnesses=grip.contacts()
            stats['current_hold']=stats['current_hold']+sim.model.opt.timestep if lift>.05 and grip.bilateral(witnesses) else 0.
            stats['best_hold']=max(stats['best_hold'],stats['current_hold'])
            if a.contacts and (not trace or sim.data.time-trace[-1]['time']>=.05-1e-9):
                from robot_exam.control.grasp_monitor import relative_grasp_pose
                tool,rotation=arm.pose()
                relative,relative_rotation=relative_grasp_pose(tool,rotation,sim.data.xpos[body],sim.data.xmat[body])
                contact_rows=[]
                for index,c in enumerate(sim.data.contact):
                    bodies=[int(sim.model.geom_bodyid[g]) for g in c.geom]
                    if body not in bodies:continue
                    other=bodies[0] if bodies[1]==body else bodies[1]
                    name=sim.model.body(other).name or ''
                    if not name.startswith('grip_'):continue
                    wrench=np.empty(6);mujoco.mj_contactForce(sim.model,sim.data,index,wrench)
                    contact_rows.append(dict(body=name,world_position=c.pos.tolist(),contact_frame=c.frame.tolist(),
                                             wrench_contact_frame=wrench.tolist()))
                trace.append(dict(time=float(sim.data.time),lift_m=lift,bilateral=grip.bilateral(witnesses),
                    object_position=sim.data.xpos[body].tolist(),tool_relative_position=relative.tolist(),
                    tool_relative_rotation=relative_rotation.tolist(),contacts=contact_rows,
                    joint_reference_minus_actual=(arm.reference-sim.data.qpos[arm.qadr]).tolist()))
    sim.add_step_controller(Observe())
    max_error=0.;error=None;completed=0;started=time.monotonic()
    try:
        for row in rows:
            if time.monotonic()-started>120:raise TimeoutError('Bounded action-replay diagnostic')
            drivers=[sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint(n).id]] for n in ['grip_left_driver_joint','grip_right_driver_joint']]
            state=np.r_[sim.data.qpos[arm.qadr],np.mean(drivers)/.8]
            max_error=max(max_error,float(np.max(np.abs(state-np.asarray(row['measured_state_before'])))))
            if max_error>1e-8:raise RuntimeError('Recorded-state divergence exceeds1e-8; do not hide a replay mismatch')
            executor.execute_prefix(np.asarray(row['actions']),float(sim.data.time),row['executed_actions'])
            completed+=1
    except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
    result=dict(status='PASS_REPRODUCIBILITY_ONLY' if error is None else 'FAIL_REPRODUCIBILITY',
        scope='Saved learned action chunks via real actuators; not new inference or an extra success-rate trial',
        source_case=str(a.case),recorded_policy_success=original['success'],training_updates=0,model_queries=0,
        replayed_chunks=completed,expected_chunks=len(rows),maximum_state_error=max_error,
        maximum_lift_m=stats['maximum_lift'],original_maximum_lift_m=original['maximum_lift_m'],
        maximum_lift_difference_m=abs(stats['maximum_lift']-original['maximum_lift_m']),
        continuous_bilateral_hold_seconds=stats['best_hold'],
        original_continuous_bilateral_hold_seconds=original['continuous_bilateral_hold_seconds'],
        final_object_position=sim.data.xpos[body].tolist(),warnings=sim.data.warning.number.tolist(),
        parking_fault=hold.faulted,error=error,wall_seconds=time.monotonic()-started)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.contacts:
        trace_path=a.output.with_name(a.output.stem+'_contacts.json')
        trace_path.write_text(json.dumps(dict(scope='Exact saved-action replay diagnostics; native contact-frame wrenches, not a verified net object wrench',rows=trace),indent=2)+'\n')
        result['contact_trace']=str(trace_path)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));hold.release()


if __name__=='__main__':main()

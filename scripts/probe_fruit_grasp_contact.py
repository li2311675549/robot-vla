#!/usr/bin/env python3
"""Paired physical skill forks from a documented real pregrasp checkpoint.

Initialization restores the saved MuJoCo integration state and sets the same
measured-joint reference/zero wheel target in every fork, followed by0.6s
physical restabilization. These are local probes, never complete-task trials.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.tasks.manipulation import pick


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--body',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--offsets',type=float,nargs='+',default=[0.,-.004,-.008]);p.add_argument('--closures',type=float,nargs='+',default=[.55,.65,.70])
    p.add_argument('--transport',choices=['reverse','static'],default='reverse');p.add_argument('--turn-radians',type=float,default=0.)
    p.add_argument('--force-feedback',action='store_true');p.add_argument('--target-force',type=float,default=12.)
    p.add_argument('--config',type=Path,help='Explicit paired controller override; preserves original scene/physics')
    p.add_argument('--carry-clearance-m',type=float,default=0.)
    p.add_argument('--candidate-indices',type=int,nargs='+',default=[0])
    p.add_argument('--compact-carry',action='store_true',help='Separate physical carry-radius hypothesis, not the earlier passive-fruit explanation')
    a=p.parse_args()
    if not 0<=a.carry_clearance_m<=.04:raise ValueError('Carry clearance must be within the existing0–40mm range')
    if len(a.candidate_indices)>4 or any(i<0 for i in a.candidate_indices):raise ValueError('At most four nonnegative candidate indices')
    if not all(-.015<=v<=0 for v in a.offsets) or not all(.4<=v<=1. for v in a.closures):raise ValueError('Use bounded ordinary grasp commands, including the original full-closure baseline')
    if a.output.exists():raise FileExistsError('Preserve all physical probes')
    a.output.mkdir(parents=True)
    fsm=json.loads((a.episode/'fsm.json').read_text())
    index=max(i for i,x in enumerate(fsm) if x['reason']=='multi_candidate_grasp')
    checkpoint=a.episode/'checkpoints'/f'{index+1:04d}.npz';saved=np.load(checkpoint)
    detection=fsm[index]['target'];settings=json.loads((a.episode/'config.json').read_text())
    if a.config:settings.update(json.loads(a.config.read_text()))
    protocol=dict(scope='Local skill forks, not autonomous5m or full10-fruit task success',source_episode=str(a.episode),
                  checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  observed_detection=detection,target_body_for_contact_verification=a.body,
                  depths_m=a.offsets,hold_closures=a.closures,transport=a.transport,turn_radians=a.turn_radians,
                  force_feedback=a.force_feedback,target_force_n=a.target_force if a.force_feedback else None,
                  config_override=str(a.config) if a.config else None,carry_clearance_m=a.carry_clearance_m,
                  candidate_indices=a.candidate_indices,
                  compact_carry=a.compact_carry,
                  controller_initialization='Set qgoal and arm.reference to restored measured joints, wheel command zero; restabilize0.6s. Same in each probe.',
                  physics_and_scoring_parameters_changed=False,live_pose_writes_during_execution=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');records=[]
    for offset,closure,candidate_index in itertools.product(protocol['depths_m'],protocol['hold_closures'],a.candidate_indices):
        suffix=f'_candidate{candidate_index}' if a.candidate_indices!=[0] else ''
        folder=a.output/(f'depth{round(offset*1000):+03d}_closure{closure:.2f}'+suffix);folder.mkdir()
        sim=Simulation(a.episode/'scene.xml')
        mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
        runtime=TaskRuntime(sim,output=folder,sim_limit=float(sim.data.time)+120,wall_limit=90)
        runtime.config={**settings,detection['category']+'_grasp_z_offset_m':offset,'payload_trace_interval_seconds':.02,
                        'fruit_contact_force_feedback':a.force_feedback,'fruit_target_contact_force_n':a.target_force,
                        'fruit_force_feedback_categories':[detection['category']] if a.force_feedback else [],
                        'compress_bulk_logs':True}
        runtime.qgoal=sim.data.qpos[runtime.arm.qadr].copy();runtime.arm.reference=runtime.qgoal.copy()
        class ContactObserver:
            maximum_table_penetration_m=0.
            def before_step(self):pass
            def after_step(self):
                for contact in sim.data.contact:
                    geoms=[int(g) for g in contact.geom]
                    names=[sim.model.geom(g).name or '' for g in geoms]
                    bodies=[sim.model.body(int(sim.model.geom_bodyid[g])).name or '' for g in geoms]
                    if any(n.endswith('_top') for n in names) and any(b.startswith('grip_') for b in bodies):
                        self.maximum_table_penetration_m=max(self.maximum_table_penetration_m,-float(contact.dist))
        observer=ContactObserver();sim.add_step_controller(observer)
        runtime.base.command(0,0);wall=time.monotonic();result=dict(success=False,offset_m=offset,hold_closure=closure,candidate_index=candidate_index)
        try:
            runtime.hold(.6);grasp=pick(runtime,a.body,detection,candidate_index=candidate_index)
            result['selected_grasp_position']=grasp['position'].tolist()
            result['selected_grasp_rotation']=grasp['rotation'].tolist()
            if not a.force_feedback:runtime.gripper.command(closure)
            if not runtime.stable_grasp(a.body,seconds=.5):raise RuntimeError('Post-lift regulated grip not stable')
            if a.carry_clearance_m:
                runtime.move(runtime.arm.pose()[0]+[0,0,a.carry_clearance_m],grasp['rotation'],seconds=1.2)
            if a.compact_carry:
                from robot_exam.control.compact_payload import prepare_compact_carry
                result['compact_preparation']=prepare_compact_carry(runtime,a.body,grasp['rotation'],.80)
            runtime.transition(State.NAVIGATE,'paired_physical_reverse_transport',timeout=60)
            runtime.payload_motion_profile=detection['category'];runtime.begin_payload(a.body,grasp['rotation'])
            if a.transport=='reverse':runtime.backout(-.85);runtime.hold(2)
            else:runtime.hold(12)
            if a.turn_radians:
                from robot_exam.control.base import wrap_angle
                anchor=runtime.base.pose().copy();goal=wrap_angle(anchor[2]+a.turn_radians);started=sim.data.time
                while abs(wrap_angle(goal-runtime.base.pose()[2]))>.025:
                    if sim.data.time-started>60:raise RuntimeError('Local turn exceeded60s')
                    runtime.base.rotate(anchor[:2],goal);runtime.tick()
                runtime.base.command(0,0);runtime.hold(2)
            result.update(success=bool(runtime.stable_grasp(a.body,seconds=.5)),lift_m=grasp['lift_m'],final_bilateral=runtime.bilateral(a.body))
        except Exception as exc:result.update(error_type=type(exc).__name__,error=str(exc))
        finally:
            result.update(sim_seconds=float(sim.data.time)-float(saved['time']),wall_seconds=time.monotonic()-wall,
                          final_base=runtime.base.pose().tolist(),warning_counts=sim.data.warning.number.tolist(),collision_events=runtime.collision_events)
            result['maximum_finger_table_penetration_m']=observer.maximum_table_penetration_m
            if observer.maximum_table_penetration_m>.003:
                result.update(success=False,safety_failure='Finger/table penetration exceeded3mm')
            if runtime.payload_trace:
                result['max_relative_drift_m']=max(x['relative_error_m'] for x in runtime.payload_trace)
                result['last_contacts']=runtime.payload_trace[-1]['finger_contacts']
            runtime.end_payload();runtime.save()
            (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        records.append(result);(a.output/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
        print(offset,closure,result['success'],result.get('error'),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Same-state physical transport pair, including contacts with non-target fruit."""
import argparse
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.control.compact_payload import prepare_compact_carry
from robot_exam.control.contact_grip import enabled_for
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime, State


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve physical paired trials')
    a.output.mkdir(parents=True)
    source=json.loads((a.parent/'result.json').read_text());meta=json.loads((a.parent/'scene.json').read_text())
    cfg=json.loads((a.parent/'config.json').read_text());events=json.loads((a.parent/'fsm.json').read_text())
    index=next(i+1 for i,e in enumerate(events) if e['reason']=='unrecorded_traditional_payload_transport')
    saved=np.load(a.parent/'checkpoints'/f'{index:04d}.npz');grasp=np.load(a.parent/'grasp/replay.npz')
    goal=np.asarray(json.loads((a.parent/'base_pose_trials.json').read_text())['selected']['pose'])
    staging=goal.copy();staging[:2]-=.4*np.array([np.cos(goal[2]),np.sin(goal[2])])
    rows=[]
    for compact in (False,True):
        out=a.output/('compact' if compact else 'baseline');out.mkdir()
        sim=Simulation(a.parent/'scene.xml')
        mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
        r=TaskRuntime(sim,output=out,sim_limit=sim.data.time+300)
        r.config=cfg.copy();r.qgoal=grasp['command_targets'][-1,:6].copy();r.arm.reference=grasp['arm_references'][-1].copy()
        r.wheel_applied=sim.data.ctrl[r.base.ids].copy();body=source['target']['body'];category=source['category']
        r.payload_motion_profile=category
        if enabled_for(r,category):
            r.grip_force_feedback={'body':body,'target_force_n':float(cfg.get('fruit_target_contact_force_n',12.)),
                                  'force_deadband_n':float(cfg.get('fruit_contact_force_deadband_n',0.))}
        grid=OccupancyGrid()
        for box in meta['obstacles']:grid.rectangle(*box)
        r.grid=grid;rotation=np.asarray(source['grasp'].get('rotation',r.arm.pose()[1]))
        others={sim.model.body(x['body']).id:x['body'] for x in meta['objects'] if x['body']!=body}
        actor=set(r._robot)|{sim.model.body(body).id};contacts=[]
        class ContactObserver:
            last=-float('inf')
            def before_step(self):pass
            def after_step(self):
                if sim.data.time-self.last<.02-1e-9:return
                self.last=float(sim.data.time)
                for i,c in enumerate(sim.data.contact):
                    b1,b2=[int(sim.model.geom_bodyid[g]) for g in c.geom]
                    if not ((b1 in others and b2 in actor) or (b2 in others and b1 in actor)):continue
                    force=np.empty(6);mujoco.mj_contactForce(sim.model,sim.data,i,force)
                    contacts.append({'time':float(sim.data.time),'base':r.base.pose().tolist(),
                                     'bodies':[sim.model.body(b1).name,sim.model.body(b2).name],
                                     'geoms':[sim.model.geom(int(g)).name for g in c.geom],
                                     'position':c.pos.tolist(),'normal_force_n':float(force[0]),
                                     'tangential_force_n':float(np.linalg.norm(force[1:3]))})
        observer=ContactObserver();sim.add_step_controller(observer)
        result={'compact':compact,'success':False,'scope':'transport-to-staging local pair, not placement or complete task',
                'controller_restore':'same integration state, final recorded high-level arm goal/reference, wheel command and configured gripper-feedback state; IK success-history cache reset equally',
                'source_checkpoint':str(a.parent/'checkpoints'/f'{index:04d}.npz')}
        start=float(sim.data.time)
        try:
            if compact:result['compact_preparation']=prepare_compact_carry(r,body,rotation,float(source['target']['initial'][2])+.055)
            r.transition(State.NAVIGATE,'paired_unrecorded_traditional_transport',timeout=240)
            r._release_operation_hold();r.payload_motion_profile=category;r.begin_payload(body,rotation)
            r.backout(-.85);r.navigate(staging,grid,timeout=180)
            result['route_returned']=True
        except RuntimeError as exc:result.update(error=str(exc),failure_category=getattr(exc,'category','planning'))
        sim.remove_step_controller(observer)
        fallen=[name for b,name in others.items() if sim.data.xpos[b,2]<.65]
        result.update(final_bilateral=r.bilateral(body),fallen_non_targets=fallen,non_target_contact_samples=len(contacts),
                      sim_seconds=float(sim.data.time)-start,data_warning=sim.data.warning.number.tolist(),collision_events=r.collision_events)
        result['success']=bool(result.get('route_returned') and result['final_bilateral'] and not fallen and not r.collision_events and not sim.data.warning.number.any())
        with gzip.open(out/'non_target_contacts.json.gz','wt') as f:json.dump(contacts,f)
        r.save();(out/'result.json').write_text(json.dumps(result,indent=2));rows.append(result)
        (a.output/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()

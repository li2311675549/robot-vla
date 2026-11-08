#!/usr/bin/env python3
"""Same real lifted-apple checkpoint: live feedback, frozen grip, deadband."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import State,TaskRuntime


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Keep paired retention trials')
    args.output.mkdir(parents=True);events=json.loads((args.parent/'fsm.json').read_text())
    index=next(i for i,e in enumerate(events) if e['reason']=='unrecorded_traditional_payload_transport')
    saved=np.load(args.parent/'checkpoints'/f'{index+1:04d}.npz')
    source=json.loads((args.parent/'result.json').read_text());settings=json.loads((args.parent/'config.json').read_text());rows=[]
    for variant in ('original_feedback','frozen_verified_closure','feedback_deadband2n'):
        folder=args.output/variant;folder.mkdir();sim=Simulation(args.parent/'scene.xml')
        mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
        r=TaskRuntime(sim,output=folder,sim_limit=sim.data.time+25);r.config={**settings,'payload_trace_interval_seconds':.02}
        r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
        body=source['target']['body'];r.payload_motion_profile='apple'
        if variant!='frozen_verified_closure':
            r.grip_force_feedback={'body':body,'target_force_n':12.,'force_deadband_n':2. if variant=='feedback_deadband2n' else 0.}
        r.begin_payload(body,r.arm.pose()[1],world_stabilize=True);start=float(sim.data.time)
        initial_closure=float(sim.data.ctrl[r.gripper.actuator]/255.)
        result={'variant':variant,'success':False,'initial_closure':initial_closure}
        try:
            r.transition(State.NAVIGATE,'paired_apple_reverse_clearance',timeout=24)
            r.backout(-.85)
            if not r.stable_grasp(body,seconds=1.5):raise RuntimeError('Post-backout stable-grip witness failed')
            result['success']=True
        except Exception as exc:result.update(error=str(exc),category=getattr(exc,'category','simulation'))
        result.update(elapsed_sim_seconds=float(sim.data.time)-start,final_bilateral=bool(r.bilateral(body)),
                      final_closure=float(sim.data.ctrl[r.gripper.actuator]/255.),
                      max_relative_displacement_m=max((x['relative_error_m'] for x in r.payload_trace),default=None),
                      max_relative_rotation_rad=max((x['relative_rotation_error_rad'] for x in r.payload_trace),default=None),
                      warning_counts=sim.data.warning.number.tolist(),collision_events=r.collision_events)
        r.end_payload();r.save();(folder/'result.json').write_text(json.dumps(result,indent=2));rows.append(result)
        print(json.dumps(result),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'scope':'checkpoint retention pair, not autonomous pick or full-task success',
        'source_parent':str(args.parent),'checkpoint':f'{index+1:04d}.npz','results':rows},indent=2))

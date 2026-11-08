#!/usr/bin/env python3
"""Paired local opening test with one declared physical gripper-release fault."""
import argparse,json,hashlib
from pathlib import Path
import mujoco,numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.door import open_segment


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--target',type=float,default=.6);a=p.parse_args()
    source=a.source.resolve()
    if a.output.exists():raise FileExistsError('Use new fault experiment output')
    a.output.mkdir(parents=True);events=json.loads((source/'fsm.json').read_text())
    index=next(i for i,e in enumerate(events,1) if e['reason']=='pull_hinge_arc' and abs(e['target']-a.target)<1e-6)
    path=source/'checkpoints'/f'{index:04d}.npz';snapshot=np.load(path);meta=json.loads((source/'scene.json').read_text())
    results=[]
    for fault in (False,True):
      for retries in (0,2):
        folder=a.output/f'fault_{int(fault)}_retries_{retries}';sim=Simulation(source/'scene.xml')
        mujoco.mj_setState(sim.model,sim.data,snapshot['integration_state'],int(snapshot['state_spec']));mujoco.mj_forward(sim.model,sim.data)
        injected={'begun':False,'complete':False};r=TaskRuntime(sim,output=folder,wall_limit=300,sim_limit=3000)
        r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy();r.wheel_applied=sim.data.ctrl[r.base.ids].copy()
        r.config['door_open_regrasp_attempts']=retries
        def hook(sim,stage):
            if not fault or injected['complete']:return
            if r.stage=='pull_hinge_arc':
                if sim.data.time-r.state_started>.8:injected['begun']=True
                if injected['begun']:r.gripper.command(0)
            elif injected['begun']:injected['complete']=True
        r.callback=hook;start=float(sim.data.time)
        result={'scope':'local arc checkpoint, not full door task; release fault deliberately injected through gripper actuator',
                'source':str(source),'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'fault_enabled':fault,'regrasp_limit':retries,'success':False}
        try:
            actual=open_segment(r,meta,a.target)
            result.update(actual_angle_rad=actual,success=bool(abs(actual-a.target)<=.07 and not r.collision_events))
        except Exception as exc:result['failure']={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
        finally:
            r.base.command(0,0);r.save();result.update(elapsed_sim_seconds=float(sim.data.time-start),retries=r.retries,
                fault_injected=injected['begun'],warnings=sim.data.warning.number.tolist(),collision_events=r.collision_events)
            (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
            (a.output/'result.json').write_text(json.dumps({'trials':len(results),'results':results},indent=2)+'\n')
            print(fault,retries,result['success'],result.get('failure'),flush=True)


if __name__=='__main__':main()

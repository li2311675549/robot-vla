#!/usr/bin/env python3
"""Resume at a real book checkpoint and execute the shared push/grasp skill."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.skills.book_edge import pick_pushed_book
from robot_exam.navigation.grid import OccupancyGrid

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,default=2)
    p.add_argument('--checkpoint-reason',help='Resume a matching real state transition instead of the final failure')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep prior local skill evidence')
    a.output.mkdir(parents=True);meta=json.loads((a.episode/'scene.json').read_text());events=json.loads((a.episode/'fsm.json').read_text())
    obj=next(o for o in meta['objects'] if o['rank']==a.rank)
    checkpoint_index=(next(i+1 for i,e in enumerate(events) if e['reason']==a.checkpoint_reason and e.get('target')==obj['body'])
                      if a.checkpoint_reason else len(events))
    saved=np.load(a.episode/'checkpoints'/f'{checkpoint_index:04d}.npz')
    sim=Simulation(a.episode/'scene.xml');mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=a.output,sim_limit=sim.data.time+300);r.config=json.loads(a.config.read_text())
    r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
    grid=OccupancyGrid()
    for box in meta['obstacles']:grid.rectangle(*box)
    r.grid=grid;obj=next(o for o in meta['objects'] if o['rank']==a.rank)
    result={'success':False,'scope':'checkpoint physical push/grasp/lift, not complete5m storage task','source_episode':str(a.episode),'book':obj['body']}
    try:
        grasp=pick_pushed_book(r,obj,grid)
        result.update(success=bool(r.bilateral(obj['body']) and not r.collision_events),lift_m=grasp['lift_m'],
                      grasp_rotation=grasp['rotation'].tolist(),push_displacement=grasp['push_displacement_m'])
    except Exception as exc:result.update(error=str(exc),category=getattr(exc,'category','simulation'))
    r.save();state=np.empty(mujoco.mj_stateSize(sim.model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(sim.model,sim.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(a.output/'final_state.npz',integration_state=state,state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),time=sim.data.time,
                        arm_reference=r.arm.reference,qgoal=r.qgoal,wheel_applied=r.wheel_applied)
    result.update(sim_seconds=float(sim.data.time)-float(saved['time']),collision_events=r.collision_events,warning_counts=sim.data.warning.number.tolist())
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

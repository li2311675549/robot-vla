#!/usr/bin/env python3
"""Physical placement replay of a real post-transport grip; not a fresh task."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.drawer import _place_side_book
from robot_exam.tasks.manipulation import side_rotation
from robot_exam.tasks.scoring import drawer_score
from robot_exam.navigation.grid import OccupancyGrid


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cabinet-x',type=float,required=True);p.add_argument('--release-clearance',type=float,default=.025)
    p.add_argument('--joint-path',action='store_true');p.add_argument('--rank',type=int,default=0)
    p.add_argument('--max-pre-height',type=float,default=2.);p.add_argument('--align-long-axis',action='store_true')
    a=p.parse_args();source=a.source.resolve()
    if a.output.exists():raise FileExistsError('Use a new placement experiment')
    a.output.mkdir(parents=True)
    events=json.loads((source/'fsm.json').read_text())
    index=[i for i,e in enumerate(events,1) if e['reason']=='side_book_drawer_insertion'][a.rank]
    checkpoint=np.load(source/'checkpoints'/f'{index:04d}.npz');sim=Simulation(source/'scene.xml')
    mujoco.mj_setState(sim.model,sim.data,checkpoint['integration_state'],int(checkpoint['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    r=TaskRuntime(sim,output=a.output,wall_limit=300,sim_limit=3000)
    r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy();r.wheel_applied=sim.data.ctrl[r.base.ids].copy()
    r.config.update(side_book_cabinet_x=a.cabinet_x,side_book_release_clearance_m=a.release_clearance,
        side_book_place_joint_path=a.joint_path,side_book_max_pre_height_m=a.max_pre_height,side_book_align_long_axis=a.align_long_axis)
    meta=json.loads((source/'scene.json').read_text());obj=next(o for o in meta['objects'] if o['rank']==a.rank)
    grid=OccupancyGrid();r.grid=grid
    for box in meta['obstacles']:grid.rectangle(*box)
    result={'scope':'restored physical post-transport grip; not full task or fresh grasp','source':str(source),
            'checkpoint':f'{index:04d}.npz','cabinet_x':a.cabinet_x,'release_clearance_m':a.release_clearance,
            'initial_bilateral':bool(r.bilateral(obj['body'])),'success':False}
    operations=[]
    try:
        if not result['initial_bilateral']:raise RuntimeError('Physical checkpoint grip is absent')
        _place_side_book(r,meta,a.rank,{'area':obj['area']},obj,{'rotation':side_rotation(),'lift_m':None,'scope':'checkpoint side pinch'},grid,.26,operations)
        r.hold(.6);score=drawer_score(sim,meta,np.asarray(meta['start']))
        result.update(score=score,success=bool(score['books'][a.rank]['inside_correct_layer'] and abs(score['books'][a.rank]['opening'])<.012 and not r.collision_events))
    except Exception as exc:result['failure']={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)}
    finally:
        r.base.command(0,0);r.save();result.update(sim_seconds=float(sim.data.time),warnings=sim.data.warning.number.tolist(),collision_events=r.collision_events)
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(sim.model,spec));mujoco.mj_getState(sim.model,sim.data,state,spec)
        np.savez_compressed(a.output/'final_state.npz',integration_state=state,state_spec=int(spec))
        result['final_contacts']=[{'bodies':[sim.model.body(int(sim.model.geom_bodyid[g])).name for g in c.geom],
                                 'penetration_m':-float(c.dist)} for c in sim.data.contact if c.dist<-.0003]
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

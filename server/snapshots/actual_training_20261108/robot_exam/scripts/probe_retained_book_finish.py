#!/usr/bin/env python3
"""Finish a genuinely retained edge grasp after the old centre-distance fault.

Restores a real failure checkpoint only at initialization. This proves the
remaining operation, not autonomous completion of the whole5m task.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.evaluation.recording import VideoRecorder
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.simulation import Simulation
from robot_exam.skills.book_edge import stable_edge_book
from robot_exam.tasks.drawer import _place_side_book
from robot_exam.tasks.manipulation import side_rotation
from robot_exam.tasks.runtime import TaskRuntime, State
from robot_exam.tasks.scoring import drawer_score


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,default=2)
    p.add_argument('--return-home',action='store_true');p.add_argument('--record',action='store_true')
    p.add_argument('--settle-only',action='store_true',help='Resume an actual released-upright-book checkpoint; not a grasp demonstration')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve earlier success/failure evidence')
    a.output.mkdir(parents=True)
    events=json.loads((a.episode/'fsm.json').read_text())
    checkpoint_index=(next(i+1 for i,x in enumerate(events) if x['reason']=='settle_released_book_flat')
                      if a.settle_only else len(events))
    saved_path=a.episode/'checkpoints'/f'{checkpoint_index:04d}.npz'
    saved=np.load(saved_path)
    scene_source=a.episode
    if not (scene_source/'scene.json').exists():
        scene_source=Path(json.loads((a.episode/'result.json').read_text())['source_checkpoint']).parent.parent
    meta=json.loads((scene_source/'scene.json').read_text())
    target=next(x for x in meta['objects'] if x['rank']==a.rank)
    sim=Simulation(scene_source/'scene.xml')
    (a.output/'scene.json').write_bytes((scene_source/'scene.json').read_bytes())
    mujoco.mj_setState(sim.model,sim.data,saved['integration_state'],int(saved['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    camera=mujoco.MjvCamera();camera.distance=3.3;camera.azimuth=140;camera.elevation=-25
    recorder=VideoRecorder(a.output/'video',camera=camera,fps=5) if a.record else None
    def callback(s,stage):
        camera.lookat[:]=s.data.xpos[s.model.body('husky').id]+[.45,0,.55]
        recorder(s,stage)
    r=TaskRuntime(sim,output=a.output,sim_limit=sim.data.time+500,callback=callback if recorder else None)
    r.config=json.loads(a.config.read_text());r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy()
    grid=OccupancyGrid()
    for box in meta['obstacles']:grid.rectangle(*box)
    r.grid=grid;ops=[]
    result={'scope':'Remaining book carry/place/closure/home from actual retained-grasp failure checkpoint; NOT a fresh5m complete trial',
            'source_checkpoint':str(saved_path),'success':False}
    if a.settle_only:result['scope']='Actual released-upright-book checkpoint: physical settle/closure/home; not a new grasp or full task'
    try:
        if a.settle_only:
            from robot_exam.tasks.drawer import _settle_released_book
            from robot_exam.tasks.manipulation import drawer_operation
            _settle_released_book(r,target,meta['drawer_bottoms'][a.rank])
            drawer_operation(r,a.rank,0.,meta);r.backout(-.5);r.stow()
        else:
            if not stable_edge_book(r,target):raise RuntimeError('Real retained grasp not independently revalidated')
            _place_side_book(r,meta,a.rank,{'area':target['area']},target,
                             {'rotation':side_rotation(),'lift_m':0.,'scope':result['scope']},grid,{},ops)
        if a.return_home:
            r.transition(State.RETURN_HOME,'remaining_operation_return_to_original_start',timeout=180)
            r.backout(-.5);r.navigate(np.asarray(meta['start']),grid,timeout=160);r.hold(1.)
        score=drawer_score(sim,meta,np.asarray(meta['start']))
        item=next(x for x in score['books'] if x['rank']==a.rank)
        result.update(success=bool(score['success'] if a.return_home else item['inside_correct_layer'] and abs(item['opening'])<.012),
                      score=score,operations=ops)
    except RuntimeError as exc:result.update(error=str(exc),failure_category=getattr(exc,'category','simulation'))
    r.save();result.update(sim_seconds=float(sim.data.time)-float(saved['time']),data_warning=sim.data.warning.number.tolist(),collision_events=r.collision_events,
                          final_physical_score=drawer_score(sim,meta,np.asarray(meta['start'])))
    if r.collision_events or sim.data.warning.number.any():result['success']=False
    if recorder:result['video']=recorder.close()
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()

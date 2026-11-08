#!/usr/bin/env python3
"""Checkpoint experiment for remaining large-book manipulation and home return."""
import argparse,json,os,time,hashlib
from pathlib import Path
import mujoco,numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.tasks.drawer import _place_side_book
from robot_exam.skills.book_edge import pick_pushed_book
from robot_exam.tasks.scoring import drawer_score
from robot_exam.navigation.grid import OccupancyGrid


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--world-carry',action='store_true');p.add_argument('--record',action='store_true')
    p.add_argument('--config',type=Path,help='Explicit development overrides, stored with the result')
    p.add_argument('--wall-limit',type=float,default=900)
    p.add_argument('--from-grasp',type=Path,help='Use a separately recorded real third-book transport checkpoint')
    a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
    overrides=json.loads(a.config.read_text()) if a.config else {}
    if out.exists():raise FileExistsError('Use a new checkpoint experiment')
    out.mkdir(parents=True);events=json.loads((source/'fsm.json').read_text())
    index=[i for i,e in enumerate(events,1) if e['reason']=='oracle_push_book_edge'][-1]
    checkpoint_path=source/'checkpoints'/f'{index:04d}.npz'
    if a.from_grasp:
        grasp_source=a.from_grasp.resolve();grasp_events=json.loads((grasp_source/'fsm.json').read_text())
        grasp_index=next(i for i,e in enumerate(grasp_events,1) if e['reason']=='side_book_transport')
        checkpoint_path=grasp_source/'checkpoints'/f'{grasp_index:04d}.npz'
    raw=np.load(checkpoint_path);os.chdir(out)
    sim=Simulation(source/'scene.xml');mujoco.mj_setState(sim.model,sim.data,raw['integration_state'],int(raw['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    recorder=None
    if a.record:
        from robot_exam.evaluation.recording import VideoRecorder
        view=mujoco.MjvCamera();view.lookat[:]=[.1,0,.65];view.distance=3.;view.azimuth=145;view.elevation=-25
        recorder=VideoRecorder(out/'video',camera=view,fps=5)
    def record_callback(s,stage):
        if stage.startswith(('RETURN_HOME','SUCCESS')):
            view.lookat[:]=s.data.xpos[s.model.body('husky').id]+[.35,0,.55]
        recorder(s,stage)
    r=TaskRuntime(sim,output=out,callback=record_callback if recorder else None,wall_limit=a.wall_limit,sim_limit=3000);r.qgoal=sim.data.qpos[r.arm.qadr].copy();r.arm.reference=r.qgoal.copy();r.wheel_applied=sim.data.ctrl[r.base.ids].copy()
    r.config=json.loads((ROOT/'configs/task_defaults.json').read_text());r.config.update(json.loads((ROOT/'configs/stage10_book_oracle_dev_v3.json').read_text()))
    r.config.update(overrides)
    if a.world_carry:r.config['side_book_world_carry']=True
    meta=json.loads((source/'scene.json').read_text());obj=next(o for o in meta['objects'] if o['rank']==2)
    grid=OccupancyGrid();r.grid=grid
    for box in meta['obstacles']:grid.rectangle(*box)
    before=drawer_score(sim,meta,np.asarray(meta['start']));result={'success':False,'scope':'checkpoint continuation; not a continuous five-metre full task',
        'source':str(source),'checkpoint':f'{index:04d}.npz','initial_score':before,
        'initial_sim_seconds':float(sim.data.time),'slip_measurement_frame':'grip_pinch',
        'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'physical_checkpoint_path':str(checkpoint_path),'fresh_grasp':not bool(a.from_grasp)};operations=[]
    (out/'config.json').write_text(json.dumps(r.config,indent=2)+'\n');wall=time.monotonic()
    try:
        if not all(x['inside_correct_layer'] and abs(x['opening'])<.012 for x in before['books'][:2]):raise RuntimeError('Prior two books not valid at checkpoint')
        if a.from_grasp:
            from robot_exam.tasks.manipulation import side_rotation
            if not r.bilateral(obj['body']) or not r.stable_grasp(obj['body'],seconds=.6):
                raise RuntimeError('Restored real grip did not pass contact/restabilization checks')
            grasp={'rotation':side_rotation(),'lift_m':None,'scope':'restored real third-book grip'}
        else:grasp=pick_pushed_book(r,obj,grid)
        _place_side_book(r,meta,2,{'area':obj['area']},obj,grasp,grid,.26,operations)
        r.transition(State.RETURN_HOME,'checkpoint_return_home',timeout=200)
        r.navigate(np.asarray(meta['start']),grid,timeout=160);r.hold(1)
        score=drawer_score(sim,meta,np.asarray(meta['start']));result.update(success=score['success'] and not r.collision_events,score=score,operations=operations)
        if result['success']:
            r.transition(State.SUCCESS,'third_book_closed_and_home_verified');r.hold(.4)
    except Exception as exc:
        result.update(success=False,failure={'category':getattr(exc,'category',type(exc).__name__),'message':str(exc)})
    finally:
        r.base.command(0,0);r.save();final_score=drawer_score(sim,meta,np.asarray(meta['start']))
        result['success']=bool(result['success'] and final_score['success'] and not r.collision_events)
        result.update(sim_seconds=float(sim.data.time),wall_seconds=time.monotonic()-wall,
            elapsed_sim_seconds=float(sim.data.time)-result['initial_sim_seconds'],score=final_score,operations=operations,
            failure_stage=None if result['success'] else r.stage,
            warnings=sim.data.warning.number.tolist(),collision_events=r.collision_events)
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(sim.model,spec))
        mujoco.mj_getState(sim.model,sim.data,state,spec)
        np.savez_compressed(out/'final_state.npz',integration_state=state,state_spec=int(spec))
        if recorder:result['video']=recorder.close()
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

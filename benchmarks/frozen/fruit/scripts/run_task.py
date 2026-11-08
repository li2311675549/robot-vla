#!/usr/bin/env python3
"""Bounded full-task attempt with persistent success/failure evidence."""
import argparse
import datetime
import json
from pathlib import Path
import time
import traceback

from _bootstrap import ROOT
from build_tasks import build
import mujoco
import numpy as np
import imageio.v3 as iio
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime,State
from robot_exam.evaluation.recording import VideoRecorder


def run(task,seed,mode,output,record=False,data_output=None,config=None,book_profile='booklet',door_profile='fixed_v1'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings=json.loads((ROOT/'configs/task_defaults.json').read_text())
    if config:settings.update(json.loads(Path(config).read_text()))
    model_path,metadata=build(task,seed,output/'scene.xml',book_profile=book_profile,door_profile=door_profile,
                              wrist_camera=settings.get('task_wrist_camera',False))
    sim=Simulation(model_path)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.10,-.1,.7];camera.distance=3.4;camera.azimuth=145;camera.elevation=-25
    recorder=VideoRecorder(output/'video',camera=camera,fps=5) if record else None
    if recorder:
        # Follow robot during the initial long drive; manipulation uses same
        # camera offset so every simulated phase is preserved at 5fps.
        def record_callback(s,stage):
            p=s.data.xpos[s.model.body('husky').id]
            camera.lookat[:]=p+[.45,0,.55]
            recorder(s,stage)
    else:record_callback=None
    runtime=TaskRuntime(sim,output=output,callback=record_callback,wall_limit=settings['task_wall_seconds_limit'],sim_limit=settings['task_sim_seconds_limit'])
    runtime.config=settings
    (output/'config.json').write_text(json.dumps(settings,indent=2))
    logger=None
    if data_output:
        from robot_exam.data.episodes import EpisodeRecorder
        logger=EpisodeRecorder(sim,Path(data_output),task,seed,mode)
        runtime.data_logger=logger
    wall=time.monotonic();result={}
    try:
        module=__import__('robot_exam.tasks.'+task,fromlist=['execute'])
        result=module.execute(runtime,metadata,mode=mode)
        if not result.get('success'):raise RuntimeError('Strict final task conditions not met')
        if runtime.collision_events:raise RuntimeError('Unhandled static obstacle collision detected')
        runtime.transition(State.SUCCESS,'all_complete_task_conditions_verified')
        result.update({'status':'PASS','failure_stage':None,'failure_category':None})
    except Exception as exc:
        failure_state=runtime.state.value
        result={**result,**getattr(exc,'partial',{}),'success':False,'status':'FAIL','failure_stage':failure_state,
                'failure_category':getattr(exc,'category','simulation_or_scoring'),'error':str(exc),'traceback':traceback.format_exc(),
                'failure_time':sim.data.time,'failure_base':runtime.base.pose().tolist(),
                'failure_qpos':sim.data.qpos.tolist(),'recoverable':getattr(exc,'category','') in ('perception','grasp_contact','ik_unreachable')}
        result['failure_qvel']=sim.data.qvel.tolist()
        result['failure_objects']={sim.model.body(i).name:{'position':sim.data.xpos[i].tolist(),
                                   'quaternion':sim.data.xquat[i].tolist()} for i in runtime.dynamic_bodies}
        result['failure_contacts']=[{'geom_names':[sim.model.geom(int(g)).name for g in c.geom],
                                    'body_names':[sim.model.body(int(sim.model.geom_bodyid[g])).name for g in c.geom],
                                    'distance_m':float(c.dist)} for c in sim.data.contact]
        runtime.transition(State.FAILURE,str(exc))
        try:
            with mujoco.Renderer(sim.model,height=480,width=640) as renderer:
                camera.lookat[:]=sim.data.xpos[sim.model.body('husky').id]+[.4,0,.5]
                renderer.update_scene(sim.data,camera=camera)
                iio.imwrite(output/'failure.png',renderer.render())
        except Exception as render_exc:result['failure_render_error']=str(render_exc)
    finally:
        runtime.base.command(0,0)
        runtime.save()
        if recorder:
            try:result['video']=recorder.close()
            except Exception as exc:result['video_error']=str(exc)
        if logger:result['dataset']=logger.close(bool(result.get('success')))
    result.update({'task':task,'seed':seed,'policy_mode':mode,'sim_seconds':sim.data.time,'wall_seconds':time.monotonic()-wall,
                   'num_retries':runtime.retries,'collision_events':runtime.collision_events,
                   'timestamp':datetime.datetime.now().astimezone().isoformat(),'model':metadata['model'],
                   'initial_base':metadata['start'],'warnings':sim.data.warning.number.tolist()})
    result['observation_scope']=('Oracle hinge/handle expert' if task=='door' else
        'RGB-D targets with simulator contact/containment verification; drawer geometry is Oracle' if mode=='vision' else 'Oracle object geometry expert')
    result['sensor_only_policy']=False
    result['scene_protocol']=metadata.get('scene_protocol','fruit_seeded_v1')
    result['book_profile']=metadata.get('book_profile')
    result['door_profile']=metadata.get('door_profile')
    if task=='door':
        adr=sim.model.jnt_qposadr[sim.model.joint('door_hinge').id]
        angles=[row.get('passive_joints',{}).get('door_hinge',0.) for row in runtime.rows]
        result['phase_metrics']={'opened':bool(max(angles,default=0.)>=1.45),
            'entered':bool(any(row['base'][0]>.7 for row in runtime.rows)),
            'closed_at_end':bool(abs(sim.data.qpos[adr])<.035),
            'maximum_door_angle_rad':float(max(angles,default=0.)),
            'final_door_angle_rad':float(sim.data.qpos[adr]),
            'note':'Opening/entry are sampled physical witnesses; complete success additionally requires original task scoring.'}
        phase=result['phase_metrics']
        phase['closed_after_entry']=phase['opened'] and phase['entered'] and phase['closed_at_end']
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result.get(k) for k in ('task','seed','status','policy_mode','sim_seconds','wall_seconds','failure_stage','failure_category','error','completed')},ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['fruit','door','drawer'],required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--mode',choices=['vision','oracle'],default=None);p.add_argument('--record',action='store_true');p.add_argument('--render',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--data-output',type=Path);p.add_argument('--config',type=Path);p.add_argument('--book-profile',choices=['booklet','a6_a5_a4','distinct_v2'],default='booklet');p.add_argument('--door-profile',choices=['fixed_v1','multistart_v2'],default='fixed_v1');a=p.parse_args()
    mode=a.mode or ('oracle' if a.task=='door' else 'vision')
    if a.task=='door' and mode=='vision':p.error('Door vision policy is not implemented; use explicitly labelled --mode oracle')
    output=a.output or ROOT/'outputs/stage34/runs'/f'{a.task}_{mode}_{a.seed}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    result=run(a.task,a.seed,mode,output,a.record or a.render,a.data_output,a.config,a.book_profile,a.door_profile)
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()

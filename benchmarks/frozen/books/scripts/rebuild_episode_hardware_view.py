#!/usr/bin/env python3
"""Re-render an existing expert episode via identical normal-control physics.

This is a new view of the SAME parent/scene, never another demonstration count.
No recorded qpos is assigned after the initial integration-state restore.
"""
import argparse,json,hashlib,os
from pathlib import Path
import mujoco,numpy as np
from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.data.episodes import EpisodeRecorder,validate_episode
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.scoring import fruit_score


def rebuild(source,output,camera_near_m=None):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Never overwrite a source or a prior reconstruction')
    meta=json.loads((source/'episode.json').read_text());raw=np.load(source/'replay.npz')
    if not meta['success'] or meta.get('schema_version')!=3:raise ValueError('Only qualified schema3 source phases')
    parent=json.loads((source.parent/'result.json').read_text())
    if parent.get('status')!='PHYSICAL_PARENT_PASS_PENDING_REPLAY_AND_EXPORT':
        raise ValueError('A successful phase of a failed parent cannot become a positive full demonstration')
    if meta.get('action_execution_contract','legacy_arm_slew_v1')!='legacy_arm_slew_v1':
        raise ValueError('This reconstruction supports the source normal-arm-slew contract only')
    frames=[json.loads(x) for x in (source/'frames.jsonl').read_text().splitlines()]
    sim=Simulation.__new__(Simulation);sim.model=mujoco.MjModel.from_binary_path(str(source/'model.mjb'));sim.data=mujoco.MjData(sim.model);sim.step_controllers=[]
    mujoco.mj_setState(sim.model,sim.data,raw['initial_state'],int(raw['state_spec']));mujoco.mj_forward(sim.model,sim.data)
    arm,gripper=ArmController(sim),GripperController(sim);arm.reference[:]=raw['initial_arm_reference']
    recorder=EpisodeRecorder(sim,output,meta['language_instruction'],meta['seed'],'oracle',initial_arm_reference=arm.reference,
                             shared_model_cache=ROOT/'outputs/stage20/multifruit_shared_models_v1',camera_near_m=camera_near_m)
    if 'nvidia' not in recorder.meta['opengl']['vendor'].lower():raise RuntimeError('Native NVIDIA renderer required for this hardware view')
    graphics=recorder.meta['opengl'];actual={k:recorder.meta[k] for k in ('opengl','shared_model_snapshot','camera_clipping')}
    recorder.meta.update(meta);recorder.meta.update(actual)
    recorder.meta['reconstruction']={'source_episode':str(source),'source_replay_sha256':hashlib.sha256((source/'replay.npz').read_bytes()).hexdigest(),
        'scope':'same expert action trajectory, new hardware-rendered observation view; not new parent/scene',
        'arm_gripper_api':'ArmController.command/GripperController.command',
        'wheel_api':'original recorded wheel targets from verified source parking controller',
        'post_initialization_qpos_assignment':False}
    # Keep the original verified parking witness, rather than inventing an
    # extra warmup that changes the initial source observation.
    if not meta.get('parking_ready_at_start'):
        import gzip
        history=json.load(gzip.open(source.parent/'parking.json.gz','rt'))
        start=float(raw['timestamps'][0]);dt=float(sim.model.opt.timestep)
        candidates=[x for group in history if np.allclose(group['anchor'],meta['parking_anchor'])
                    for x in group['samples'] if -dt-1e-8<=start-x['time']<=.051 and x.get('ready')]
        witness=min(candidates,key=lambda x:abs(x['time']-start))
        recorder.meta['parking_ready_at_start']=True
        recorder.meta['parking_stable_seconds']=witness['stable_seconds']-max(0.,witness['time']-start)
    step_to_frame={int(x['physics_index']):x for x in frames};stage=frames[0]['stage'];maximum=0.
    for i,target in enumerate(raw['command_targets']):
        maximum=max(maximum,float(np.max(abs(sim.data.qpos-raw['qpos'][i]))))
        if maximum>1e-4:raise RuntimeError('Original physical trajectory not reproduced')
        sim.data.ctrl[:]=raw['controls'][i]
        speed=float(raw['arm_max_speeds'][i]);arm.command(target[:6],max_speed=speed);gripper.command(float(target[6]))
        if i in step_to_frame:stage=step_to_frame[i]['stage']
        recorder.capture_pre_step(stage,arm_target=target[:6],arm_reference=arm.reference,max_speed=speed)
        sim.step(sim.data.ctrl)
    maximum=max(maximum,float(np.max(abs(sim.data.qpos-raw['final_qpos']))))
    recorder.meta['reconstruction']['maximum_qpos_difference']=maximum
    recorder.close(maximum<=1e-4 and not sim.data.warning.number.any())
    # Scene is a small metadata file; preserve the unchanged original alongside
    # each parent so the independent item scorer/replayer sees the same bins.
    scene=output.parent/'scene.json'
    if not scene.exists():scene.write_bytes((source.parent/'scene.json').read_bytes())
    validation=validate_episode(output,replay=True)
    result={'status':'PASS' if validation['status']=='PASS' and maximum<=1e-4 else 'FAIL',
            'source_episode':str(source),'output_episode':str(output),'maximum_qpos_difference':maximum,
            'opengl':graphics,'camera_clipping':recorder.meta['camera_clipping'],
            'new_demonstrations_count':0,'new_scenes_count':0,'full_rate_validation':validation}
    (output/'hardware_reconstruction.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--camera-near-m',type=float);a=p.parse_args()
    if rebuild(a.source,a.output,a.camera_near_m)['status']!='PASS':raise SystemExit(1)

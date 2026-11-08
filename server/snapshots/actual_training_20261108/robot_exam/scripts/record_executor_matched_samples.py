#!/usr/bin/env python3
"""Recapture six train scenes through the actual deployment controller.

These are derived real physical replays, not six new randomized scenes.
Expert high-level goals and post-limit references/ctrl remain separate.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import mujoco
import numpy as np
import _bootstrap
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController,GripperController
from robot_exam.control.parking import WheelParkingHold
from robot_exam.data.episodes import EpisodeRecorder,validate_episode
from robot_exam.vla.interface import ActionChunkExecutor


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve()
    if a.output.exists():raise FileExistsError('Preserve derived captures')
    entries=[e for e in json.loads(a.manifest.read_text())['accepted'] if e['seed'] in (15001,15002) and e['split']=='train']
    if len(entries)!=6:raise RuntimeError('Require the predeclared original six train scenes')
    a.output.mkdir(parents=True);results=[]
    for entry in entries:
        if shutil.disk_usage(a.output).free<26*1024**3:raise RuntimeError('Keep25GiB plus capture headroom')
        folder=Path(entry['path']);destination=a.output/f"seed_{entry['seed']}"/entry['skill'];destination.mkdir(parents=True)
        source=np.load(folder/'replay.npz');source_frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
        targets=np.asarray([r['action'] for r in source_frames]);expected=json.loads((folder.parent/'skill_result.json').read_text())
        model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));data=mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
        sim=Simulation.__new__(Simulation);sim.model=model;sim.data=data;sim.step_controllers=[]
        sim.step(data.ctrl,500);hold=WheelParkingHold(sim);hold.begin()
        for _ in range(1100):sim.step(data.ctrl)
        assert hold.ready and np.max(np.abs(data.qpos-source['qpos'][0]))<1e-9
        arm=ArmController(sim);grip=GripperController(sim);executor=ActionChunkExecutor(sim,arm,grip,base_hold=hold)
        instruction=f"Pick up the {entry['skill']} and place it on the green circular target marker on the table."
        logger=EpisodeRecorder(sim,destination/'episode',instruction,entry['seed'],'oracle',initial_arm_reference=arm.reference)
        logger.meta.update(skill=entry['skill'],sensor_schema='front_wrist_rgb_v1',base_control='encoder_parking_v1',
            action_execution_contract='full_horizon_interpolate_replan5hz_v1',execution_max_speed_rad_s=.7,
            execution_max_acceleration_rad_s2=1.4,controller='ActionChunkExecutor interpolation and acceleration filter, then ArmController gravity compensation once',
            generation='Derived expert goals replayed through deployment; RGB/state recaptured from actual dynamics',
            source_episode=str(folder),source_scene_seed=entry['seed'],split='train',fixture_scale=entry.get('scale'))
        original_step=sim.step;initial=float(data.xpos[model.body('object').id,2]);maximum=best=current=0.
        def observed(ctrl,count=1):
            nonlocal maximum,best,current
            assert count==1
            goal=executor.trace[-1]['goal']
            logger.capture_pre_step('deployment_expert_'+entry['skill'],arm_target=goal[:6],arm_reference=arm.reference,max_speed=.7)
            value=original_step(ctrl,count)
            lift=float(data.xpos[model.body('object').id,2]-initial);maximum=max(maximum,lift)
            current=current+model.opt.timestep if lift>.05 and grip.bilateral() else 0.;best=max(best,current)
            return value
        sim.step=observed;start=float(data.time);wall=time.monotonic();last=-1;error=None
        try:
            for i in range(len(source['controls'])):
                if time.monotonic()-wall>300:raise TimeoutError('Bound matched capture')
                frame=min(int((data.time-start+1e-8)*5),len(targets)-1)
                if frame!=last:executor.submit(targets[frame:frame+10],data.time);last=frame
                executor.step()
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
        body=model.body('object').id;dof=model.jnt_dofadr[model.joint('object_free').id]
        pos=data.xpos[body].copy();goal=np.asarray(expected['drop']);speed=float(np.linalg.norm(data.qvel[dof:dof+3]))
        distance=float(np.linalg.norm(pos[:2]-goal[:2]));passed=bool(error is None and maximum>.055 and best>=1.5 and distance<.04 and abs(pos[2]-goal[2])<.01 and speed<.01 and not grip.contacts() and not data.warning.number.any() and not hold.faulted)
        result={'status':'PASS' if passed else 'FAIL','seed':entry['seed'],'skill':entry['skill'],'drop':goal.tolist(),
            'final_object':pos.tolist(),'maximum_lift_m':maximum,'continuous_bilateral_hold_seconds':best,
            'placement_error_m':distance,'final_object_speed_m_s':speed,'warnings':data.warning.number.tolist(),
            'error':error,'scope':'Derived deployment-controller physical replay, not learned-policy success or new scene'}
        logger.close(passed);hold.release();source.close()
        (destination/'skill_result.json').write_text(json.dumps(result,indent=2)+'\n')
        validation=validate_episode(destination/'episode',replay=True)
        results.append({'path':str(destination/'episode'),'seed':entry['seed'],'skill':entry['skill'],'physical':result,'validation':validation})
        (a.output/'progress.json').write_text(json.dumps({'episodes':results,'expected':6,'completed':len(results),'successes':sum(r['physical']['status']=='PASS' and r['validation']['status']=='PASS' for r in results)},indent=2)+'\n')
        print(entry['seed'],entry['skill'],result['status'],validation['status'],validation['max_replay_qpos_error'],flush=True)


if __name__=='__main__':main()

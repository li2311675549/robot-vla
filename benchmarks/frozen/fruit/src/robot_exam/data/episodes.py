"""Pre-action RGB-D/state sampling plus full-rate actuator replay evidence."""
import json
from types import SimpleNamespace
from pathlib import Path
import shutil

import imageio.v3 as iio
import mujoco
import numpy as np

from robot_exam.control.arm import ArmController
from robot_exam.perception.rgbd import RGBDCamera


class EpisodeRecorder:
    def __init__(self,sim,output,task,seed,mode,frame_rate=5,initial_arm_reference=None):
        self.sim=sim;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.arm=ArmController(sim);self.camera=RGBDCamera(sim)
        camera_names=[sim.model.camera(i).name for i in range(sim.model.ncam)]
        self.wrist=RGBDCamera(sim,camera_name='wrist_rgb') if 'wrist_rgb' in camera_names else None
        self.frame_rate=frame_rate;self.next_frame=sim.data.time
        self.controls=[];self.command_targets=[];self.references=[];self.arm_speeds=[];self.qpos=[];self.times=[];self.frames=[]
        self.initial_arm_reference=self.arm.reference.copy() if initial_arm_reference is None else np.asarray(initial_arm_reference).copy()
        if self.initial_arm_reference.shape!=(6,):raise ValueError('Expected initial six-joint controller reference')
        self.bytes=0;self.start_time=sim.data.time
        self.state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
        self.initial_state=np.empty(mujoco.mj_stateSize(sim.model,self.state_spec))
        mujoco.mj_getState(sim.model,sim.data,self.initial_state,self.state_spec)
        mujoco.mj_saveModel(sim.model,str(self.output/'model.mjb'))
        self.meta={'schema_version':3,'action_semantics':'ur5_absolute_joint_goal_v1','task':task,'seed':seed,'expert_source':mode,'language_instruction':{'fruit':'Sort all apples, bananas and oranges into their labelled bins.','door':'Pull open the door, enter the room and close it.','drawer':'Store books by increasing area into drawers 1, 2 and 3, close drawers and return home.'}.get(task,task),
                   'frame_rate':frame_rate,'physics_timestep':sim.model.opt.timestep,
                   'action_space':'6 absolute UR5 joint position targets (rad, before controller rate limiting and gravity compensation) + normalized gripper tendon closure [0,1]',
                   'controller':'ArmController.command: 0.7rad/s target slew limit plus model gravity feedforward; GripperController.command: normalized closure',
                   'state_space':'6 measured arm joint angles (rad) + normalized mean gripper driver angle',
                   'alignment':'pre-action state/image at timestamp t; command target is applied through the normal controller to step [t,t+dt)',
                   'policy_observations_exclude':['object truth positions','door/drawer truth','base truth (kept only in replay qpos auxiliary)'],
                   'mujoco_version':mujoco.__version__}
        self.meta['sensor_schema']='front_wrist_rgb_v1' if self.wrist else 'front_rgbd_v1'
        self.meta['actual_cameras']=['front','wrist'] if self.wrist else ['front']
        if self.wrist:
            cam_body=int(sim.model.cam_bodyid[self.wrist.id])
            self.meta['wrist_mount_position_tool_m']=sim.model.body_pos[cam_body].tolist()

    def capture_pre_step(self,stage,arm_target=None,arm_reference=None,max_speed=.7):
        s=self.sim
        s.prepare_controls()
        self.controls.append(s.data.ctrl.copy());self.qpos.append(s.data.qpos.copy());self.times.append(s.data.time)
        if arm_target is None:raise ValueError('Explicit pre-slew expert joint goal required for schema 3')
        target=np.asarray(arm_target,dtype=float)
        if target.shape!=(6,) or not np.isfinite(target).all():raise ValueError('Normal replay requires six finite absolute UR5 joint targets')
        reference=target if arm_reference is None else np.asarray(arm_reference,dtype=float)
        if reference.shape!=(6,) or not np.isfinite(reference).all():raise ValueError('Expected six finite post-slew references')
        if not np.isfinite(max_speed) or max_speed<=0:raise ValueError('Invalid controller rate limit')
        self.references.append(reference.copy());self.arm_speeds.append(float(max_speed))
        actuator=s.model.actuator('grip_fingers_actuator').id
        self.command_targets.append(np.r_[target,s.data.ctrl[actuator]/255.])
        if s.data.time+1e-9<self.next_frame:return
        if self.bytes>5*1024**3 or shutil.disk_usage(self.output).free<25*1024**3:
            raise RuntimeError('Dataset disk budget reached')
        index=len(self.frames);stem=f'{index:06d}'
        rgb,depth=self.camera.capture()
        wrist_rgb=self.wrist.capture_rgb() if self.wrist else None
        rgb_path=self.output/'rgb'/f'{stem}.png';depth_path=self.output/'depth'/f'{stem}.npz'
        rgb_path.parent.mkdir(exist_ok=True);depth_path.parent.mkdir(exist_ok=True)
        iio.imwrite(rgb_path,rgb);np.savez_compressed(depth_path,depth_m=depth)
        if self.wrist:
            wrist_path=self.output/'rgb_wrist'/f'{stem}.png';wrist_path.parent.mkdir(exist_ok=True)
            iio.imwrite(wrist_path,wrist_rgb);self.bytes+=wrist_path.stat().st_size
        driver=[]
        for name in ('grip_left_driver_joint','grip_right_driver_joint'):
            driver.append(s.data.qpos[s.model.jnt_qposadr[s.model.joint(name).id]])
        state=[*s.data.qpos[self.arm.qadr].tolist(),float(np.mean(driver)/.8)]
        action=self.command_targets[-1].tolist()
        self.frames.append({'frame_index':index,'timestamp':s.data.time,'physics_index':len(self.controls)-1,
                            'observation_timestamp':float(s.data.time),'action_timestamp':float(s.data.time),
                            'observation.state':state,'action':action,
                            'action.reference':reference.tolist(),'action.actuator_ctrl':s.data.ctrl[self.arm.actuators].tolist(),
                            'controller.max_speed_rad_s':float(max_speed),'rgb':str(rgb_path.relative_to(self.output)),
                            'depth':str(depth_path.relative_to(self.output)),'task':self.meta['language_instruction'],'stage':str(stage)})
        self.bytes+=rgb_path.stat().st_size+depth_path.stat().st_size;self.next_frame+=1/self.frame_rate
        if self.wrist:
            self.frames[-1].update(rgb_wrist=str(wrist_path.relative_to(self.output)),
                camera_timestamps={'front':float(s.data.time),'wrist':float(s.data.time)},
                camera_extrinsics={'front':self.camera.world_from_optical().tolist(),
                                   'wrist':self.wrist.world_from_optical().tolist()})

    def close(self,success):
        self.camera.close()
        if self.wrist:self.wrist.close()
        np.savez_compressed(self.output/'replay.npz',controls=np.asarray(self.controls),command_targets=np.asarray(self.command_targets),arm_references=np.asarray(self.references),arm_max_speeds=np.asarray(self.arm_speeds),initial_arm_reference=self.initial_arm_reference,qpos=np.asarray(self.qpos),timestamps=np.asarray(self.times),
                            initial_state=self.initial_state,state_spec=int(self.state_spec),final_qpos=self.sim.data.qpos.copy())
        self.meta.update({'success':bool(success),'frames':len(self.frames),'physics_steps':len(self.controls),'sim_seconds':self.sim.data.time-self.start_time,
                          'camera_K':self.camera.K.tolist(),'source':'actual controlled physical simulation','full_rate_controls':'replay.npz'})
        if self.wrist:self.meta['wrist_camera_K']=self.wrist.K.tolist()
        (self.output/'episode.json').write_text(json.dumps(self.meta,ensure_ascii=False,indent=2))
        with (self.output/'frames.jsonl').open('w') as f:
            for row in self.frames:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        return {'path':str(self.output),'frames':len(self.frames),'success':bool(success),'image_depth_bytes':self.bytes}


def validate_episode(path,replay=False):
    path=Path(path);meta=json.loads((path/'episode.json').read_text())
    frames=[json.loads(line) for line in (path/'frames.jsonl').read_text().splitlines()]
    data=np.load(path/'replay.npz');model=mujoco.MjModel.from_binary_path(str(path/'model.mjb'))
    errors=[]
    times=data['timestamps'];ctrl=data['controls'];qpos=data['qpos']
    if len(ctrl)!=len(times) or len(ctrl)!=len(qpos) or not len(ctrl):errors.append('physics alignment/empty episode')
    if len(times)>1 and not np.allclose(np.diff(times),model.opt.timestep,atol=1e-8):errors.append('physics timestamps')
    if not np.isfinite(ctrl).all() or not np.isfinite(qpos).all():errors.append('nonfinite values')
    arm_actuators=[model.actuator(n+'_position').id for n in ('shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint')]
    maximum_action_jump=float(np.max(np.abs(np.diff(ctrl[:,arm_actuators],axis=0)))) if len(ctrl)>1 else 0.
    targets=data['command_targets'] if 'command_targets' in data else None
    if meta.get('schema_version',1)>=2 and (targets is None or targets.shape!=(len(ctrl),7)):
        errors.append('normal controller action targets missing or misaligned')
    maximum_target_jump=float(np.max(np.abs(np.diff(targets[:,:6],axis=0)))) if targets is not None and len(targets)>1 else 0.
    if meta.get('schema_version',1)<3 and maximum_target_jump>.05:errors.append('abnormal normal-controller target jump')
    if meta.get('schema_version',1)>=3:
        if meta.get('action_semantics')!='ur5_absolute_joint_goal_v1':errors.append('unknown action semantics')
        refs=data['arm_references'];speeds=data['arm_max_speeds']
        if refs.shape!=(len(ctrl),6) or speeds.shape!=(len(ctrl),):errors.append('reference/rate alignment')
        if len(refs)>1 and np.any(np.abs(np.diff(refs,axis=0))>speeds[1:,None]*model.opt.timestep+1e-8):errors.append('reference slew limit')
        joints=[model.joint(n).id for n in ('shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint')]
        limits=model.jnt_range[joints]
        if not np.isfinite(targets).all() or np.any(targets[:,:6]<limits[:,0]) or np.any(targets[:,:6]>limits[:,1]):errors.append('canonical goal range/nonfinite')
    if maximum_action_jump>.05:errors.append('abnormal per-physics-step arm action jump')
    for row in frames:
        k=row['physics_index'];action=np.asarray(row['action']);state=np.asarray(row['observation.state'])
        if action.shape!=(7,) or state.shape!=(7,):
            errors.append('action/state dimensions');continue
        if not np.isfinite(action).all() or not np.isfinite(state).all():errors.append('nonfinite frame state/action')
        if abs(row['timestamp']-times[k])>1e-9:errors.append('frame timestamp mismatch')
        if not 0<=action[-1]<=1:errors.append('gripper action outside [0,1]')
        if meta.get('schema_version',1)>=3:
            if row.get('observation_timestamp')!=row['timestamp'] or row.get('action_timestamp')!=row['timestamp']:
                errors.append('pre-step observation/action synchronization')
            if not np.allclose(row['action.reference'],data['arm_references'][k],atol=1e-9):errors.append('reference channel mismatch')
            if not np.allclose(row['action.actuator_ctrl'],ctrl[k,arm_actuators],atol=1e-9):errors.append('actuator channel mismatch')
        rgb=iio.imread(path/row['rgb']);depth=np.load(path/row['depth'])['depth_m']
        if rgb.shape!=(480,640,3) or not np.isfinite(depth).all() or depth.min()<=0:errors.append('RGB-D invalid')
        if meta.get('sensor_schema')=='front_wrist_rgb_v1':
            wrist_path=path/row.get('rgb_wrist','missing_wrist.png')
            if not wrist_path.is_file():errors.append('missing real wrist image')
            elif iio.imread(wrist_path).shape!=(480,640,3):errors.append('invalid wrist image')
            if row.get('camera_timestamps')!={'front':row['timestamp'],'wrist':row['timestamp']}:
                errors.append('dual-camera timestamp mismatch')
        if meta.get('schema_version',1)>=2:
            if not np.allclose(action,targets[k],atol=1e-8):errors.append('normal controller action mismatch')
        elif not np.allclose(action[:6],ctrl[k,[model.actuator(n+'_position').id for n in ('shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint')]]):errors.append('actual action mismatch')
    limited=model.actuator_ctrllimited.astype(bool)
    if np.any(ctrl[:,limited]<model.actuator_ctrlrange[limited,0]-1e-6) or np.any(ctrl[:,limited]>model.actuator_ctrlrange[limited,1]+1e-6):errors.append('actuator limits')
    maximum_error=None
    if replay:
        live=mujoco.MjData(model);mujoco.mj_setState(model,live,data['initial_state'],int(data['state_spec']))
        mujoco.mj_forward(model,live)
        maximum_error=0.
        if meta.get('schema_version',1)>=2:
            adapter=SimpleNamespace(model=model,data=live)
            arm=ArmController(adapter)
            from robot_exam.control.arm import GripperController
            gripper=GripperController(adapter)
            if 'initial_arm_reference' in data:arm.reference[:]=data['initial_arm_reference']
            for i,target in enumerate(targets):
                maximum_error=max(maximum_error,float(np.max(np.abs(live.qpos-qpos[i]))))
                # Non-policy wheel commands are exogenous replay inputs;
                # arm/gripper still pass through their normal controller.
                live.ctrl[:]=ctrl[i]
                speed=float(data['arm_max_speeds'][i]) if 'arm_max_speeds' in data else .7
                arm.command(target[:6],max_speed=speed);gripper.command(float(target[6]));mujoco.mj_step(model,live)
            replay_kind='normal ArmController/GripperController API'
        else:
            for i,command in enumerate(ctrl):
                maximum_error=max(maximum_error,float(np.max(np.abs(live.qpos-qpos[i]))))
                live.ctrl[:]=command;mujoco.mj_step(model,live)
            replay_kind='raw actuator controls'
        maximum_error=max(maximum_error,float(np.max(np.abs(live.qpos-data['final_qpos']))))
        replay_tolerance=1e-4 if meta.get('schema_version',1)>=2 else 1e-5
        if maximum_error>replay_tolerance:errors.append(f'physics replay mismatch (tol={replay_tolerance:g})')
        if live.warning.number.any():errors.append('replay simulation warning')
    result={'status':'PASS' if not errors else 'FAIL','frames':len(frames),'physics_steps':len(ctrl),'success_label':meta['success'],
            'oracle':meta['expert_source']=='oracle','errors':sorted(set(errors)),'max_replay_qpos_error':maximum_error,
            'replay_api':replay_kind if replay else None,'max_per_step_arm_action_jump_rad':maximum_action_jump,
            'max_per_step_command_target_jump_rad':maximum_target_jump}
    (path/'validation.json').write_text(json.dumps(result,indent=2));return result

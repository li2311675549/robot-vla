#!/usr/bin/env python3
"""One parked-mobile dual-view rollout: identical physical score for expert/mock/VLA."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import mujoco
import _bootstrap  # noqa: F401 - add project src before importing robot_exam
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController, GripperController
from robot_exam.control.parking import WheelParkingHold
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.skills.pick_place import PickPlace
from robot_exam.vla.interface import ActionChunkExecutor
from robot_exam.vla.client import query_and_execute, connect_openpi_policy


def require_checkpoint(metadata, expected):
    if expected is not None and str(metadata.get('checkpoint_step_label')) != expected:
        raise ValueError(f'Checkpoint mismatch: expected {expected}, got {metadata.get("checkpoint_step_label")}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--policy', choices=('expert', 'mock_hold', 'openpi'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--expected-checkpoint-label', help='Reject a different server checkpoint before any learned action')
    parser.add_argument('--policy-noise-seed',type=int,help='Transport-only evaluation RNG; scene remains unchanged')
    parser.add_argument('--client-source', type=Path, default=Path('/home/li/vla/openpi/packages/openpi-client/src'))
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--execute-actions', type=int, default=1)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--video-backend',choices=('ffmpeg','pyav'),default='ffmpeg')
    parser.add_argument('--network-image-size', type=int, choices=(224,))
    parser.add_argument('--websocket-compression',choices=('deflate',))
    parser.add_argument('--max-response-seconds',type=float,default=.4,
                        help='Wall-time budget while simulation pauses; strict default remains0.4s')
    parser.add_argument('--interpolate-lookahead',action='store_true',help='Retain future prediction for interpolation while executing only the selected prefix')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve every rollout')
    if not 0 < args.seconds <= 120 or not 1 <= args.execute_actions <= 10:
        parser.error('Use a 0..120 second budget and 1..10 prefix actions')
    if not 0 < args.max_response_seconds <= 2.:
        parser.error('Response budget must be positive and at most2s')
    fixture_meta=args.fixture/'fixture.json'
    meta = json.loads((fixture_meta if fixture_meta.exists() else args.fixture/'episode/episode.json').read_text())
    if meta['sensor_schema'] != 'front_wrist_rgb_v1' or meta['base_control'] != 'encoder_parking_v1':
        raise ValueError('Only real dual-camera mobile fixtures supported')
    reference = meta if fixture_meta.exists() else json.loads((args.fixture/'skill_result.json').read_text())
    goal = np.asarray(reference['drop'])
    args.output.mkdir(parents=True)
    sim = Simulation(args.fixture/'scene.xml')
    recorder=None;phase='initialize'
    if args.record:
        from robot_exam.evaluation.recording import VideoRecorder
        if args.video_backend=='pyav':
            from robot_exam.evaluation.recording_pyav import PyAVVideoRecorder as VideoRecorder
        camera=mujoco.MjvCamera();camera.lookat[:]=[.45,0,.65]
        camera.distance=2.7;camera.azimuth=135;camera.elevation=-30
        recorder=VideoRecorder(args.output/'video',camera=camera,fps=5)
        class RecordObserver:
            def before_step(self):pass
            def after_step(self):recorder(sim,phase)
        sim.add_step_controller(RecordObserver())
    sim.step(sim.data.ctrl, 500)
    hold = WheelParkingHold(sim)
    hold.begin()
    for _ in range(1100):
        sim.step(sim.data.ctrl)
    if not hold.ready:
        raise RuntimeError('Parking admission did not stabilize')
    arm, gripper = ArmController(sim), GripperController(sim)
    initial = sim.data.xpos[sim.model.body('object').id].copy()
    dof = sim.model.jnt_dofadr[sim.model.joint('object_free').id]
    start, wall = float(sim.data.time), time.monotonic()
    step = sim.step
    maximum_lift = best_hold = lifted_hold = 0.
    samples, queries = [], []
    settled = 0.
    unexpected_contacts=[]
    robot=set()
    base_body=sim.model.body('husky').id
    for body in range(sim.model.nbody):
        parent=body
        while parent:
            if parent==base_body:robot.add(body);break
            parent=int(sim.model.body_parentid[parent])

    def measured_step(ctrl, count=1):
        nonlocal maximum_lift, best_hold, lifted_hold, settled
        if count != 1:
            raise RuntimeError('Single-step actuator control required')
        if sim.data.time-start > args.seconds or time.monotonic()-wall > 600:
            raise TimeoutError('Bounded rollout limit reached')
        value = step(ctrl, count)
        for c in sim.data.contact:
            if c.dist>=-.003:continue
            g1,g2=map(int,c.geom);b1,b2=map(int,sim.model.geom_bodyid[[g1,g2]])
            if b1 not in robot and b2 not in robot:continue
            names=[sim.model.body(b).name or '' for b in (b1,b2)]
            if 'floor' in [sim.model.geom(g).name for g in (g1,g2)]:continue
            if all(n.startswith('grip_') for n in names):continue
            if 'object' in names and any(n.startswith('grip_') for n in names):continue
            if b1==b2:continue
            event={'time':float(sim.data.time),'bodies':names,'penetration_m':-float(c.dist)}
            unexpected_contacts.append(event)
            raise RuntimeError('Unexpected robot penetration: '+str(event))
        position = sim.data.xpos[sim.model.body('object').id]
        lift = float(position[2]-initial[2])
        bilateral = gripper.bilateral()
        maximum_lift = max(maximum_lift, lift)
        dt = sim.model.opt.timestep
        lifted_hold = lifted_hold+dt if lift > .05 and bilateral else 0.
        best_hold = max(best_hold, lifted_hold)
        speed = float(np.linalg.norm(sim.data.qvel[dof:dof+3]))
        at_goal = (maximum_lift > .055 and best_hold >= 1.5 and
                   np.linalg.norm(position[:2]-goal[:2]) < .04 and abs(position[2]-goal[2]) < .01
                   and speed < .01 and not gripper.contacts())
        settled = settled+dt if at_goal else 0.
        if len(samples) == 0 or sim.data.time-samples[-1]['time'] >= .05-1e-9:
            samples.append({'time': float(sim.data.time), 'position': position.tolist(),
                            'bilateral': bool(bilateral), 'lift_m': lift})
        return value

    sim.step = measured_step
    error = None
    front = wrist = policy = None
    metadata=None;video=None
    observation_count=0
    def save_observation(observation):
        nonlocal observation_count
        if observation_count%25==0:
            import imageio.v3 as iio
            directory=args.output/'observations';directory.mkdir(exist_ok=True)
            for name in ('front_rgb','wrist_rgb'):
                iio.imwrite(directory/f'{observation_count:04d}_{name}.png',observation[name])
            (directory/f'{observation_count:04d}.json').write_text(json.dumps({
                'timestamp':observation['timestamp'],'state':observation['state'].tolist(),
                'prompt':observation['prompt']},indent=2)+'\n')
        observation_count+=1
    try:
        phase=args.policy
        if args.policy == 'expert':
            class OracleReleaseExpert(PickPlace):
                def move(self, position, seconds, stage):
                    position = np.asarray(position).copy()
                    if stage == 'lower':
                        position[:2] -= (self.sim.data.xpos[self.object_body]-self.arm.pose()[0])[:2]
                    return super().move(position, seconds, stage)
            OracleReleaseExpert(sim).run(initial, goal)
        else:
            executor = ActionChunkExecutor(sim, arm, gripper, base_hold=hold)
            front, wrist = RGBDCamera(sim), RGBDCamera(sim, camera_name='wrist_rgb')
            class HoldPolicy:
                def infer(self, obs):
                    assert 'front_rgb' in obs and 'wrist_rgb' in obs
                    return {'actions': np.tile(np.r_[arm.reference, 0.], (10, 1))}
            if args.policy == 'mock_hold':
                policy = HoldPolicy()
            else:
                sys.path.insert(0, str(args.client_source))
                policy = connect_openpi_policy(args.host, args.port,compression=args.websocket_compression,noise_seed=args.policy_noise_seed)
                metadata = policy.get_server_metadata()
                require_checkpoint(metadata, args.expected_checkpoint_label)
                if args.policy_noise_seed is not None and metadata.get('episode_policy_noise_seed')!=args.policy_noise_seed:
                    raise ValueError('Server evaluation noise seed mismatch')
                if metadata.get('actual_cameras') != ['front', 'wrist'] or metadata.get('action_semantics') != 'ur5_absolute_joint_goal_v1':
                    raise ValueError('Server camera or action semantics mismatch')
            while sim.data.time-start < args.seconds-1e-8 and settled < 1.:
                queries.append(query_and_execute(sim, arm, gripper, front, policy,
                    meta['language_instruction'], executor=executor,
                    execute_actions=args.execute_actions, wrist_camera=wrist,
                    max_latency_seconds=args.max_response_seconds,
                    interpolate_lookahead=args.interpolate_lookahead,
                    network_image_size=args.network_image_size,observation_callback=save_observation))
    except Exception as exc:
        error = {'category': type(exc).__name__, 'message': str(exc)}
    finally:
        if front:
            front.close()
        if wrist:
            wrist.close()
        if args.policy == 'openpi' and policy is not None:
            policy._ws.close()
        if recorder:
            try:video=recorder.close()
            except Exception as exc:video={'status':'FAIL','error':str(exc)}
    position = sim.data.xpos[sim.model.body('object').id].copy()
    placement = float(np.linalg.norm(position[:2]-goal[:2]))
    speed = float(np.linalg.norm(sim.data.qvel[dof:dof+3]))
    success = bool(error is None and maximum_lift > .055 and best_hold >= 1.5 and
                   placement < .04 and abs(position[2]-goal[2]) < .01 and speed < .01 and
                   not gripper.contacts() and not sim.data.warning.number.any() and not hold.faulted)
    result = {'success': success, 'policy': args.policy, 'seed': meta['seed'], 'skill': meta['skill'],
              'scope': 'normal reset, parked mobile single fruit; mock is not VLA', 'error': error,
              'sim_seconds': float(sim.data.time-start), 'wall_seconds': time.monotonic()-wall,
              'maximum_lift_m': maximum_lift, 'continuous_bilateral_hold_seconds': best_hold,
              'placement_error_m': placement, 'final_speed_m_s': speed,
              'warnings': sim.data.warning.number.tolist(), 'parking_fault': hold.faulted,
              'execute_actions': args.execute_actions, 'queries': len(queries)}
    result.update(server_metadata=metadata,video=video,network_image_size=args.network_image_size,
        simulation_paused_during_network=True,rollout_kind='development',
        unexpected_contacts=unexpected_contacts,websocket_compression=args.websocket_compression,
        maximum_response_wall_seconds=args.max_response_seconds,
        interpolate_lookahead=args.interpolate_lookahead,
        gripper_projected_chunks=sum(bool(q.get('gripper_projection') and q['gripper_projection']['projected_values']) for q in queries))
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'trajectory.json').write_text(json.dumps(samples)+'\n')
    (args.output/'queries.json').write_text(json.dumps(queries)+'\n')
    (args.output/'parking.json').write_text(json.dumps(hold.samples)+'\n')
    hold.release()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

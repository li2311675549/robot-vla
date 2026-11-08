"""Parked multi-fruit policy evaluation, separated from oracle-only scoring.

Initialization restores a saved, physically parked scene before execution.
Every subsequent movement goes through the existing actuator controller.
"""
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from robot_exam.control.arm import ArmController, GripperController
from robot_exam.control.parking import WheelParkingHold
from robot_exam.data.scene_quality import fruit_scene_preservation
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime, State
from robot_exam.tasks.scoring import fruit_score
from robot_exam.vla.client import query_and_execute
from robot_exam.vla.interface import ActionChunkExecutor, ACTION_SEMANTICS


def fruit_body_names(model):
    # The static support body is named fruit_table. It must never be scored as
    # an ungrasped fruit with a centre below the table plane.
    return [model.body(i).name for i in range(model.nbody)
            if (model.body(i).name or '').startswith('fruit_')
            and model.body(i).name.removeprefix('fruit_').isdigit()
            and model.body_jntnum[i] > 0]


def check_server_metadata(metadata, checkpoint, expected_norm=None):
    if metadata.get('action_semantics') != ACTION_SEMANTICS:
        raise ValueError('Server action semantics mismatch')
    if metadata.get('actual_cameras') != ['front', 'wrist']:
        raise ValueError('Require the two actual trained cameras')
    if checkpoint is not None and str(metadata.get('checkpoint_step_label')) != str(checkpoint):
        raise ValueError('Server checkpoint differs from evaluation protocol')
    if metadata.get('repo_id') != 'local/robot_exam_stage21_train':
        raise ValueError('This evaluation requires the new multi-fruit model, not an old single-fruit checkpoint')
    if expected_norm is not None and metadata.get('norm_stats_sha256') != expected_norm:
        raise ValueError('Server normalization differs from the frozen mixed training assets')


def restore_fixture(path):
    path = Path(path)
    meta = json.loads((path/'episode.json').read_text())
    if (meta.get('schema_version') != 3 or meta.get('action_semantics') != ACTION_SEMANTICS
            or meta.get('sensor_schema') != 'front_wrist_rgb_v1'
            or meta.get('scene_object_count') not in (3, 5, 10)
            or meta.get('category') not in ('apple', 'banana', 'orange')
            or meta.get('phase') not in ('grasp', 'place')):
        raise ValueError('Expected qualified Schema3 multi-fruit phase initialization')
    if not meta.get('parking_ready_at_start') or meta.get('parking_stable_seconds', 0) < 2-1e-6:
        raise ValueError('Missing actual initial two-second parking witness')
    model = mujoco.MjModel.from_binary_path(str(path/'model.mjb'))
    sim = Simulation.__new__(Simulation)
    sim.model, sim.data, sim.step_controllers = model, mujoco.MjData(model), []
    # Initialization only; never called again during the rollout.
    with np.load(path/'replay.npz') as raw:
        mujoco.mj_setState(model, sim.data, raw['initial_state'], int(raw['state_spec']))
        mujoco.mj_forward(model, sim.data)
        arm, gripper = ArmController(sim), GripperController(sim)
        arm.reference[:] = raw['initial_arm_reference']
    parking = WheelParkingHold(sim)
    parking.begin()
    parking.anchor = np.asarray(meta['parking_anchor'])
    parking.wheel_anchor = np.asarray(meta['parking_wheel_anchor'])
    parking.stable_seconds = float(meta['parking_stable_seconds'])
    parking.ready = parking.armed = True
    sim.check()
    return sim, arm, gripper, parking, meta


def evaluate(path, output, policy_kind, *, policy=None, checkpoint=None,
             seconds=30., record=False, max_response_seconds=2., network_image_size=224,
             expected_norm=None):
    if policy_kind not in ('openpi', 'replay', 'mock_hold'):
        raise ValueError('Unknown policy kind')
    if not 0 < seconds <= 60 or not 0 < max_response_seconds <= 2:
        raise ValueError('Bounded phase and response budgets required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Keep every rollout, including infrastructure failures')
    output.mkdir(parents=True)
    sim = arm = parking = front = wrist = recorder = None
    queries, contacts, trajectory = [], [], []
    metadata, video, error = None, None, None
    result = {'status': 'FAIL', 'success': False, 'policy_kind': policy_kind,
              'scope': 'Parked multi-fruit phase, not5m or complete10fruit classification',
              'source_fixture': str(Path(path).resolve()), 'model_inference': policy_kind == 'openpi',
              'initial_state_restores': 0, 'runtime_state_restores': 0,
              'phase_budget_seconds': seconds, 'response_budget_seconds': max_response_seconds,
              'policy_input_keys': ['front_rgb', 'wrist_rgb', 'state', 'prompt', 'timestamp'],
              'simulation_paused_during_network': True}
    started = time.monotonic()
    start_time = 0.
    try:
        sim, arm, gripper, parking, meta = restore_fixture(path)
        result.update(initial_state_restores=1, phase=meta['phase'], seed=meta['seed'],
                      category=meta['category'], scene_object_count=meta['scene_object_count'],
                      language_instruction=meta['language_instruction'])
        start_time = float(sim.data.time)
        monitor = TaskRuntime(sim)
        monitor.state = State.APPROACH
        fruit_names = fruit_body_names(sim.model)
        target = meta['target_body']
        initial_z = {name: float(sim.data.xpos[sim.model.body(name).id, 2]) for name in fruit_names}
        max_lift = {name: 0. for name in fruit_names}
        last_score = {}
        if record:
            from robot_exam.evaluation.recording import VideoRecorder
            recorder = VideoRecorder(output/'video', fps=5)

        class Witness:
            def before_step(self):
                pass

            def after_step(self):
                for name in fruit_names:
                    lift = float(sim.data.xpos[sim.model.body(name).id, 2])-initial_z[name]
                    max_lift[name] = max(max_lift[name], lift)
                contacts.append(bool(monitor.bilateral(target)))
                monitor.check_collisions()
                if monitor.collision_events:
                    raise RuntimeError('Unsafe static contact recorded by existing collision monitor')
                if not trajectory or sim.data.time-trajectory[-1]['time'] >= .05-1e-8:
                    trajectory.append({'time': float(sim.data.time), 'bilateral': contacts[-1],
                        'target_position': sim.data.xpos[sim.model.body(target).id].tolist(),
                        'measured_arm': sim.data.qpos[arm.qadr].tolist(),
                        'parking': parking.metrics()})
                if recorder:
                    recorder(sim, meta['phase'])
                if time.monotonic()-started > 600:
                    raise TimeoutError('Wall-clock phase budget exceeded')

        sim.add_step_controller(Witness())
        executor = ActionChunkExecutor(sim, arm, gripper, base_hold=parking)
        if policy_kind == 'replay':
            frames = [json.loads(s) for s in (Path(path)/'frames.jsonl').read_text().splitlines()]
            actions = np.asarray([f['action'] for f in frames])

            class Replay:
                def infer(self, obs):
                    index = min(int(round((obs['timestamp']-start_time)*5)), len(actions)-1)
                    return {'actions': actions[np.minimum(np.arange(index, index+10), len(actions)-1)]}
            policy = Replay()
        elif policy_kind == 'mock_hold':
            class Hold:
                def infer(self, obs):
                    return {'actions': np.tile(obs['state'], (10, 1))}
            policy = Hold()
        else:
            if policy is None:
                raise ValueError('Connect to an explicitly authorized real policy server')
            metadata = policy.get_server_metadata()
            check_server_metadata(metadata, checkpoint, expected_norm)
        front, wrist = RGBDCamera(sim), RGBDCamera(sim, camera_name='wrist_rgb')

        def save_observation(obs):
            if set(obs) != set(result['policy_input_keys']):
                raise ValueError('Hidden truth or missing camera in policy input')
            if not queries:
                import imageio.v3 as iio
                for camera in ('front_rgb', 'wrist_rgb'):
                    iio.imwrite(output/(camera+'.png'), obs[camera])
                (output/'first_observation.json').write_text(json.dumps({
                    'state': obs['state'].tolist(), 'prompt': obs['prompt'], 'timestamp': obs['timestamp']}))

        stable_place = 0.
        while sim.data.time-start_time < seconds-1e-8:
            queries.append(query_and_execute(sim, arm, gripper, front, policy,
                meta['language_instruction'], executor=executor, wrist_camera=wrist,
                execute_actions=1, interpolate_lookahead=True, network_image_size=network_image_size,
                max_latency_seconds=max_response_seconds, observation_callback=save_observation))
            if meta['phase'] == 'grasp':
                lift = float(sim.data.xpos[sim.model.body(target).id, 2])-initial_z[target]
                last_second = contacts[-round(1./sim.model.opt.timestep):]
                bilateral_fraction = float(np.mean(last_second))
                # Same thresholds as qualified low-rate replay, with a full
                # observed second required before deciding success.
                physical = (lift >= .055 and len(last_second) >= round(1./sim.model.opt.timestep)
                            and bilateral_fraction >= .9 and last_second[-1])
                last_score = {'lift_m': lift, 'last_second_bilateral_fraction': bilateral_fraction,
                              'final_bilateral': contacts[-1]}
            else:
                scene = json.loads((Path(path).parent/'scene.json').read_text())
                item = next(x for x in fruit_score(sim, scene)['objects'] if x['body'] == target)
                stable_place = stable_place+.2 if item['correct_bin'] and not contacts[-1] else 0.
                physical = stable_place >= .5
                last_score = {'placement': item, 'released_stable_seconds': stable_place,
                              'final_bilateral': contacts[-1]}
            auxiliary = fruit_scene_preservation(sim, {'objects': [{'body': n} for n in fruit_names]}, target)
            if physical and auxiliary['success']:
                result.update(success=True, status='PASS')
                break
        result.update(score=last_score, non_target_preservation=auxiliary,
                      max_lift_by_body_m=max_lift,
                      other_fruit_lifted=[n for n in fruit_names if n != target and max_lift[n] >= .055],
                      collision_events=monitor.collision_events)
        if result['other_fruit_lifted']:
            result.update(success=False, status='FAIL', failure_category='wrong_target_or_multiple_grasp')
        elif not result['success']:
            result['failure_category'] = 'phase_timeout_or_target_not_verified'
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        result.update(success=False, status='FAIL', failure_category=getattr(exc, 'category', type(exc).__name__))
    finally:
        for camera in (front, wrist):
            if camera is not None:
                camera.close()
        if recorder:
            try:
                video = recorder.close()
            except Exception as exc:
                video = {'status': 'FAIL', 'error': str(exc)}
                result.update(success=False, status='FAIL', failure_category='video_failure')
        if sim is not None:
            result.update(sim_seconds=float(sim.data.time)-start_time, warnings=sim.data.warning.number.tolist())
        if parking is not None:
            result.update(parking_final=parking.metrics(), parking_faulted=parking.faulted)
            if parking.faulted or not parking.ready:
                result.update(success=False, status='FAIL', failure_category='parking_drift')
            parking.release()
        result.update(error=error, wall_seconds=time.monotonic()-started, queries=len(queries),
                      server_metadata=metadata, video=video)
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        (output/'queries.json').write_text(json.dumps(queries)+'\n')
        (output/'trajectory.json').write_text(json.dumps(trajectory)+'\n')
    return result

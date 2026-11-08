#!/usr/bin/env python3
"""One parked-mobile dual-view rollout: identical physical score for expert/mock/VLA."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import _bootstrap  # noqa: F401 - add project src before importing robot_exam
from robot_exam.simulation import Simulation
from robot_exam.control.arm import ArmController, GripperController
from robot_exam.control.parking import WheelParkingHold
from robot_exam.perception.rgbd import RGBDCamera
from robot_exam.skills.pick_place import PickPlace
from robot_exam.vla.interface import ActionChunkExecutor
from robot_exam.vla.client import query_and_execute, connect_openpi_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--policy', choices=('expert', 'mock_hold', 'openpi'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--client-source', type=Path, default=Path('/home/li/vla/openpi/packages/openpi-client/src'))
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--execute-actions', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve every rollout')
    if not 0 < args.seconds <= 120 or not 1 <= args.execute_actions <= 10:
        parser.error('Use a 0..120 second budget and 1..10 prefix actions')
    meta = json.loads((args.fixture/'episode/episode.json').read_text())
    if meta['sensor_schema'] != 'front_wrist_rgb_v1' or meta['base_control'] != 'encoder_parking_v1':
        raise ValueError('Only real dual-camera mobile fixtures supported')
    reference = json.loads((args.fixture/'skill_result.json').read_text())
    goal = np.asarray(reference['drop'])
    args.output.mkdir(parents=True)
    sim = Simulation(args.fixture/'scene.xml')
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

    def measured_step(ctrl, count=1):
        nonlocal maximum_lift, best_hold, lifted_hold, settled
        if count != 1:
            raise RuntimeError('Single-step actuator control required')
        if sim.data.time-start > args.seconds or time.monotonic()-wall > 600:
            raise TimeoutError('Bounded rollout limit reached')
        value = step(ctrl, count)
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
    try:
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
                policy = connect_openpi_policy(args.host, args.port)
                metadata = policy.get_server_metadata()
                if metadata.get('actual_cameras') != ['front', 'wrist'] or metadata.get('action_semantics') != 'ur5_absolute_joint_goal_v1':
                    raise ValueError('Server camera or action semantics mismatch')
            while sim.data.time-start < args.seconds-1e-8 and settled < 1.:
                queries.append(query_and_execute(sim, arm, gripper, front, policy,
                    meta['language_instruction'], executor=executor,
                    execute_actions=args.execute_actions, wrist_camera=wrist))
    except Exception as exc:
        error = {'category': type(exc).__name__, 'message': str(exc)}
    finally:
        if front:
            front.close()
        if wrist:
            wrist.close()
        if args.policy == 'openpi' and policy is not None:
            policy._ws.close()
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
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'trajectory.json').write_text(json.dumps(samples)+'\n')
    (args.output/'queries.json').write_text(json.dumps(queries)+'\n')
    (args.output/'parking.json').write_text(json.dumps(hold.samples)+'\n')
    hold.release()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Phase-aware5Hz ActionChunkExecutor replay in the original full fruit scene.

Only the saved initial integration state and controller initial state are
restored. Every subsequent motion uses normal arm/gripper/wheel controls.
"""
import argparse
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.control.arm import ArmController, GripperController
from robot_exam.control.parking import WheelParkingHold
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime
from robot_exam.tasks.scoring import fruit_score
from robot_exam.vla.interface import ActionChunkExecutor


def replay(path, output):
    path, output = Path(path), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous replay result')
    meta = json.loads((path/'episode.json').read_text())
    raw = np.load(path/'replay.npz')
    frames = [json.loads(x) for x in (path/'frames.jsonl').read_text().splitlines()]
    if meta.get('scene_object_count') not in (3,5,10) or meta.get('sensor_schema') != 'front_wrist_rgb_v1':
        raise ValueError('Expected explicit3/5curriculum or10fruit dual-view phase')
    ready = meta.get('parking_ready_at_start', False)
    stable = float(meta.get('parking_stable_seconds', 0))
    parking_proof = 'episode initial controller metadata'
    if not ready:
        # Early pilot metadata omitted the counter. Recover it from the actual
        # same-run sampled parking trace, not a fabricated replay warmup.
        trace_path = path.parent/'parking.json.gz'
        if not trace_path.is_file():
            raise ValueError('Missing source proof of initial physical parking stability')
        with gzip.open(trace_path, 'rt') as stream:
            history = json.load(stream)
        first_time = float(raw['timestamps'][0])
        witnesses = [s for segment in history if np.allclose(segment['anchor'], meta['parking_anchor'])
                     for s in segment['samples']
                     if -float(meta['physics_timestep'])-1e-8 <= first_time-s['time'] <= .051
                     and s.get('ready')]
        if witnesses:
            witness = min(witnesses, key=lambda s: abs(s['time']-first_time))
            stable = float(witness['stable_seconds'])-max(0., witness['time']-first_time)
            ready = True
            parking_proof = f'{trace_path}: actual sample time{witness["time"]}, counter adjusted to clip time'
    if not ready or stable < 2.-1e-6:
        raise ValueError('Missing source proof of initial physical parking stability')
    model = mujoco.MjModel.from_binary_path(str(path/'model.mjb'))
    sim = Simulation.__new__(Simulation)
    sim.model, sim.data, sim.step_controllers = model, mujoco.MjData(model), []
    mujoco.mj_setState(model, sim.data, raw['initial_state'], int(raw['state_spec']))
    mujoco.mj_forward(model, sim.data)
    arm, gripper = ArmController(sim), GripperController(sim)
    arm.reference[:] = raw['initial_arm_reference']
    parking = WheelParkingHold(sim)
    parking.begin()
    parking.anchor = np.asarray(meta['parking_anchor'])
    parking.wheel_anchor = np.asarray(meta['parking_wheel_anchor'])
    parking.stable_seconds = stable
    parking.ready = parking.armed = True
    executor = ActionChunkExecutor(sim, arm, gripper, base_hold=parking)
    monitor = TaskRuntime(sim)
    body = meta['target_body']
    initial_height = float(sim.data.xpos[model.body(body).id, 2])
    contacts = []
    result = {'status': 'FAIL', 'phase': meta['phase'], 'target_body': body,
              'source_episode': str(path), 'scope': 'physical low-rate phase replay, not model inference',
              'executor': 'ActionChunkExecutor5Hz full-horizon interpolate, replan each0.2s',
              'initialization': 'saved integration+arm reference+verified parking anchors; no extra warmup',
              'initial_parking_proof': parking_proof,
              'qpos_writes_after_initialization': 0}
    actions = np.asarray([f['action'] for f in frames])
    count = len(raw['timestamps'])
    steps_per_frame = round((1./meta['frame_rate'])/model.opt.timestep)
    step = 0
    try:
        while step < count:
            index = min(step//steps_per_frame, len(actions)-1)
            chunk = actions[index:min(index+10, len(actions))]
            executor.submit(chunk, float(sim.data.time))
            for _ in range(min(steps_per_frame, count-step)):
                executor.step()
                contacts.append(bool(monitor.bilateral(body)))
                step += 1
            executor.chunk = None
        if meta['phase'] == 'grasp':
            lift = float(sim.data.xpos[model.body(body).id, 2]-initial_height)
            hold = contacts[-round(1./model.opt.timestep):]
            success = lift >= .055 and bool(hold) and float(np.mean(hold)) >= .9 and contacts[-1]
            result.update(lift_m=lift, last_second_bilateral_fraction=float(np.mean(hold)),
                          final_bilateral=contacts[-1])
        elif meta['phase'] == 'place':
            metadata = json.loads((path.parent/'scene.json').read_text())
            item = next(x for x in fruit_score(sim, metadata)['objects'] if x['body'] == body)
            success = item['correct_bin'] and not contacts[-1]
            result.update(placement=item, final_bilateral=contacts[-1])
        else:
            raise ValueError('Unknown phase')
        result['physical_phase_success'] = bool(success)
        result['status'] = 'PASS' if success and parking.ready and not sim.data.warning.number.any() else 'FAIL'
    except Exception as exc:
        result.update(error=str(exc), error_type=type(exc).__name__)
    result.update(executed_steps=step, warnings=sim.data.warning.number.tolist(),
                  parking_final=parking.metrics(), parking_faulted=bool(parking.faulted))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('episode', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = replay(args.episode, args.output)
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)

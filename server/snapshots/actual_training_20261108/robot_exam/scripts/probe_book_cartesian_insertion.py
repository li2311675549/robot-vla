#!/usr/bin/env python3
"""Local actuator-only insertion with an orientation-constrained lowering path.

The wrapper replaces one move inside the shared insertion skill in this test
process only. No production controller is patched on disk or state teleported.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime, TaskFailure
from robot_exam.control.joint_rrt import plan_joint_rrt
from robot_exam.tasks import drawer
from robot_exam.tasks.manipulation import side_rotation
from robot_exam.tasks.scoring import drawer_score


def angle(a, b):
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--opposite-axis', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous physical probes')
    a.output.mkdir(parents=True)
    events = json.loads((a.episode / 'fsm.json').read_text())
    index = next(i for i, x in enumerate(events) if x['reason'] == 'side_book_drawer_insertion')
    checkpoint = a.episode / 'checkpoints' / f'{index + 1:04d}.npz'
    saved = np.load(checkpoint)
    sim = Simulation(a.episode / 'scene.xml')
    mujoco.mj_setState(sim.model, sim.data, saved['integration_state'], int(saved['state_spec']))
    mujoco.mj_forward(sim.model, sim.data)
    r = TaskRuntime(sim, output=a.output / 'execution', sim_limit=sim.data.time + 180)
    r.config = {**json.loads((a.episode / 'config.json').read_text()), 'compress_bulk_logs': True}
    r.qgoal = sim.data.qpos[r.arm.qadr].copy()
    r.arm.reference = r.qgoal.copy()
    meta = json.loads((a.episode / 'scene.json').read_text())
    obj = next(x for x in meta['objects'] if x['rank'] == 0)
    original = drawer._checked_side_book_move
    traces = []

    def constrained_move(runtime, position, rotation, body, allow_held_contact=True):
        start, current_rotation = runtime.arm.pose()
        position = np.asarray(position)
        lowering = (allow_held_contact and position[2] < start[2] - .04
                    and angle(current_rotation, rotation) < .04)
        if not allow_held_contact:
            # Independently verified released-book recovery: vertical retreat
            # avoids the low backward target's wrist/forearm intersection.
            return original(runtime, start + [0, 0, .06], rotation, body, False)
        if not lowering:
            return original(runtime, position, rotation, body, allow_held_contact)
        count = max(2, int(np.ceil(np.linalg.norm(position - start) / .025)))
        for step in range(1, count + 1):
            target = start + (position - start) * step / count
            q, _ = runtime.arm.solve(target, rotation, seed=runtime.qgoal, max_iterations=500)
            path, report = plan_joint_rrt(runtime.arm, q, allowed_bodies=[body], wall_seconds=3.)
            scratch = runtime.arm.scratch
            q0 = runtime.sim.data.qpos[runtime.arm.qadr].copy()
            # RRT may find a collision-free joint detour that flips a held
            # book. Check orientation along every interpolation segment.
            peak = 0.
            for waypoint in path:
                for alpha in np.linspace(0, 1, 9):
                    scratch.qpos[:] = runtime.sim.data.qpos
                    scratch.qpos[runtime.arm.qadr] = q0 + alpha * (waypoint - q0)
                    mujoco.mj_forward(runtime.sim.model, scratch)
                    peak = max(peak, angle(rotation, scratch.site_xmat[runtime.arm.site].reshape(3, 3)))
                q0 = waypoint
            if peak > .12:
                raise TaskFailure('book_orientation_path', f'Unacceptable held-book path rotation {peak:.4f}rad')
            for waypoint in path:
                runtime.joint_move(waypoint, seconds=.7)
            traces.append({'time': float(sim.data.time), 'target': target.tolist(),
                           'planned_peak_rotation_rad': peak, 'planning': report,
                           'bilateral': runtime.bilateral(body),
                           'object_position': sim.data.xpos[sim.model.body(body).id].tolist()})
            if not runtime.bilateral(body):
                raise TaskFailure('gripper_slip', 'Book contact lost during bounded-orientation lowering')

    if a.opposite_axis:
        original_alignment = drawer.rectangular_axis_alignment
        drawer.rectangular_axis_alignment = lambda yaw: original_alignment(yaw) + np.pi
    drawer._checked_side_book_move = constrained_move
    result = {'scope': 'local restored real-grasp insertion, not complete5m task',
              'source_checkpoint': str(checkpoint), 'opposite_axis': a.opposite_axis,
              'success': False}
    try:
        ops = []
        drawer._insert_side_book(r, meta, 0, {'area': obj['area']}, obj,
                                 {'rotation': side_rotation(), 'lift_m': 0., 'scope': result['scope']}, {}, ops)
        item = next(x for x in drawer_score(sim, meta, np.asarray(meta['start']))['books'] if x['rank'] == 0)
        result.update(success=bool(item['inside_correct_layer'] and abs(item['opening']) < .012),
                      book_score=item, operations=ops)
    except RuntimeError as exc:
        result.update(error=str(exc), failure_category=getattr(exc, 'category', 'planning'))
    r.save()
    result.update(sim_seconds=float(sim.data.time) - float(saved['time']), traces=traces,
                  data_warning=sim.data.warning.number.tolist(), collision_events=r.collision_events)
    if r.collision_events or sim.data.warning.number.any():
        result['success'] = False
    (a.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'traces'}), flush=True)


if __name__ == '__main__':
    main()

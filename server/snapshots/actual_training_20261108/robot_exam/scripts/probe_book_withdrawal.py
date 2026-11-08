#!/usr/bin/env python3
"""Diagnose a released-book withdrawal from a documented real checkpoint.

Only initialization restores integration state. Candidate IK and collision
queries use scratch data; optional execution uses ordinary actuator commands.
This is a local recovery test, never a complete 5m storage-task success.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import ROOT
from robot_exam.control.joint_rrt import plan_joint_rrt
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import TaskRuntime, State


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--finish-drawer', action='store_true')
    p.add_argument('--rank', type=int, default=0)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve prior recovery experiments')
    a.output.mkdir(parents=True)
    events = json.loads((a.episode / 'fsm.json').read_text())
    checkpoint = a.episode / 'checkpoints' / f'{len(events):04d}.npz'
    saved = np.load(checkpoint)
    sim = Simulation(a.episode / 'scene.xml')
    mujoco.mj_setState(sim.model, sim.data, saved['integration_state'], int(saved['state_spec']))
    mujoco.mj_forward(sim.model, sim.data)
    r = TaskRuntime(sim, output=a.output / 'execution', sim_limit=sim.data.time + 90)
    r.config = {**json.loads((a.episode / 'config.json').read_text()), 'compress_bulk_logs': True}
    r.qgoal = sim.data.qpos[r.arm.qadr].copy()
    r.arm.reference = r.qgoal.copy()
    before = sim.data.qpos.copy()
    position, rotation = r.arm.pose()
    candidates = []
    # Raise before retreating when a low, rearward Cartesian target folds the
    # wrist into the forearm. Do not assume an isolated endpoint is enough.
    for delta in ([0, 0, .06], [0, 0, .12], [0, 0, .20],
                  [-.04, 0, .12], [-.08, 0, .16], [-.14, 0, .025]):
        row = {'delta': delta, 'feasible': False, 'errors': []}
        for seed in (r.qgoal, [0, -1.5, 2, -2, 1.57, 0], [0, -1.5, -2, 1, -1.57, 0]):
            try:
                q, metrics = r.arm.solve(position + delta, rotation, seed=seed, max_iterations=500)
                path, report = plan_joint_rrt(r.arm, q, wall_seconds=3.)
                row.update(feasible=True, q=q.tolist(), path=[x.tolist() for x in path],
                           ik=metrics, planning=report)
                break
            except RuntimeError as exc:
                row['errors'].append(str(exc))
        candidates.append(row)
    np.testing.assert_array_equal(before, sim.data.qpos)
    result = {'scope': 'local released-book withdrawal, not full task',
              'source_checkpoint': str(checkpoint), 'initial_time': float(saved['time']),
              'candidate_search_did_not_change_live_qpos': True, 'candidates': candidates,
              'executed': False, 'success': False}
    feasible = [x for x in candidates if x['feasible']]
    if a.execute and feasible:
        selected = feasible[0]
        r.transition(State.PLACE, 'checkpoint_released_book_vertical_withdrawal', timeout=60)
        try:
            for q in selected['path']:
                r.joint_move(np.asarray(q), seconds=2.)
            r.hold(1.)
            error = float(np.linalg.norm(r.arm.pose()[0] - position - selected['delta']))
            result.update(executed=True, selected=selected['delta'], endpoint_error_m=error,
                          success=bool(error < .018 and not r.collision_events and not sim.data.warning.number.any()))
            if a.finish_drawer:
                from robot_exam.tasks.manipulation import drawer_operation
                from robot_exam.tasks.scoring import drawer_score
                meta = json.loads((a.episode / 'scene.json').read_text())
                r.stow()
                drawer_operation(r, a.rank, 0., meta)
                score = drawer_score(sim, meta, np.asarray(meta['start']))
                item = next(x for x in score['books'] if x['rank'] == a.rank)
                result['book_score_after_closure'] = item
                result['success'] = bool(result['success'] and item['inside_correct_layer']
                                         and abs(item['opening']) < .012)
        except RuntimeError as exc:
            result.update(executed=True, success=False, error=str(exc))
        r.save()
    result.update(sim_seconds=float(sim.data.time) - float(saved['time']),
                  data_warning=sim.data.warning.number.tolist(), collision_events=r.collision_events)
    (a.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'candidates'}), flush=True)


if __name__ == '__main__':
    main()

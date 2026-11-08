#!/usr/bin/env python3
"""Paired strict dock forks from an actual multi-fruit pilot carry checkpoint."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from _bootstrap import ROOT
from robot_exam.navigation.grid import OccupancyGrid
from robot_exam.simulation import Simulation
from robot_exam.tasks.runtime import State, TaskRuntime
from robot_exam.control.arm import ArmController
from robot_exam.control.grasp_monitor import relative_grasp_pose
from types import SimpleNamespace


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--variants', nargs='+', choices=['production','preapproach','anchored_preapproach','braked_preapproach','direct_braked','direct_guarded','direct_pivot','lattice_pivot','direct_capture'],
                   default=['production','preapproach','anchored_preapproach'])
    p.add_argument('--timeout', type=float, default=60)
    p.add_argument('--restore-original-grasp-anchor', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep existing paired evidence')
    args.output.mkdir(parents=True)
    events = json.loads((args.parent/'fsm.json').read_text())
    index = next(i for i, e in enumerate(events) if e['reason'] == 'precision_bin_dock')
    saved = np.load(args.parent/'checkpoints'/f'{index+1:04d}.npz')
    source = json.loads((args.parent/'result.json').read_text())
    settings = json.loads((args.parent/'config.json').read_text())
    meta = json.loads((args.parent/'scene.json').read_text())
    goal = json.loads((args.parent/'base_pose_trials.json').read_text())['selected']['pose']
    variants = {'production':(None,None), 'preapproach':('preapproach','axis_bounded_v5'),
                'anchored_preapproach':('preapproach','anchored_v4'),
                'braked_preapproach':('preapproach','axis_bounded_v5'),
                'direct_braked':('direct','axis_bounded_v5'),'direct_guarded':('direct','axis_bounded_v5'),
                'direct_pivot':('direct','axis_pivot_v6'),'lattice_pivot':('lattice','axis_pivot_v6'),
                'direct_capture':('direct','axis_bounded_v5')}
    rows = []
    for name in args.variants:
        strategy, tracking = variants[name]
        folder = args.output/name
        folder.mkdir()
        sim = Simulation(args.parent/'scene.xml')
        mujoco.mj_setState(sim.model, sim.data, saved['integration_state'], int(saved['state_spec']))
        mujoco.mj_forward(sim.model, sim.data)
        r = TaskRuntime(sim, output=folder, sim_limit=float(sim.data.time)+(2*args.timeout+10 if name=='direct_guarded' else args.timeout+5))
        r.config = {**settings, 'stationary_manipulation': False}
        if strategy:
            r.config.update(local_dock_strategy=strategy, local_dock_tracking_version=tracking)
        if name in ('braked_preapproach','direct_braked','direct_guarded','direct_pivot','lattice_pivot','direct_capture'):
            r.config.update(payload_precision_twist_slew=True,payload_twist_slew_categories=['apple','banana','orange'])
        if name=='direct_capture':r.config['local_dock_capture_goal_during_primitives']=True
        r.qgoal = sim.data.qpos[r.arm.qadr].copy()
        r.arm.reference[:] = r.qgoal
        r.wheel_applied = sim.data.ctrl[r.base.ids].copy()
        body = source['target']['body']
        r.payload_motion_profile = source['category']
        r.begin_payload(body, r.arm.pose()[1], world_stabilize=True)
        if args.restore_original_grasp_anchor:
            original_index = next(i for i,e in enumerate(events) if e['reason']=='unrecorded_traditional_payload_transport')
            original_saved=np.load(args.parent/'checkpoints'/f'{original_index+1:04d}.npz')
            scratch=mujoco.MjData(sim.model)
            mujoco.mj_setState(sim.model,scratch,original_saved['integration_state'],int(original_saved['state_spec']))
            mujoco.mj_forward(sim.model,scratch)
            tool=ArmController(SimpleNamespace(model=sim.model,data=scratch)).pose()
            body_id=sim.model.body(body).id
            r.payload_relative,r.payload_relative_rotation=relative_grasp_pose(*tool,scratch.xpos[body_id],scratch.xmat[body_id])
            r.payload_world_relative=scratch.xpos[body_id]-tool[0]
        grid = OccupancyGrid()
        for box in meta['obstacles']:
            grid.rectangle(*box)
        r.grid = grid
        start = float(sim.data.time)
        row = {'variant': name, 'success': False}
        try:
            r.transition(State.NAVIGATE, 'paired_dock_from_actual_multiobject_carry', timeout=args.timeout+1)
            if name=='direct_guarded':
                from robot_exam.control.payload_dock import dock_held_object
                row['dock_attempts']=dock_held_object(r,body,goal,r.arm.pose()[1],
                    float(source['target']['initial'][2])+.055,'direct',timeout=args.timeout,maximum_recoveries=1)
            else:r.dock(goal, timeout=args.timeout, tolerance=.03)
            row['success'] = bool(r.bilateral(body) and not r.collision_events and not sim.data.warning.number.any())
        except Exception as exc:
            row.update(error=str(exc), error_type=type(exc).__name__)
        row.update(elapsed_sim_seconds=float(sim.data.time)-start, final_base=r.base.pose().tolist(),
                   final_bilateral=bool(r.bilateral(body)), collision_events=r.collision_events,
                   warning_counts=sim.data.warning.number.tolist())
        r.end_payload(); r.save()
        (folder/'result.json').write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.output/'summary.json').write_text(json.dumps({'scope': 'paired checkpoint dock forks; not autonomous full parent success',
        'source': str(args.parent), 'checkpoint': f'{index+1:04d}.npz', 'goal': goal,
        'restore_original_grasp_anchor':args.restore_original_grasp_anchor,
        'initialization': 'same integration checkpoint; all variants measured arm reference/carry orientation and saved actuator wheel command; original upstream controller history not claimed reconstructed',
        'results': rows}, indent=2))

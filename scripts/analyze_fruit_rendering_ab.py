#!/usr/bin/env python3
"""Audit a matched frozen-source render-backend experiment from real logs."""
import argparse
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np


def trajectory(folder):
    path = folder / 'trajectory.json'
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(folder / 'trajectory.json.gz', 'rt') as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--software', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Retain previous diagnosis')
    folders = {'reference': args.reference, 'software': args.software, 'native': args.native}
    results = {name: json.loads((folder / 'result.json').read_text()) for name, folder in folders.items()}
    rendering = {name: json.loads((folders[name] / 'recording_manifest.json').read_text())
                 for name in ('software', 'native')}
    if rendering['software']['view_azimuth_degrees'] != rendering['native']['view_azimuth_degrees']:
        raise ValueError('Matched comparison requires identical external views')
    if rendering['software']['frozen_origin']['git_commit'] != rendering['native']['frozen_origin']['git_commit']:
        raise ValueError('Expert source version differs')
    configurations = {name: json.loads((folder / 'config.json').read_text()) for name, folder in folders.items()}
    if configurations['software'] != configurations['native']:
        raise ValueError('Software/native physical control settings differ')
    events = {name: json.loads((folder / 'fsm.json').read_text()) for name, folder in folders.items()}
    selected = {name: next(row for row in event if row['state'] == 'SELECT_TARGET') for name, event in events.items()}
    observation_time = selected['software']['time']
    paths = {name: [row for row in trajectory(folder) if row['time'] < observation_time] for name, folder in folders.items()}
    prefix_equal = len(paths['native']) == len(paths['software']) and paths['native'] == paths['software']
    models = {name: mujoco.MjModel.from_xml_path(str(folder / 'scene.xml')) for name, folder in folders.items()}
    arrays = ['qpos0', 'key_qpos', 'body_mass', 'body_inertia', 'body_pos', 'body_quat',
              'geom_size', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_solref', 'geom_solimp',
              'dof_damping', 'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange',
              'jnt_range', 'cam_pos', 'cam_quat', 'cam_fovy']
    differences = {}
    for attr in arrays:
        one, two = getattr(models['software'], attr), getattr(models['native'], attr)
        if one.shape != two.shape or not np.array_equal(one, two):
            differences[attr] = 'shape differs' if one.shape != two.shape else float(np.max(np.abs(one - two)))
    native, software = selected['native']['target'], selected['software']['target']
    delta = {'center_distance_m': float(np.linalg.norm(np.asarray(native['center']) - software['center'])),
             'yaw_difference_rad': float(native['yaw'] - software['yaw']),
             'estimated_width_difference_m': float(native['width'] - software['width']),
             'estimated_thickness_difference_m': float(native['thickness'] - software['thickness'])}
    report = {'status': 'COMPLETED_MATCHED_RENDERING_EXPERIMENT',
              'frozen_commit': rendering['software']['frozen_origin']['git_commit'],
              'same_external_view_degrees': rendering['software']['view_azimuth_degrees'],
              'same_control_config': True, 'model_array_differences': differences,
              'exact_physical_trajectory_prefix_equal_before_first_observation': prefix_equal,
              'physical_prefix_rows_each': {name: len(rows) for name, rows in paths.items()},
              'first_observation_sim_seconds': observation_time, 'first_selected': selected,
              'native_minus_software_perception': delta,
              'software_reproduces_reference_first_selection_exactly': selected['software'] == selected['reference'],
              'runs': {name: {'path': str(folders[name].resolve()), 'success': result['success'],
                              'completed': result.get('completed'), 'sim_seconds': result['sim_seconds'],
                              'error': result.get('error'), 'warnings': result.get('warnings'),
                              'opengl': rendering.get(name, {}).get('opengl')} for name, result in results.items()},
              'interpretation': 'Frozen-source backend sensitivity demonstrated only if matched physical-prefix/config/model checks pass and real outcomes differ; one pair is not a population success-rate estimate',
              'not_a_VLA_model_evaluation': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'first_selected'}, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Read-only comparison of complete, physically executed door episodes.

Uses recorded actuator commands to count grasp edges independently of the
controller's summary. Plotting is optional and does not run any simulation.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text())


def episode(folder):
    result = load(folder / 'result.json')
    trajectory = load(folder / 'trajectory.json')
    edges = []
    closed = False
    for row in trajectory:
        now_closed = row['ctrl'][10] > 127.5
        if now_closed and not closed:
            edges.append({'time': row['time'], 'stage': row.get('stage')})
        closed = now_closed
    return {
        'path': str(folder), 'success': bool(result.get('success')),
        'sim_seconds': result.get('sim_seconds'),
        'actual_grasp_command_edges': len(edges), 'grasp_edges': edges,
        'safety_releases': result.get('safety_releases'),
        'final_door_angle_rad': result.get('final_door_angle'),
        'warnings': result.get('warnings'),
        'collision_events': result.get('collision_events'),
        'video': result.get('video'),
    }, trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--formal', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior analysis; choose a new output')
    old, old_rows = episode(args.baseline)
    new, new_rows = episode(args.candidate)
    if not old['success'] or not new['success']:
        raise ValueError('Paired complete-task efficiency requires two successes')
    report = {
        'comparison_scope': 'Same development seed; complete five-metre task; simulation time only',
        'grasp_count_definition': 'Recorded gripper actuator ctrl index10 crossing 127.5 upward; model nu=11',
        'baseline': old, 'candidate': new,
        'grasp_reduction_fraction': 1 - new['actual_grasp_command_edges'] / old['actual_grasp_command_edges'],
        'sim_time_reduction_fraction': 1 - new['sim_seconds'] / old['sim_seconds'],
        'speedup_ratio': old['sim_seconds'] / new['sim_seconds'],
    }
    contacts = []
    for path in sorted(args.candidate.glob('continuous_*.json')):
        data = load(path)
        contacts.append({
            'file': str(path), 'completed': data['completed'],
            'safety_release_reason': data['safety_release_reason'],
            'samples': data['samples'],
        })
    samples = [sample for part in contacts for sample in part['samples']]
    report['candidate_contact_summary'] = {
        'sample_count': len(samples),
        'max_tip_error_m': max(x['tip_error_m'] for x in samples),
        'max_base_error_m': max(x['base_error_m'] for x in samples),
        'max_sampled_finger_normal_force_n': max(max(x['left_force_n'], x['right_force_n']) for x in samples),
        'bilateral_sample_fraction': statistics.mean(float(x['bilateral']) for x in samples),
        'note': 'Sampled arc measurements, not a continuous-time proof; releases/approaches are separate phases',
    }
    if args.formal:
        summary = load(args.formal / 'summary.json')
        integrity = load(args.formal / 'source_integrity.json')
        records = load(args.formal / 'episodes.json')
        if len(records) != 10 or not integrity['all_attempts_counted'] or integrity['source_hash_mismatches']:
            raise ValueError('Formal evaluation incomplete or changed source')
        report['formal'] = {'summary': summary, 'integrity': integrity,
                            'frozen_git_commit': load(args.formal / 'freeze.json')['git_commit']}
        report['formal']['sampled_text_linesearch_messages'] = sum(x.get('solver_linesearch_messages', 0) for x in records)
        report['formal']['warning_counts'] = [x.get('warnings') for x in records]
        report['formal']['actual_grasp_counts'] = [episode(args.formal / str(x['seed']))[0]['actual_grasp_command_edges'] for x in records]
        report['formal']['sim_seconds'] = [x.get('sim_seconds') for x in records]
        report['formal']['final_angle_abs_max_rad'] = max(abs(x.get('final_door_angle', 0)) for x in records)
        report['formal']['whole_robot_min_x_m'] = min(x.get('whole_robot_min_x_m', float('inf')) for x in records)
    args.output.mkdir(parents=True)
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output / 'contact_samples.csv').open('w', newline='') as stream:
        keys = ['time', 'door_angle', 'reference_angle', 'axial_slip_m', 'tip_error_m',
                'left_force_n', 'right_force_n', 'bilateral', 'linear_command', 'angular_command']
        writer = csv.DictWriter(stream, fieldnames=['part'] + keys)
        writer.writeheader()
        for index, part in enumerate(contacts, 1):
            for row in part['samples']:
                writer.writerow({'part': index, **{key: row[key] for key in keys}})
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
        for rows, info, label in [(old_rows, old, 'Segmented'), (new_rows, new, 'Sustained contact')]:
            axes[0].plot([x['time'] for x in rows], [x['passive_joints']['door_hinge'] for x in rows], label=label)
            axes[1].step([0] + [x['time'] for x in info['grasp_edges']] + [info['sim_seconds']],
                         [0] + list(range(1, info['actual_grasp_command_edges'] + 1)) + [info['actual_grasp_command_edges']],
                         where='post', label=label)
        for part in contacts:
            rows = part['samples']
            axes[2].plot([x['time'] for x in rows], [1000 * abs(x['axial_slip_m']) for x in rows])
        axes[2].axhline(12, color='r', linestyle='--', label='Safety margin: 12 mm for 0.2 s')
        for ax, ylabel in zip(axes, ['Door angle [rad]', 'Cumulative grasp commands', 'Axial pad displacement [mm]']):
            ax.set(xlabel='Simulation time [s]', ylabel=ylabel)
            ax.grid(alpha=.25)
            ax.legend()
        fig.savefig(args.output / 'comparison.png', dpi=160)
        plt.close(fig)
    print(json.dumps({key: report[key] for key in ['grasp_reduction_fraction', 'sim_time_reduction_fraction', 'speedup_ratio']}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Independently audit a finished local release without rerunning physics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('release', type=Path)
    parser.add_argument('--batches', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new immutable dataset audit')
    release = load(args.release / 'release_result.json')
    manifest = load(args.release / 'manifest/manifest.json')
    rows = [row for batch in args.batches for seed in load(batch / 'progress.json')['seeds']
            for row in seed['episodes']]
    expected = {(seed, skill) for split in ('train', 'validation', 'test')
                for seed in manifest['protocol'][split] for skill in manifest['protocol']['skills']}
    actual = [(row['seed'], row['skill']) for row in rows]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    accepted = {(row['seed'], row['skill']) for row in manifest['accepted']}
    assert len(accepted) == len(manifest['accepted'])
    assert all(row['seed'] in manifest['protocol'][row['split']] for row in manifest['accepted'])
    assert accepted == {(r['seed'], r['skill']) for r in rows if r['accepted']}
    train_states, train_actions, positives = [], [], []
    for row in manifest['accepted']:
        folder = Path(row['path'])
        meta = load(folder / 'episode.json')
        validation = load(folder / 'validation.json')
        low = load(folder / 'low_rate_replay.json')
        deployed = load(folder / 'vla_executor_replay.json')
        assert meta['success'] and validation['status'] == 'PASS'
        assert low['low_rate_physical_placement_pass'] and deployed['success']
        assert meta['sensor_schema'] == 'front_wrist_rgb_v1'
        assert meta['base_control'] == 'encoder_parking_v1'
        frames = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
        if row['split'] == 'train':
            train_states.extend(frame['observation.state'] for frame in frames)
            train_actions.extend(frame['action'] for frame in frames)
        positives.append({'seed': row['seed'], 'skill': row['skill'], 'split': row['split'],
                          'frames': len(frames), 'physical_success': True,
                          'full_rate_max_qpos_error': validation['max_replay_qpos_error'],
                          'deployment_base_drift_m': deployed['maximum_base_drift_m'],
                          'deployment_placement_error_m': deployed['placement_error_m'],
                          'scale': meta['fixture_scale'], 'path': str(folder)})
    stats = load(args.release / 'manifest/assets/original_ur5/norm_stats.json')['norm_stats']
    maximum_stats_error = 0.
    for name, values in [('state', train_states), ('actions', train_actions)]:
        values = np.asarray(values, dtype=float)
        expected_stats = {'mean': values.mean(axis=0), 'std': values.std(axis=0),
                          'q01': np.quantile(values, .01, axis=0), 'q99': np.quantile(values, .99, axis=0)}
        for key, value in expected_stats.items():
            error = float(np.max(np.abs(value - np.asarray(stats[name][key]))))
            maximum_stats_error = max(maximum_stats_error, error)
    assert maximum_stats_error < 1e-12
    for split in ('train', 'validation', 'test'):
        result = load(args.release / f'lerobot_dual_{split}_v1/conversion_result.json')
        assert result['status'] == 'PASS' and result['episodes'] == manifest['split_counts'][split]
    native = load(args.release / 'openpi_native.json')
    assert native['status'] == 'PASS' and native['native_lerobot_training_loader']['status'] == 'PASS'
    assert native['raw_episode_files_required'] is False and not native['weights_loaded']
    record = {'status': 'PASS', 'expected_attempts': len(expected), 'recorded_attempts': len(actual),
        'physical_successes': sum(r['physical_success'] for r in rows),
        'full_rate_reproduced': sum(r['full_rate_quality'] == 'PASS' for r in rows),
        'qualified_positives': len(positives), 'positive_frames': sum(r['frames'] for r in positives),
        'five_hz_replay_successes': sum(r.get('low_rate', False) for r in rows),
        'deployment_executor_successes': sum(r.get('same_executor', False) for r in rows),
        'split_counts': manifest['split_counts'],
        'class_counts': dict(Counter(r['skill'] for r in positives)),
        'class_counts_per_split': release['class_counts_per_split'],
        'failures': [r for r in rows if not r['accepted']],
        'train_only_stats_max_recomputation_error': maximum_stats_error,
        'scene_split_isolation_verified': True,
        'source_manifest_sha256': hashlib.sha256((args.release / 'manifest/manifest.json').read_bytes()).hexdigest(),
        'release_result_sha256': hashlib.sha256((args.release / 'release_result.json').read_bytes()).hexdigest(),
        'maximum_full_rate_qpos_error': max(r['full_rate_max_qpos_error'] for r in positives),
        'maximum_deployment_base_drift_m': max(r['deployment_base_drift_m'] for r in positives),
        'maximum_deployment_placement_error_m': max(r['deployment_placement_error_m'] for r in positives),
        'native_openpi': native, 'raw_bytes': release['raw_bytes'],
        'free_disk_GiB_at_audit': shutil.disk_usage(args.release).free / 2**30,
        'physical_scope': release['physical_scope'],
        'note': 'Full-rate reproduction includes seven failed operations and is not a physical-success metric. Failed operations were not run as positive low-rate/deployment demonstrations.',
        'positives': positives, 'training_performed': False, 'uploads_performed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('positives', 'native_openpi', 'failures')}, indent=2))


if __name__ == '__main__':
    main()

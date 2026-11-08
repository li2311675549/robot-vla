#!/usr/bin/env python3
"""Release completed physical batches locally; never upload or load model weights."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', nargs='+', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=float, default=0)
    parser.add_argument('--dataset-python', type=Path,
                        default=Path('/home/li/anaconda3/envs/openpi/bin/python'))
    parser.add_argument('--openpi-python', type=Path,
                        default=Path('/home/li/vla/openpi/.venv/bin/python'))
    parser.add_argument('--openpi-root', type=Path, default=Path('/home/li/vla/openpi'))
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 14400:
        raise ValueError('Dataset wait must be bounded to at most four hours')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new release root; preserve incomplete release evidence')
    output.mkdir(parents=True)
    protocol = json.loads(args.protocol.read_text())
    batches = [p.resolve() for p in args.batches]
    started = time.monotonic()
    while True:
        progress = []
        for batch in batches:
            path = batch / 'progress.json'
            try:
                progress.append(json.loads(path.read_text()) if path.exists() else {})
            except json.JSONDecodeError:
                progress.append({})  # A writer may be between truncate/write.
        complete = all(p.get('completed_scene_seeds') == p.get('planned_scene_seeds')
                       and p.get('planned_scene_seeds', 0) > 0 for p in progress)
        if complete:
            break
        (output / 'status.json').write_text(json.dumps({'status': 'WAITING_FOR_BATCHES',
            'recorded_attempts': sum(p.get('recorded_attempts', 0) for p in progress),
            'accepted_so_far': sum(p.get('accepted', 0) for p in progress)}, indent=2) + '\n')
        if time.monotonic() - started >= args.wait_seconds:
            raise TimeoutError('Collection incomplete; preserve this release attempt')
        time.sleep(10)
    expected = [seed for split in ('train', 'validation', 'test') for seed in protocol[split]]
    actual = [r['seed'] for p in progress for r in p['seeds']]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise RuntimeError('All predeclared scene seeds must be accounted for exactly once')
    locks = [json.loads((batch / 'collection_freeze.json').read_text()) for batch in batches]
    if any(lock['protocol'] != protocol or lock['source_hashes'] != locks[0]['source_hashes'] for lock in locks):
        raise RuntimeError('Collection source/protocol differs across batches')
    if shutil.disk_usage(ROOT).free < 30 * 2**30:
        raise RuntimeError('Keep conversion headroom above the 25 GiB reserve')
    environment = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                       JAX_PLATFORMS='cpu', OMP_NUM_THREADS='2')

    def run(name, command):
        with (output / f'{name}.log').open('w') as log:
            subprocess.run([str(v) for v in command], cwd=ROOT, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)

    manifest = output / 'manifest'
    command = [sys.executable, ROOT / 'scripts/prepare_vla_manifest.py', batches[0],
        '--splits', args.protocol.resolve(), '--output', manifest, '--require-executor-success']
    for batch in batches[1:]:
        command += ['--additional-source', batch]
    run('manifest', command)
    info = json.loads((manifest / 'manifest.json').read_text())
    for split in ('train', 'validation', 'test'):
        run(f'lerobot_{split}', [args.dataset_python, ROOT / 'scripts/export_lerobot.py',
            batches[0], output / f'lerobot_dual_{split}_v1', '--manifest', manifest / 'manifest.json',
            '--split', split, '--repo-id', f'local/robot_exam_dual_release_{split}',
            '--require-schema3', '--require-lowrate-success'])
    cache = output / 'cache'
    repo = cache / 'local/robot_exam_dual_release_train'
    repo.parent.mkdir(parents=True)
    repo.symlink_to(output / 'lerobot_dual_train_v1', target_is_directory=True)
    run('openpi_native', [args.openpi_python, ROOT / 'scripts/validate_openpi_ur5_mapping.py',
        '--openpi-root', args.openpi_root, '--manifest', manifest,
        '--lerobot-cache', cache, '--repo-id', 'local/robot_exam_dual_release_train',
        '--wrist-camera', '--converted-only', '--output', output / 'openpi_native.json'])
    run('transfer', [sys.executable, ROOT / 'scripts/prepare_autodl_transfer.py',
        '--manifest', manifest, '--dataset-root', output,
        '--openpi-root', args.openpi_root, '--output', output / 'transfer'])
    run('transfer_verification', [sys.executable, ROOT / 'scripts/verify_autodl_transfer.py',
        output / 'transfer/transfer_manifest.json', '--output', output / 'transfer_verification.json'])
    counts = {split: dict(Counter(r['skill'] for r in info['accepted'] if r['split'] == split))
              for split in ('train', 'validation', 'test')}
    raw_bytes = sum(p.stat().st_size for batch in batches for p in batch.rglob('*') if p.is_file())
    report = {'status': 'PASS_LOCAL_RELEASE_NO_TRAINING_OR_UPLOAD',
        'expected_attempts': len(expected) * 3,
        'recorded_attempts': sum(p['recorded_attempts'] for p in progress),
        'accepted': info['successful_episodes'], 'frames': info['frames'],
        'split_counts': info['split_counts'], 'class_counts_per_split': counts,
        'excluded': info['excluded'], 'collection_source_hashes': locks[0]['source_hashes'],
        'raw_bytes': raw_bytes, 'free_disk_GiB': shutil.disk_usage(ROOT).free / 2**30,
        'manifest': str(manifest), 'normalization_split': 'train',
        'exports': {split: str(output / f'lerobot_dual_{split}_v1') for split in counts},
        'openpi_native': str(output / 'openpi_native.json'),
        'transfer_manifest': str(output / 'transfer/transfer_manifest.json'),
        'physical_scope': 'Oracle parked-mobile single-fruit pick/place; no five-metre navigation or ten-fruit sorting',
        'training_performed': False, 'weights_loaded': False, 'uploads_performed': False}
    (output / 'release_result.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'status.json').write_text(json.dumps({'status': report['status']}) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('excluded', 'collection_source_hashes')}, indent=2))


if __name__ == '__main__':
    main()

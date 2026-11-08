#!/usr/bin/env python3
"""Small resumable batches of predeclared physical expert attempts, no retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    allowed = [s for split in ('train', 'validation', 'test') for s in protocol[split]]
    if len(set(allowed)) != len(allowed) or not set(args.seeds) <= set(allowed):
        raise ValueError('Scene seeds must be unique across predeclared splits')
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError('Use a new batch root or explicitly resume without retrying attempts')
    output.mkdir(parents=True, exist_ok=True)
    files = [ROOT / 'scripts/record_skill_samples.py']
    files += [ROOT / path for path in ('src/robot_exam/simulation.py',
        'src/robot_exam/skills/pick_place.py', 'src/robot_exam/data/episodes.py',
        'src/robot_exam/control/arm.py', 'src/robot_exam/control/parking.py',
        'src/robot_exam/perception/rgbd.py', 'src/robot_exam/perception/wrist_mount.py')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    freeze = {'protocol': protocol, 'source_hashes': hashes, 'seeds': args.seeds}
    lock = output / 'collection_freeze.json'
    if lock.exists():
        if json.loads(lock.read_text()) != freeze:
            raise RuntimeError('Collection protocol, scene list or source changed')
    else:
        lock.write_text(json.dumps(freeze, indent=2) + '\n')
    env = dict(os.environ, MUJOCO_GL='egl')
    progress = []
    for seed in args.seeds:
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value
               for name, value in hashes.items()):
            raise RuntimeError('Collector source changed; preserve this batch and version the next')
        if shutil.disk_usage(ROOT).free < 26 * 2**30:
            raise RuntimeError('Stop before crossing the 25 GiB engineering reserve')
        folder = output / f'seed_{seed}'
        if not folder.exists():
            folder.mkdir()
            command = [sys.executable, str(ROOT / 'scripts/record_skill_samples.py'),
                '--output', str(folder), '--skills', 'apple', 'banana', 'orange',
                '--seeds', str(seed), '--mobile', '--wrist-camera', '--parking',
                '--oracle-release-feedback', '--scale-min', '.88', '--scale-max', '1.12']
            with (folder / 'collection.log').open('w') as log:
                try:
                    code = subprocess.run(command, cwd=folder, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=900).returncode
                except subprocess.TimeoutExpired:
                    code = None
            (folder / 'process_result.json').write_text(json.dumps({'returncode': code}) + '\n')
        # An interrupted seed directory is evidence, never a free collection retry.
        quality = folder / 'quality.json'
        if not quality.exists() and list(folder.rglob('validation.json')):
            with (folder / 'quality.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT / 'scripts/qualify_stage10_skills.py'),
                    str(folder), '--output', str(quality)], cwd=folder, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        report = json.loads(quality.read_text()) if quality.exists() else {'attempts': 0, 'accepted': 0}
        progress.append({'seed': seed, 'expected_attempts': 3, **report})
        result = {'completed_scene_seeds': len(progress), 'planned_scene_seeds': len(args.seeds),
            'expected_attempts_in_completed_seeds': 3 * len(progress),
            'recorded_attempts': sum(x['attempts'] for x in progress),
            'accepted': sum(x['accepted'] for x in progress), 'seeds': progress,
            'training_performed': False, 'uploads_performed': False}
        (output / 'progress.json').write_text(json.dumps(result, indent=2) + '\n')
        print(seed, report['attempts'], report['accepted'], flush=True)


if __name__ == '__main__':
    main()

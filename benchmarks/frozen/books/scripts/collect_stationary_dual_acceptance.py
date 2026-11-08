#!/usr/bin/env python3
"""Collect real trial artifacts without pooling different evaluation protocols."""
import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from _bootstrap import ROOT


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formal(folder, snapshot):
    root = ROOT / 'outputs/stage11' / folder
    freeze = json.loads((root / 'freeze.json').read_text())
    records = json.loads((root / 'episodes.json').read_text()) if (root / 'episodes.json').exists() else []
    summary = json.loads((root / 'summary.json').read_text()) if (root / 'summary.json').exists() else None
    mismatches = [name for name, expected in freeze['source_hashes'].items()
                  if not (snapshot / name).exists() or sha(snapshot / name) != expected]
    complete = [r['seed'] for r in records] == freeze['protocol']['seeds']
    return {'complete': complete, 'git_commit': freeze['git_commit'],
            'protocol': freeze['protocol'], 'source_hash_mismatches': mismatches,
            'summary': summary, 'episodes': [
                {key: row.get(key) for key in ('seed', 'success', 'completed', 'sim_seconds',
                  'wall_seconds', 'failure_stage', 'root_failure', 'error', 'num_retries',
                  'warnings', 'collision_events', 'solver_linesearch_messages', 'evidence')}
                for row in records], 'raw_results': str(root.relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve historical acceptance snapshots')
    fruit = formal('fruit_formal_baseline_v1', ROOT / 'outputs/stage11/frozen_eval_d2b07d4')
    books = formal('books_formal_parked_dual_v1', ROOT / 'outputs/stage11/frozen_books_d523166')
    if not args.allow_incomplete and not (fruit['complete'] and books['complete']):
        raise RuntimeError('Wait for both predeclared ten-trial evaluations')
    log = ROOT / 'outputs/stage11/pytest_stationary_dual_withdrawal_final_v1.log'
    match = re.search(r'(\d+) passed.*?in ([\d.]+)s', log.read_text())
    if not match:
        raise RuntimeError('Actual final pytest result is missing or incomplete')
    manifest = read('outputs/stage11/dual_pilot_manifest_v1/manifest.json')
    visibility = read('outputs/stage11/dual_visibility_audit_v1.json')
    dev = {}
    for name in ('books_full_seed1011_v1', 'books_parked_dual_seed1011_v9',
                 'books_parked_dual_seed1011_feedback_v1', 'books_parked_dual_seed1012_feedback_v1',
                 'books_parked_dual_seed1013_feedback_v1', 'books_parked_dual_seed1013_withdrawal_v1',
                 'books_parked_dual_seed1011_withdrawal_v2',
                 'fruit_parked_dual_seed1401_dev_v1'):
        path = ROOT / 'outputs/stage11' / name / 'result.json'
        if path.exists():
            result = json.loads(path.read_text())
            dev[name] = {k: result.get(k) for k in ('success', 'completed', 'sim_seconds',
                'wall_seconds', 'failure_stage', 'failure_category', 'error', 'score', 'warnings')}
            dev[name]['path'] = str(path.relative_to(ROOT))
    videos = {}
    for name in ('books_full_seed1011_recorded_v1', 'books_parked_dual_seed1011_recorded_v1'):
        result = read(f'outputs/stage11/{name}/result.json')
        path = ROOT / f'outputs/stage11/{name}/video/demo.mp4'
        videos[name] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                        'bytes': path.stat().st_size, 'physical_success': result['success'],
                        'validation': result['video']}
    disk = shutil.disk_usage(ROOT)
    report = {'timestamp': datetime.datetime.now().astimezone().isoformat(),
        'status': 'local_evaluation_complete_reliability_limits_remain' if fruit['complete'] and books['complete'] else 'IN_PROGRESS',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).splitlines(),
        'pytest': {'passed': int(match[1]), 'seconds': float(match[2]), 'log': str(log.relative_to(ROOT))},
        'formal_fruit': fruit, 'formal_books': books,
        'old_door_formal': {'successes': 7, 'attempts': 10, 'mode': 'Oracle multistart_v2',
                            'evidence': 'outputs/stage8/door_formal_multistart_v2/summary.json',
                            'rerun_this_turn': False},
        'development_results_not_formal': dev, 'videos': videos,
        'dual_view_data': {'accepted': manifest['successful_episodes'], 'frames': manifest['frames'],
            'split_counts': manifest['split_counts'], 'manifest': 'outputs/stage11/dual_pilot_manifest_v1/manifest.json',
            'quality': read('outputs/stage11/dual_pilot_quality_v1.json'),
            'visibility_sampled_episodes': visibility['episodes'],
            'visible_wrist_target_episodes': visibility['episodes_with_visible_wrist_target'],
            'scope': 'Oracle parked-mobile short skills, not ten-fruit navigation episodes'},
        'local_transfer_verification': read('outputs/stage11/autodl_transfer_local_verification_v2.json'),
        'free_disk_GiB': disk.free / 2**30,
        'model_weights_downloaded': False, 'external_uploads': False,
        'paid_compute': False, 'pi05_training_or_inference_performed': False,
        'limitations': ['Formal outcomes must not be used to tune their exposed seeds.',
            'Eleven dual-view episodes demonstrate data plumbing, not learning sufficiency.',
            'Parking during manipulation is not terminal docking success.',
            'Current EGL uses Mesa llvmpipe; GUI and real-model deployment not validated this turn.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'git_commit', 'pytest', 'free_disk_GiB')}, indent=2))


if __name__ == '__main__':
    main()

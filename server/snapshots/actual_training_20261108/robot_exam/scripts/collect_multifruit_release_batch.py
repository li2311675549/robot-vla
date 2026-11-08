#!/usr/bin/env python3
"""Frozen, storage-bounded data collection using the existing physical expert.

Qualification batches and bulk batches use separate plans. Bulk requires an
actual qualification receipt. Every attempted scene/category remains indexed;
only physically successful, dual-rate-replayed parents are positive candidates.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from _bootstrap import ROOT
from robot_exam.data.multifruit_plan import collection_registry


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Preserve all prior attempts; use a new output path')
    if plan['kind'] == 'bulk':
        receipt = Path(plan['qualification_receipt'])
        if digest(receipt) != plan['qualification_sha256'] or json.loads(receipt.read_text())['status'] != 'PASS':
            raise ValueError('Bulk collection requires completed physical/format/coverage qualification')
    elif plan['kind'] != 'qualification' or len(plan['scenes']) > 6:
        raise ValueError('Qualification is bounded to six scene contexts')
    registry = {s['seed']: s for s in collection_registry(json.loads(
        (ROOT / 'configs/stage20_multifruit_dataset_expansion_proposal_v1.json').read_text()))['collection']}
    for scene in plan['scenes']:
        expected = registry[scene['seed']]
        if scene['split'] != expected['split'] or scene['object_count'] != expected['object_count']:
            raise ValueError('Plan must preserve pre-existing split and clutter allocation')
    if len({x['seed'] for x in plan['scenes']}) != len(plan['scenes']):
        raise ValueError('Repeated contexts do not expand independent scene coverage')
    out.mkdir(parents=True)
    source = list((ROOT / 'src/robot_exam').rglob('*.py')) + [
        ROOT / 'scripts' / name for name in ('collect_multifruit_parent.py',
        'collect_multifruit_split_parent.py', 'replay_multifruit_phase.py',
        'calibrate_fruit_rest_pose.py', 'build_tasks.py')]
    source += [Path(__file__).resolve(), args.plan.resolve(), Path(plan['expert_config'])]
    frozen = {str(p): digest(p) for p in source}
    (out / 'source_freeze.json').write_text(json.dumps(frozen, indent=2))
    state = {'status': 'RUNNING', 'purpose': 'Parked multiobject expert data; not full-task or VLA evaluation',
             'plan': str(args.plan.resolve()), 'attempts': [], 'scene_preparations': [],
             'all_attempts_retained': True, 'training': False, 'upload': False}
    lock = threading.Lock()
    stop = threading.Event()
    start_free = shutil.disk_usage(out).free

    def save():
        with lock:
            temp = out / 'progress.tmp'
            temp.write_text(json.dumps(state, indent=2))
            temp.replace(out / 'progress.json')

    def storage_ok():
        free = shutil.disk_usage(out).free
        return free >= plan['minimum_free_gib'] * 2**30 and start_free - free <= plan['maximum_added_gib'] * 2**30

    def unchanged():
        return all(digest(path) == sha for path, sha in frozen.items())

    def collect_scene(scene):
        if stop.is_set():
            return
        if not storage_ok() or not unchanged():
            stop.set()
            return
        from build_tasks import build
        from calibrate_fruit_rest_pose import calibrate
        prep = out / f'seed_{scene["seed"]}' / 'preparation'
        prep.mkdir(parents=True)
        reference, _ = build('fruit', scene['seed'], prep / 'reference.xml', wrist_camera=True,
                             layout_profile='bounded_scatter_v3', bin_label_profile='upright_v2',
                             fruit_count=scene['object_count'])
        calibration = calibrate(reference, prep / 'calibration', seconds=600.)
        with lock:
            state['scene_preparations'].append({'seed': scene['seed'], 'split': scene['split'],
                                               'status': calibration['status']})
        save()
        if calibration['status'] != 'PASS_ALL_EXACT_FRUITS_STABLE':
            return
        settings = json.loads(Path(plan['expert_config']).read_text())
        settings.update(fruit_spawn_profile='passive_rest_init_v2',
                        fruit_rest_calibration=str(prep / 'calibration/calibration.json'),
                        require_non_target_scene_preservation=True)
        config_path = prep / 'settings.json'
        config_path.write_text(json.dumps(settings, indent=2))
        for category in scene['categories']:
            if stop.is_set() or not storage_ok() or not unchanged():
                stop.set()
                return
            folder = prep.parent / category
            folder.mkdir()
            raw = folder / 'raw'
            command = [sys.executable, str(ROOT / 'scripts/collect_multifruit_split_parent.py'),
                       '--output', str(raw), '--config', str(config_path), '--seed', str(scene['seed']),
                       '--category', category, '--fruit-count', str(scene['object_count']), '--split', scene['split']]
            started = time.monotonic()
            row = {'seed': scene['seed'], 'split': scene['split'], 'category': category,
                   'scene_object_count': scene['object_count'], 'path': str(raw), 'replays': []}
            with (folder / 'capture.log').open('x') as log:
                try:
                    cp = subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, timeout=1050)
                    row['returncode'] = cp.returncode
                except subprocess.TimeoutExpired:
                    row['capture_timeout'] = True
            result_path = raw / 'result.json'
            physical = json.loads(result_path.read_text()) if result_path.exists() else {}
            row.update(physical_status=physical.get('status', 'INFRASTRUCTURE_FAILURE'),
                       error=physical.get('error'), failure_stage=physical.get('failure_stage'),
                       non_target_preserved=physical.get('scene_preservation', {}).get('success'),
                       source_changed=not unchanged() or physical.get('source_changed_during_attempt', False),
                       wall_seconds=time.monotonic() - started)
            if row['physical_status'] == 'PHYSICAL_PARENT_PASS_PENDING_REPLAY_AND_EXPORT' and not row['source_changed']:
                for phase in ('grasp', 'place'):
                    dest = raw / phase / 'action_chunk_replay_v2.json'
                    with (raw / phase / 'action_chunk_replay_v2.log').open('x') as log:
                        try:
                            subprocess.run([sys.executable, str(ROOT / 'scripts/replay_multifruit_phase.py'),
                                            str(raw / phase), '--output', str(dest)],
                                           cwd=raw / phase, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                        except subprocess.TimeoutExpired:
                            pass
                    row['replays'].append(json.loads(dest.read_text())['status'] if dest.exists() else 'INFRASTRUCTURE_FAILURE')
            logs = list(folder.rglob('MUJOCO_LOG*'))
            row['text_warning_count'] = sum(p.read_text(errors='replace').count('WARNING') for p in logs)
            row['quality'] = ('PHYSICAL_REPLAY_CANDIDATE_PENDING_FORMAT_AND_IMAGE_QA'
                              if row['replays'] == ['PASS', 'PASS'] and row['text_warning_count'] == 0
                              and row['non_target_preserved'] else 'NOT_ADMITTED')
            with lock:
                state['attempts'].append(row)
            save()
            print(scene['seed'], category, row['physical_status'], row['replays'], flush=True)

    save()
    with ThreadPoolExecutor(max_workers=min(4, plan.get('workers', 1))) as pool:
        futures = [pool.submit(collect_scene, scene) for scene in plan['scenes']]
        for future in futures:
            future.result()
    state.update(status='STOPPED_STORAGE_OR_SOURCE_GATE' if stop.is_set() else 'COMPLETE_ALL_ATTEMPTS_RETAINED',
                 source_unchanged=unchanged(), final_free_gib=shutil.disk_usage(out).free / 2**30)
    save()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Continue an explicitly stopped dispatcher with bounded parallel fruit runs.

Previously started child simulations finish normally. No task physics, policy,
camera input, score or timeout is changed; only independent-process scheduling.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from _bootstrap import ROOT
from collect_task_video_showcase import admit, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--wait-episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Do not replay an existing collection')
    previous = json.loads((args.previous / 'progress.json').read_text())
    if any(previous['groups'][t]['status'] != 'QUOTA_MET' for t in ('door', 'drawer')):
        raise ValueError('Only the remaining fruit quota may be rescheduled')
    freeze = json.loads((args.previous / 'source_freeze.json').read_text())
    if any(digest(Path(p)) != h for p, h in freeze.items()):
        raise ValueError('The physical recording source/config changed')
    plan = json.loads(args.plan.read_text())['task_groups']['fruit']
    out.mkdir(parents=True)
    previous.update(status='RUNNING', continuation='Same physical recordings, scheduling-only continuation',
                    previous_progress=str(args.previous.resolve() / 'progress.json'))
    state = previous
    group = state['groups']['fruit']
    group['status'] = 'RUNNING'
    lock = threading.Lock()

    def save():
        temp = out / 'progress.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / 'progress.json')

    def counts():
        rows = group['records']
        return sum(r['success'] for r in rows), sum(not r['success'] for r in rows)

    def add(path, origin):
        try:
            record = admit(path, 'fruit')
            record['origin'] = origin
            with lock:
                if any(r['video_sha256'] == record['video_sha256'] for r in group['records']):
                    raise ValueError('Identical bytes not counted as another recording')
                group['records'].append(record)
                save()
        except Exception as error:
            with lock:
                group['excluded'].append({'episode': str(path), 'reason': str(error), 'origin': origin})
                save()

    def finish_existing():
        started = time.monotonic()
        while not (args.wait_episode / 'result.json').exists() and time.monotonic() - started < 3900:
            time.sleep(3)
        add(args.wait_episode, 'Already running child from stopped dispatcher; allowed to finish')

    def execute(index, job):
        if shutil.disk_usage(out).free < 26 * 2**30:
            raise RuntimeError('Storage safety gate')
        if any(digest(Path(p)) != h for p, h in freeze.items()):
            raise RuntimeError('Frozen physical recording source changed')
        folder = out / 'fruit' / f'attempt_{index:03d}_seed{job["seed"]}'
        folder.parent.mkdir(exist_ok=True)
        log = folder.with_suffix('.log')
        started = time.monotonic()
        command = [sys.executable, str(ROOT / 'scripts/record_task_view.py'), '--task', 'fruit',
                   '--seed', str(job['seed']), '--mode', job['mode'], '--config', job['config'],
                   '--output', str(folder), '--view-azimuth', str(job['angle'])]
        with log.open('x') as stream:
            try:
                code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=3900).returncode
            except subprocess.TimeoutExpired:
                code = None
        with lock:
            group['new_attempts'].append({'index': index, 'job': job, 'episode': str(folder),
                'returncode': code, 'wall_seconds': time.monotonic() - started, 'log': str(log),
                'continuation': True})
            save()
        add(folder, 'Fresh full physical run; repeated scene explicitly declared')
        print('fruit', index, counts(), flush=True)

    used_angles = {r.get('recording_view', {}).get('view_azimuth_degrees')
                   for r in group['records'] if r.get('recording_view')}
    wait_index = int(args.wait_episode.name.split('_')[1])
    used_angles.add(plan['jobs'][wait_index]['angle'])
    jobs = iter((i, job) for i, job in enumerate(plan['jobs']) if job['angle'] not in used_angles)
    save()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(finish_existing)}
        exhausted = False
        while pending or not exhausted:
            # Three new physical runs plus at most the existing child. Count
            # in-flight runs against the quota to avoid unnecessary repeats.
            while len(pending) < 3 and counts()[0] + len(pending) < plan['success_goal'] and not exhausted:
                try:
                    index, job = next(jobs)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(pool.submit(execute, index, job))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            if counts()[0] >= plan['success_goal']:
                exhausted = True
    success, failure = counts()
    group['status'] = 'QUOTA_MET' if success >= plan['success_goal'] and failure >= plan['failure_goal'] else 'QUOTA_INCOMPLETE_ALL_ATTEMPTS_RETAINED'
    group['selected'] = [r for r in group['records'] if r['success']][:plan['success_goal']]
    group['selected'] += [r for r in group['records'] if not r['success']][:plan['failure_goal']]
    group['counts'] = {'valid_success_recordings': success, 'valid_failure_recordings': failure}
    state['status'] = 'ALL_THREE_QUOTAS_MET' if all(g['status'] == 'QUOTA_MET' for g in state['groups'].values()) else 'INCOMPLETE_WITH_EVIDENCE'
    save()


if __name__ == '__main__':
    main()

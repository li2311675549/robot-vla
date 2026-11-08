#!/usr/bin/env python3
"""Bounded, resumable hardware observation views of existing physical parents."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from _bootstrap import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--camera-near-m', type=float, default=.005)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError('Keep completed and failed view batches; use a new output root')
    out.mkdir(parents=True)
    paths = [ROOT / x for x in ('src/robot_exam/data/episodes.py', 'src/robot_exam/control/arm.py',
                               'scripts/rebuild_episode_hardware_view.py', 'scripts/replay_multifruit_phase.py')]
    fingerprint = lambda: {str(x.relative_to(ROOT)): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    frozen = fingerprint()
    progress = {'status': 'RUNNING', 'new_parents': 0, 'new_scenes': 0,
                'scope': 'replayed observations of the same successful physical operations',
                'camera_near_m': a.camera_near_m, 'source_sha256': frozen, 'rows': []}
    save = lambda: (out / 'progress.json').write_text(json.dumps(progress, indent=2))
    save()
    parents = json.loads(a.parents.read_text())['parents']
    for parent_name in parents:
        parent = Path(parent_name).resolve()
        result = json.loads((parent / 'result.json').read_text())
        identity = f'multifruit_{result["seed"]}_{result["category"]}'
        for phase in ('grasp', 'place'):
            if fingerprint() != frozen:
                raise RuntimeError('Do not mix replay code revisions within an observation batch')
            if shutil.disk_usage(out).free < 25.5 * 1024**3:
                raise RuntimeError('Preserve25GiB free-space safety budget')
            folder = out / identity / phase
            logdir = out / 'run_logs' / f'{identity}_{phase}'
            logdir.mkdir(parents=True)
            cmd = [sys.executable, str(ROOT / 'scripts/rebuild_episode_hardware_view.py'),
                   '--source', str(parent / phase), '--output', str(folder),
                   '--camera-near-m', str(a.camera_near_m)]
            with (logdir / 'reconstruction.log').open('x') as log:
                cp = subprocess.run(cmd, cwd=logdir, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            reconstruction = json.loads((folder / 'hardware_reconstruction.json').read_text()) if (folder / 'hardware_reconstruction.json').exists() else {}
            row = {'parent_id': identity, 'phase': phase, 'source': str(parent / phase),
                   'path': str(folder), 'reconstruction_status': reconstruction.get('status', 'ERROR'),
                   'max_qpos_difference': reconstruction.get('maximum_qpos_difference')}
            if cp.returncode == 0 and reconstruction.get('status') == 'PASS':
                with (logdir / '5hz.log').open('x') as log:
                    cp = subprocess.run([sys.executable, str(ROOT / 'scripts/replay_multifruit_phase.py'),
                                         str(folder), '--output', str(folder / 'action_chunk_replay_v2.json')],
                                        cwd=logdir, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                low = json.loads((folder / 'action_chunk_replay_v2.json').read_text()) if (folder / 'action_chunk_replay_v2.json').exists() else {}
                row['5hz_status'] = low.get('status', 'ERROR')
            row['text_warning_logs'] = [str(x) for x in logdir.glob('MUJOCO_LOG*')]
            progress['rows'].append(row)
            save()
            if row['reconstruction_status'] != 'PASS' or row.get('5hz_status') != 'PASS':
                progress['status'] = 'STOPPED_FAILED_VIEW_RETAINED'
                save()
                return
            print(identity, phase, 'PASS', flush=True)
    progress['status'] = 'PASS_ALL_SAME_PARENT_VIEWS'
    progress['physical_parent_count_not_new'] = len(parents)
    progress['phases'] = len(progress['rows'])
    save()


if __name__ == '__main__':
    main()

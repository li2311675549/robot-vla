#!/usr/bin/env python3
"""Strict paired dock trials reusing original physical/controller snapshots."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from benchmark_dock_stage7 import fixture, trial


@contextmanager
def isolated_log_directory(path):
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT / 'outputs/stage7/dock_pairs_v2_strict')
    p.add_argument('--cases', type=int, default=10)
    p.add_argument('--independent-cases', type=int, default=3)
    p.add_argument('--planner', choices=['direct','preapproach'],default='direct')
    p.add_argument('--kinds',nargs='+',choices=['empty','loaded'],default=['empty','loaded'])
    p.add_argument('--comparison',choices=['capture','angular_cap'],default='capture')
    p.add_argument('--independent-start',type=int)
    a = p.parse_args()
    out, source = a.output.resolve(), a.source.resolve()
    if out.exists():
        raise FileExistsError('Keep all prior dock trials')
    out.mkdir(parents=True)
    common = {'local_dock_strategy': a.planner, 'local_dock_tracking_version': 'axis_bounded_v5',
              'payload_precision_twist_slew': True, 'payload_twist_slew_categories': ['apple','banana','orange']}
    record = {'status': 'RUNNING', 'protocol': 'same integration/controller snapshots;23mm/.035rad/.02mps/.03radps/2s plus independent2s',
              'timeout_sim_seconds': 25., 'planner':a.planner,'comparison':a.comparison,
              'angular_cap_comparison_note':'.45 vs existing steer request.85 rad/s; identical model/force/friction limits; capture enabled for both' if a.comparison=='angular_cap' else None,
              'source': str(source), 'rows': [], 'independent_rows': []}
    save = lambda: (out / 'summary.json').write_text(json.dumps(record, indent=2))
    save()
    for kind in a.kinds:
        for index in range(a.cases):
            original = source / kind / f'case_{index:02d}'
            saved = np.load(original / 'initial_state.npz')
            snapshot = json.loads((original / 'fixture.json').read_text())
            snapshot.update({k: saved[k] for k in saved.files})
            snapshot['state_spec'] = int(snapshot['state_spec'])
            digest = hashlib.sha256((original / 'initial_state.npz').read_bytes()).hexdigest()
            for capture in (False, True):
                folder = out / kind / f'case_{index:02d}' / ('capture' if capture else 'baseline')
                options={**common,'local_dock_capture_goal_during_primitives':capture or a.comparison=='angular_cap'}
                if a.comparison=='angular_cap':options['payload_dock_angular_speed_rad_s']=.85 if capture else .45
                with isolated_log_directory(folder):
                    result = trial(original / 'scene.xml', snapshot, 'capture' if capture else 'baseline',
                                   folder, kind, timeout=25.,
                                   config_overrides=options)
                result.update(case_index=index, same_initial_state_sha256=digest,controller_config=options)
                record['rows'].append(result)
                save()
                print(kind, index, result['strategy'], result['strict_success_after_2s'], flush=True)
        first_independent=a.cases if a.independent_start is None else a.independent_start
        for index in range(first_independent, first_independent + a.independent_cases):
            folder = out / 'independent' / kind / f'case_{index:02d}'
            try:
                with isolated_log_directory(folder):
                    scene, snapshot = fixture(index, kind, folder / 'fixture')
                    options={**common,'local_dock_capture_goal_during_primitives':True}
                    if a.comparison=='angular_cap':options['payload_dock_angular_speed_rad_s']=.85
                    result = trial(scene, snapshot, 'capture', folder / 'capture', kind, timeout=25.,config_overrides=options)
                result['case_index'] = index
            except RuntimeError as exc:
                result = {'kind': kind, 'case_index': index, 'strict_success_after_2s': False,
                          'fixture_or_trial_error': str(exc)}
            record['independent_rows'].append(result)
            save()
    record['status'] = 'COMPLETE_ALL_ATTEMPTS_RETAINED'
    record['counts'] = {f'{kind}/{strategy}': {'attempts': len(rows), 'strict_pass': sum(x['strict_success_after_2s'] for x in rows)}
                        for kind in a.kinds for strategy in ('baseline', 'capture')
                        for rows in [[r for r in record['rows'] if r['kind'] == kind and r['strategy'] == strategy]]}
    save()


if __name__ == '__main__':
    main()

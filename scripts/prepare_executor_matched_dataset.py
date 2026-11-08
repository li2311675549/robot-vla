#!/usr/bin/env python3
"""Prepare the six-scene deployment-matched diagnostic on a no-card server.

Copies only six already-exported episodes; references the existing independent
validation/test views. Never copies weights, imports a model or starts training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def load(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--payload', type=Path, required=True)
    p.add_argument('--base-prepared', type=Path, required=True)
    p.add_argument('--original-assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve prior diagnostic datasets')
    if shutil.disk_usage(a.output.parent).free < 26 * 1024**3:
        raise RuntimeError('Keep25GiB reserve plus preparation headroom')
    identity = load(a.payload / 'payload_manifest.json')
    for relative, expected in identity['files'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe payload member')
        file = a.payload / path
        if file.is_symlink() or digest(file) != expected:
            raise ValueError(f'Payload identity mismatch: {relative}')
    manifest = load(a.payload / 'fit_manifest/manifest.json')
    entries = manifest['accepted']
    expected_pairs = {(seed, fruit) for seed in (15001, 15002)
                      for fruit in ('apple', 'banana', 'orange')}
    if len(entries) != 6 or {(e['seed'], e['skill']) for e in entries} != expected_pairs:
        raise ValueError('Only the declared six original training scenes qualify')
    if any(e['split'] != 'train' or e.get('action_execution_contract') !=
           'full_horizon_interpolate_replan5hz_v1' for e in entries):
        raise ValueError('Wrong split or execution contract')
    original_norm = a.original_assets / 'original_ur5/norm_stats.json'
    norm = a.payload / 'fit_manifest/assets/original_ur5/norm_stats.json'
    if load(original_norm) != load(norm):
        raise ValueError('Controlled six-scene fit must keep original121train norm')
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    info = load(a.payload / 'train/meta/info.json')
    assert info['total_episodes'] == 6 and info['total_frames'] == 612 and info['fps'] == 5
    conversion = load(a.payload / 'train/conversion_result.json')
    assert conversion['action_semantics'] == 'ur5_absolute_joint_goal_v1'
    assert len(conversion['sources']) == 6
    assert all(row['action_execution_contract'] == 'full_horizon_interpolate_replan5hz_v1'
               for row in conversion['sources'])
    tasks = {row['task_index']: row['task'] for row in map(json.loads,
             (a.payload / 'train/meta/tasks.jsonl').read_text().splitlines())}
    rows = 0
    for index, entry in enumerate(entries):
        path = a.payload / 'train' / info['data_path'].format(
            episode_chunk=index // info['chunks_size'], episode_index=index)
        table = pq.ParquetFile(path).read(columns=['action', 'observation.state',
            'timestamp', 'episode_index', 'frame_index', 'task_index', 'episode_success'], use_threads=False)
        assert len(table) == 102
        assert table['episode_index'].to_pylist() == [index] * len(table)
        assert table['frame_index'].to_pylist() == list(range(len(table)))
        assert all(table['episode_success'].to_pylist())
        assert np.allclose(np.diff(table['timestamp'].to_pylist()), .2, atol=1e-5)
        assert all(tasks[i] == f"Pick up the {entry['skill']} and place it on the green circular target marker on the table."
                   for i in table['task_index'].to_pylist())
        for column in ('action', 'observation.state'):
            values = np.asarray(table[column].to_pylist())
            assert values.shape == (102, 7) and np.isfinite(values).all()
            if column == 'action':
                assert np.all((values[:, 6] >= 0) & (values[:, 6] <= 1))
        rows += len(table)
    a.output.mkdir(parents=True)
    for directory in ('train', 'fit_manifest'):
        shutil.copytree(a.payload / directory, a.output / directory)
    aliases = {'robot_exam_executor_matched6_train': a.output / 'train'}
    for split in ('validation', 'test'):
        source = a.base_prepared / 'cache/local' / f'robot_exam_green_target_full_{split}'
        if not source.is_dir():
            raise FileNotFoundError(f'Independent {split} missing')
        aliases[f'robot_exam_green_target_full_{split}'] = source.resolve()
    for name, source in aliases.items():
        link = a.output / 'cache/local' / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(source, link.parent), target_is_directory=True)
    report = {'status': 'PASS_PREPARED_SCALAR_AND_HASH_VERIFIED', 'episodes': 6,
        'frames': rows, 'scene_seeds': [15001, 15002], 'new_random_scenes': 0,
        'scope': 'Separate diagnostic alternative; never append as six new independent scenes',
        'norm_source': 'Frozen original121train statistics for controlled comparison',
        'norm_sha256': digest(norm), 'oldnorm_unchanged': True,
        'payload_files_verified': len(identity['files']), 'training_started': False,
        'torch_imported': False, 'jax_imported': False, 'model_loaded': False}
    (a.output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

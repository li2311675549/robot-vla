#!/usr/bin/env python3
"""Verify every completed Parquet payload, then finish an interrupted API load.

No re-encoding, new episode, score change, frame deletion or success relabel is
performed. Missing conversion metadata is reconstructed only after exact
state/action/prompt/RGB-byte parity and an actual LeRobot v2.1 load pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or (args.dataset / 'conversion_result.json').exists():
        raise FileExistsError('Only recover an incomplete export; preserve existing receipts')
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    rows = [row for row in manifest['accepted'] if row['split'] == args.split]
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    tasks = {row['task_index']: row['task'] for row in
             (json.loads(line) for line in (args.dataset / 'meta/tasks.jsonl').read_text().splitlines())}
    assert len(rows) == info['total_episodes'] and info['fps'] == 5
    assert len(list((args.dataset / 'data').rglob('*.parquet'))) == len(rows)
    sources, verified, offset, image_count = [], [], 0, 0
    for index, row in enumerate(rows):
        raw = Path(row['path'])
        frames = [json.loads(line) for line in (raw / 'frames.jsonl').read_text().splitlines()]
        path = args.dataset / info['data_path'].format(episode_chunk=index // info['chunks_size'], episode_index=index)
        table = pq.read_table(path)
        count = len(frames)
        assert len(table) == count
        np.testing.assert_array_equal(table['episode_index'].to_numpy(), np.full(count, index))
        np.testing.assert_array_equal(table['index'].to_numpy(), np.arange(offset, offset + count))
        np.testing.assert_allclose(table['timestamp'].to_numpy(), np.arange(count) / 5, atol=1e-5)
        for column, key in [('action', 'action'), ('observation.state', 'observation.state')]:
            np.testing.assert_array_equal(np.asarray(table[column].to_pylist(), dtype=np.float32),
                                          np.asarray([frame[key] for frame in frames], dtype=np.float32))
        assert np.asarray(table['episode_success'].to_pylist(), dtype=bool).all()
        assert all(tasks[task_index] == frame['task'] for task_index, frame in zip(table['task_index'].to_pylist(), frames))
        for column, key in [('observation.images.front', 'rgb'), ('observation.images.wrist', 'rgb_wrist')]:
            for encoded, frame in zip(table[column].to_pylist(), frames):
                if encoded['bytes'] != (raw / frame[key]).read_bytes():
                    raise ValueError(f'Exported PNG bytes differ from recorded camera: {path}/{key}')
                image_count += 1
        sources.append({'episode': index, 'raw_path': str(raw.resolve()), 'expert_source': row['expert_source'],
                        'seed': row['seed'], 'skill': row['skill'], 'phase': row['phase'], 'split': row['split'],
                        'parent_id': row['parent_id'], 'scene_context_id': row['scene_context_id'],
                        'action_semantics': row['action_semantics'],
                        'action_execution_contract': row['action_execution_contract']})
        verified.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'frames': count})
        offset += count
    assert offset == info['total_frames']
    proof = {'status': 'ALL_PARQUET_BYTES_AND_LABELS_MATCH_RAW_PENDING_API_LOAD',
             'episodes': len(rows), 'frames': offset, 'rgb_images_exact_bytes': image_count,
             'files': verified, 'sources': sources}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    preliminary = args.receipt.with_name(args.receipt.stem + '_payload_verified.json')
    preliminary.write_text(json.dumps(proof, indent=2))
    subprocess.run([sys.executable, str(Path(__file__).with_name('export_lerobot.py')),
                    str(args.manifest.parent), str(args.dataset), '--validate-only', '--repo-id', args.repo_id], check=True)
    path = args.dataset / 'conversion_result.json'
    conversion = json.loads(path.read_text())
    assert conversion['status'] == 'PASS' and conversion['episodes'] == len(rows) and conversion['frames'] == offset
    assert conversion['sample_shapes']['action'] == [7] and conversion['sample_shapes']['observation.images.wrist'] == [3, 480, 640]
    conversion.update(sources=sources, action_semantics='ur5_absolute_joint_goal_v1',
                      sensor_schema='front_wrist_rgb_v1', actual_split=args.split,
                      recovery='Exact raw/Parquet parity; resumed genuine LeRobot API load without re-encoding')
    path.write_text(json.dumps(conversion, indent=2))
    proof.update(status='PASS_RECOVERED_WITH_EXACT_PAYLOAD_AND_REAL_API_LOAD', wall_seconds=time.monotonic() - started,
                 no_frames_or_physical_labels_changed=True)
    args.receipt.write_text(json.dumps(proof, indent=2))
    print(json.dumps({k: v for k, v in proof.items() if k not in ('sources', 'files')}, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Check exact Pi05 observation aliases in the six-scene diagnostic.

No nearest-neighbour tolerance, model prediction or causal claim is used.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--norm', type=Path, required=True)
    p.add_argument('--client-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve prior alias audits')
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    sys.path.insert(0, str(a.client_source))
    from openpi_client.image_tools import resize_with_pad
    norm = json.loads(a.norm.read_text())['norm_stats']['state']
    q01, q99 = np.asarray(norm['q01']), np.asarray(norm['q99'])
    info = json.loads((a.dataset / 'meta/info.json').read_text())
    results = []
    for episode in range(info['total_episodes']):
        path = a.dataset / info['data_path'].format(episode_chunk=episode // info['chunks_size'], episode_index=episode)
        rows = pq.ParquetFile(path).read(columns=['observation.images.front', 'observation.images.wrist',
               'observation.state', 'action', 'task_index'], use_threads=False).to_pylist()
        actions = np.asarray([r['action'] for r in rows])
        groups = {}
        for i, row in enumerate(rows):
            state = np.asarray(row['observation.state'])
            state = (state - q01) / (q99 - q01 + 1e-6) * 2 - 1
            # Native PI05 tokenizes the seven normalized states BEFORE padding.
            bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            h = hashlib.sha256(bins.astype(np.int32).tobytes())
            h.update(str(row['task_index']).encode())
            for camera in ('front', 'wrist'):
                with Image.open(io.BytesIO(row[f'observation.images.{camera}']['bytes'])) as img:
                    pixels = np.asarray(img.convert('RGB'))
                h.update(resize_with_pad(pixels, 224, 224).tobytes())
            groups.setdefault(h.hexdigest(), []).append(i)
        duplicates = []
        for indices in groups.values():
            if len(indices) < 2:
                continue
            future = np.stack([actions[np.minimum(np.arange(i, i+10), len(actions)-1)] for i in indices])
            duplicates.append({'frame_indices': indices,
                'first_action_max_joint_difference_rad': float(np.max(np.ptp(actions[indices, :6], axis=0))),
                'first_action_gripper_difference': float(np.ptp(actions[indices, 6])),
                'future10_max_joint_difference_rad': float(np.max(np.ptp(future[:, :, :6], axis=0))),
                'future10_gripper_difference': float(np.max(np.ptp(future[:, :, 6], axis=0)))})
        results.append({'episode': episode, 'frames': len(rows), 'exact_duplicate_groups': duplicates})
    report = {'status': 'COMPLETED_EXACT_OBSERVATION_ALIAS_AUDIT', 'episodes': results,
        'duplicate_groups': sum(len(r['exact_duplicate_groups']) for r in results),
        'scope': 'Within each episode: exact dual224pixels + discretized7state + task. Equal continuous inputs are not required by PI05. No model inference or proof of failure cause.',
        'norm_sha256': hashlib.sha256(a.norm.read_bytes()).hexdigest(), 'training_started': False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

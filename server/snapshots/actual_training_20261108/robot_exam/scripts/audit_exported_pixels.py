#!/usr/bin/env python3
"""Read-only all-frame RGB/depth/scalar alignment audit, no dataset rewrite."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve prior audits')
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    root = a.preparation_root
    prep = json.loads((root / 'green_target_v2/preparation.json').read_text())
    datasets = [(f'original173/{split}', Path(prep['views'][f'full_{split}']['source']))
                for split in ('train', 'validation', 'test')]
    datasets += [(f'timing12/{split}', root / 'timing_candidates_release_v1' / split)
                 for split in ('train', 'validation', 'test')]
    datasets += [('derived_matched6/train', root / 'executor_matched6_release_v4/train')]
    a.output.mkdir(parents=True)
    report = {'status': 'RUNNING', 'datasets': {}, 'rgb_comparisons': 0, 'depth_files': 0,
              'frames': 0, 'episodes': 0, 'model_loaded': False, 'training_started': False,
              'scope': '185 independent-source episodes plus6 derived scenes; pixel/scalar/depth integrity only'}
    started = time.monotonic()

    def save():
        (a.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')

    try:
        for name, dataset in datasets:
            info = json.loads((dataset / 'meta/info.json').read_text())
            sources = json.loads((dataset / 'conversion_result.json').read_text())['sources']
            checks = []
            for source in sources:
                i = source['episode']
                file = dataset / info['data_path'].format(episode_chunk=i // info['chunks_size'], episode_index=i)
                identity = digest(file)
                raw = Path(source['raw_path'])
                frames = [json.loads(s) for s in (raw / 'frames.jsonl').read_text().splitlines()]
                rows = pq.ParquetFile(file).read(columns=['observation.images.front', 'observation.images.wrist',
                    'observation.state', 'action', 'frame_index', 'timestamp', 'episode_success'], use_threads=False).to_pylist()
                assert len(rows) == len(frames)
                for frame, original in zip(rows, frames, strict=True):
                    report['current'] = {'dataset': name, 'episode': i, 'frame': frame['frame_index']}
                    assert frame['episode_success'] is True
                    assert frame['frame_index'] == original['frame_index']
                    assert abs(frame['timestamp'] - (original['timestamp'] - frames[0]['timestamp'])) < 1e-5
                    for key, label in (('observation.state', 'observation.state'), ('action', 'action')):
                        assert np.array_equal(np.asarray(frame[key], dtype=np.float32), np.asarray(original[label], dtype=np.float32))
                    for camera, key in (('front', 'rgb'), ('wrist', 'rgb_wrist')):
                        with Image.open(io.BytesIO(frame[f'observation.images.{camera}']['bytes'])) as encoded:
                            decoded = np.array(encoded.convert('RGB'))
                        with Image.open(raw / original[key]) as image:
                            assert decoded.shape == (480, 640, 3) and np.array_equal(decoded, np.asarray(image.convert('RGB')))
                        report['rgb_comparisons'] += 1
                    with np.load(raw / original['depth']) as archive:
                        depth = archive['depth_m']
                        assert depth.shape == (480, 640) and np.isfinite(depth).all() and np.all(depth > 0)
                    report['depth_files'] += 1
                    report['frames'] += 1
                assert digest(file) == identity
                report['episodes'] += 1
                checks.append({'episode': i, 'seed': source['seed'], 'skill': source['skill'],
                               'frames': len(rows), 'parquet_sha256': identity})
                report['datasets'][name] = checks
                save()
                print(name, i, 'PASS', flush=True)
        report.update(status='PASS_ALL_RGB_DEPTH_SCALAR_ALIGNMENT', wall_seconds=time.monotonic() - started)
        report.pop('current', None)
    except BaseException as exc:
        report.update(status='FAILED_OR_INTERRUPTED', error={'type': type(exc).__name__, 'message': str(exc)},
                      wall_seconds=time.monotonic() - started)
        save()
        raise
    save()
    print(json.dumps({k: report[k] for k in ('status', 'episodes', 'frames', 'rgb_comparisons', 'depth_files', 'wall_seconds')}))


if __name__ == '__main__':
    main()

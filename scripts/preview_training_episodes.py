#!/usr/bin/env python3
"""Make review MP4s from actual exported dual-view training pixels, offline.

Run in the existing anaconda3/envs/openpi environment (pyarrow/Pillow/numpy).
Never reruns simulation or changes the released dataset. Default selection is
one train episode per fruit; --all exports every episode of the chosen split.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--all', action='store_true')
    p.add_argument('--episodes', nargs='+', type=int)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier reviews; use a new output')
    if shutil.disk_usage(args.dataset).free < 25 * 1024**3 + 256 * 1024**2:
        raise RuntimeError('Keep 25 GiB free plus preview headroom')
    conversion = json.loads((args.dataset / 'conversion_result.json').read_text())
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    sources = conversion['sources']
    if args.all and args.episodes:
        raise ValueError('Choose --all or explicit --episodes')
    if args.episodes:
        chosen = [row for row in sources if row['episode'] in args.episodes]
        if {r['episode'] for r in chosen} != set(args.episodes):
            raise ValueError('Unknown exported episode number')
    elif args.all:
        chosen = sources
    else:
        chosen = [next(r for r in sources if r['skill'] == skill) for skill in ('apple', 'banana', 'orange')]
    args.output.mkdir(parents=True)
    reports = []
    for source in chosen:
        index = source['episode']
        parquet = args.dataset / info['data_path'].format(episode_chunk=index // info['chunks_size'], episode_index=index)
        identity = digest(parquet)
        rows = pq.read_table(parquet, columns=['observation.images.front', 'observation.images.wrist',
                                               'timestamp', 'frame_index', 'episode_success']).to_pylist()
        times = np.asarray([r['timestamp'] for r in rows], dtype=float)
        fps = float(info['fps'])
        if len(rows) < 2 or not np.allclose(np.diff(times), 1 / fps, atol=1e-5):
            raise ValueError('Missing or irregular training frames')
        raw = Path(source['raw_path'])
        originals = [json.loads(line) for line in (raw / 'frames.jsonl').read_text().splitlines()]
        if len(rows) != len(originals):
            raise ValueError('Raw/export frame count differs')
        name = f"{source['skill']}_seed{source['seed']}_episode{index:06d}"
        target = args.output / f'{name}_dual.mp4'
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', '1280x512', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                   '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
        comparisons = 0
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for i, row in enumerate(rows):
                success = row['episode_success']
                if isinstance(success, list):
                    success = len(success) == 1 and success[0] is True
                if row['frame_index'] != i or success is not True:
                    raise ValueError('Invalid frame index or nonpositive training label')
                canvas = Image.new('RGB', (1280, 512), 'black')
                for x, key, raw_key in [(0, 'front', 'rgb'), (640, 'wrist', 'rgb_wrist')]:
                    image = Image.open(io.BytesIO(row[f'observation.images.{key}']['bytes'])).convert('RGB')
                    if image.size != (640, 480):
                        raise ValueError('Unexpected training image shape')
                    with Image.open(raw / originals[i][raw_key]) as original:
                        if not np.array_equal(np.asarray(image), np.asarray(original.convert('RGB'))):
                            raise ValueError('Exported pixels differ from original collection')
                    comparisons += 1
                    canvas.paste(image, (x, 32))
                label = (f"{source['skill']} | seed {source['seed']} | episode {index} | "
                         f"t={times[i]:.1f}s | FRONT (left) / WRIST (right) | actual dataset, {fps:g} Hz")
                ImageDraw.Draw(canvas).text((8, 8), label, fill='white')
                encoder.stdin.write(canvas.tobytes())
                if i == len(rows) // 2:
                    canvas.save(args.output / f'{name}_middle.png')
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Video encoder failed')
        probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                                '-show_entries', 'stream=nb_read_frames,duration,width,height', '-of', 'json', str(target)],
                               check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)['streams'][0]
        if int(stream['nb_read_frames']) != len(rows):
            raise ValueError('Video frame count differs from training episode')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(target), '-f', 'null', '-'], check=True)
        if digest(parquet) != identity:
            raise RuntimeError('Dataset changed during read-only review')
        reports.append({**source, 'dataset_file': str(parquet.resolve()), 'dataset_sha256': identity,
                        'video': str(target.resolve()), 'video_sha256': digest(target), 'frames': len(rows),
                        'fps': fps, 'stream': stream, 'raw_export_pixel_comparisons_passed': comparisons,
                        'full_decode': 'PASS', 'dataset_unchanged': True})
    report = {'status': 'PASS_DATASET_PIXEL_PREVIEWS_ONLY', 'dataset': str(args.dataset.resolve()),
              'new_simulation_performed': False, 'training_performed': False,
              'note': 'Lossy review copies of recorded RGB pixels; not new demonstrations or model predictions. No frames dropped.',
              'episodes': reports}
    (args.output / 'preview_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    links = '\n'.join(f"- [{r['skill']} / seed{r['seed']} / episode{r['episode']}]({Path(r['video']).name})" for r in reports)
    (args.output / 'README.md').write_text('# 训练数据双视角人工检查\n\n左：机身；右：腕部。实际训练图像按原5Hz播放，原图不缩放，顶栏为复查标签。\n\n' + links +
        '\n\nMP4只是有损预览，不替换训练用PNG/Parquet。所有帧已与原采集像素逐一核对并完整解码；没有重跑或拼接仿真任务。\n')
    print(json.dumps({'status': report['status'], 'videos': len(reports), 'output': str(args.output)}))


if __name__ == '__main__':
    main()

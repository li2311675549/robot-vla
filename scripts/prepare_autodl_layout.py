#!/usr/bin/env python3
"""Link already copied exports into a local cache. No network or environment install."""
import argparse
import json
from pathlib import Path
import shutil


def prepare_layout(root, cache):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    manifest = json.loads((root/'dataset_manifest/manifest.json').read_text())
    accepted = manifest['accepted']
    seen = {}
    results = {}
    for split in ('train', 'validation', 'test'):
        directory = root/'datasets'/split
        info = json.loads((directory/'meta/info.json').read_text())
        episodes = [json.loads(line) for line in (directory/'meta/episodes.jsonl').read_text().splitlines() if line]
        entries = [row for row in accepted if row['split'] == split]
        if len(entries) != len(episodes) or sum(x['length'] for x in episodes) != info['total_frames']:
            raise ValueError(f'Manifest/export episode or frame mismatch: {split}')
        if info['fps'] != 5 or info['features']['action']['shape'] != [7]:
            raise ValueError('Only Schema3 5Hz/7D exports supported')
        for name in ('front', 'wrist'):
            if f'observation.images.{name}' not in info['features']:
                raise ValueError('Two real camera exports required')
        for entry in entries:
            seed = entry['seed']
            if seed in seen and seen[seed] != split:
                raise ValueError('Scene seed leakage across splits')
            seen[seed] = split
        destination = cache/'local'/f'robot_exam_dual_release_{split}'
        if destination.is_symlink():
            if destination.resolve() != directory.resolve():
                raise FileExistsError(f'Refusing to redirect existing cache: {destination}')
        elif destination.exists():
            raise FileExistsError(f'Refusing to replace existing directory: {destination}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(directory, target_is_directory=True)
        results[split] = {'episodes': len(entries), 'frames': info['total_frames'],
                          'repo_id': f'local/robot_exam_dual_release_{split}'}
    return {'status': 'PASS_LAYOUT_ONLY', 'splits': results, 'scene_split_isolation': True,
            'free_disk_GiB': shutil.disk_usage(root).free/1024**3,
            'training_performed': False, 'network_used': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit output')
    report = prepare_layout(args.root, args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

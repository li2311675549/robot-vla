#!/usr/bin/env python3
"""Standard-library-only integrity check for the portable dataset release."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def verify(root, output):
    root = root.resolve()
    if output.exists():
        raise FileExistsError('Preserve prior verification evidence')
    started = time.monotonic()
    receipt = json.loads((root / '数据交付校验.json').read_text())
    cache, paths = {}, set()
    for row in receipt['files']:
        path = root / row['path']
        if row['path'] in paths or not path.resolve().is_relative_to(root):
            raise ValueError('Duplicate or nonportable payload path')
        paths.add(row['path'])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing payload or unresolved symlink: {path}')
        stat = path.stat()
        if stat.st_size != row['bytes']:
            raise ValueError(f'Size differs: {path}')
        key = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
        if key not in cache:
            hasher = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(2**20), b''):
                    hasher.update(block)
            cache[key] = hasher.hexdigest()
        if cache[key] != row['sha256']:
            raise ValueError(f'Checksum differs: {path}')
    preparation = json.loads((root / 'mixture/preparation.json').read_text())
    for row in preparation['input_files']:
        path = root / row['path']
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Mixture still references a missing external Parquet')
        hasher = hashlib.sha256(path.read_bytes()).hexdigest()
        if hasher != row['sha256']:
            raise ValueError('Portable mixture Parquet checksum mismatch')
    if (root / 'assets/original_ur5/norm_stats.json').read_bytes() != (root / 'mixture/assets/original_ur5/norm_stats.json').read_bytes():
        raise ValueError('Training normalization assets disagree')
    report = {'status': 'PASS_ALL_PORTABLE_PAYLOAD_HASHES_AND_MIXTURE_PATHS',
              'root': str(root), 'files': len(paths), 'unique_inodes_hashed': len(cache),
              'mixture_parquets': len(preparation['input_files']),
              'old_new_source_mass': preparation['source_mass'],
              'actual_training_frames': preparation['frames'],
              'wall_seconds': time.monotonic() - started,
              'scope': 'Payload integrity/relative paths only; separate physical/native-loader proofs required'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.output), ensure_ascii=False, indent=2))

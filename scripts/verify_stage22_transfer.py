#!/usr/bin/env python3
"""Streaming SHA check for explicitly enumerated payloads; standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def verify(root, manifest):
    root = Path(root).resolve()
    rows = json.loads(Path(manifest).read_text())['files']
    seen = set()
    total = 0
    for row in rows:
        relative = Path(row['path'])
        path = root/relative
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('Unsafe or duplicate manifest entry')
        seen.add(str(relative))
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or external payload: {relative}')
        if path.stat().st_size != row['bytes']:
            raise ValueError(f'Incorrect file size: {relative}')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024**2), b''):
                h.update(chunk)
        if h.hexdigest() != row['sha256']:
            raise ValueError(f'SHA mismatch: {relative}')
        total += row['bytes']
    return {'status': 'PASS_ALL_PAYLOAD_SHA256', 'files': len(rows), 'bytes': total,
            'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            'root': str(root)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep previous transfer verification')
    start = time.monotonic()
    report = {'status': 'FAIL'}
    try:
        report = verify(args.root, args.manifest)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

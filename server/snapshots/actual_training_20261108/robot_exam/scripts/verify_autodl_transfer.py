#!/usr/bin/env python3
"""Verify a local transfer manifest, or the same files after an authorized copy."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--copied-root', type=Path,
                        help='Optional remote layout root; default checks original local files')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite a transfer audit')
    manifest = json.loads(args.manifest.read_text())
    rows = list(manifest['files'])
    for key in ('code_archive', 'openpi_snapshot'):
        if key in manifest:
            item = manifest[key]
            rows.append({**item, 'local_path': item['path'],
                         'destination_relative_path': Path(item['path']).name})
    failures = []
    for row in rows:
        relative = Path(row['destination_relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Destination must remain inside the transfer root')
        path = args.copied_root / relative if args.copied_root else Path(row['local_path'])
        if not path.is_file():
            failures.append({'path': str(path), 'reason': 'missing'})
        elif path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            failures.append({'path': str(path), 'reason': 'size_or_sha256_mismatch'})
    result = {'status': 'FAIL' if failures else 'PASS', 'files_checked': len(rows),
              'failures': failures, 'copied_layout': args.copied_root is not None,
              'network_used': False, 'training_performed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

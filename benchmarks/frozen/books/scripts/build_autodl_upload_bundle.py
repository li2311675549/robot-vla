#!/usr/bin/env python3
"""Materialize a checksum-verified upload layout locally; never upload/download."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transfer-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runbook', type=Path, required=True)
    parser.add_argument('--archive', action='store_true', help='Also create one uncompressed tar for upload')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new upload bundle directory')
    manifest = json.loads(args.transfer_manifest.read_text())
    rows = list(manifest['files'])
    for key in ('code_archive', 'openpi_snapshot'):
        item = manifest[key]
        rows.append({'local_path': item['path'], 'destination_relative_path': Path(item['path']).name,
                     'bytes': item['bytes'], 'sha256': item['sha256']})
    total = sum(row['bytes'] for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A cross-filesystem copy plus optional tar must never consume the safety reserve.
    required = total*(2 if args.archive else 1)+25*1024**3
    if shutil.disk_usage(args.output.parent).free < required:
        raise RuntimeError('Insufficient space for conservative bundle+archive budget plus 25GiB reserve')
    for row in rows:
        relative = Path(row['destination_relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe transfer destination')
        source = Path(row['local_path'])
        if source.stat().st_size != row['bytes'] or sha(source) != row['sha256']:
            raise ValueError(f'Changed input: {source}')
    args.output.mkdir()
    links = copies = 0
    for row in rows:
        target = args.output/row['destination_relative_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(row['local_path'], target)
            links += 1
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(row['local_path'], target)
            copies += 1
    shutil.copy2(args.transfer_manifest, args.output/'transfer_manifest.json')
    shutil.copy2(args.runbook, args.output/'SERVER_START_HERE.md')
    report = {'status': 'LOCAL_UPLOAD_BUNDLE_VERIFIED', 'code_commit': manifest['git_commit'],
              'files': len(rows), 'hardlinks': links, 'copies': copies, 'bytes': total,
              'note': 'Hardlinked exports share local data: do not edit. Tar is independent.',
              'uploads_performed': False, 'weights_included': False, 'training_performed': False}
    if args.archive:
        archive = args.output.with_suffix('.tar')
        if archive.exists():
            raise FileExistsError(archive)
        with tarfile.open(archive, 'w') as stream:
            stream.add(args.output, arcname='robot_exam_training', recursive=True)
        report['archive'] = {'path': str(archive.resolve()), 'bytes': archive.stat().st_size,
                             'sha256': sha(archive)}
        archive.with_suffix('.tar.sha256').write_text(f"{report['archive']['sha256']}  {archive.name}\n")
    (args.output.parent/(args.output.name+'_result.json')).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

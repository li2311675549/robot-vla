#!/usr/bin/env python3
"""Verified lossless storage compaction; never touches scores or demonstrations.

Old upload tar payloads remain in the existing materialized bundle. Preserve
their exact non-payload bytes as a small recipe, and verify reconstruction SHA
before removing a tar. Bulk JSON logs use byte-exact gzip with a restore index.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 4*1024**2
BULK_NAMES = {'trajectory.json', 'payload_traces.json', 'dock.json',
              'ik_diagnostics.json', 'parking.json', 'physical_trace.json'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, obj):
    temp = Path(str(path)+'.pending')
    with temp.open('w') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def stream_recipe(recipe, sink):
    """Feed exact reconstructed archive bytes to a file or a SHA sink."""
    skeleton = Path(recipe['skeleton'])
    if digest(skeleton) != recipe['skeleton_sha256']:
        raise ValueError('Archive skeleton checksum mismatch')
    with skeleton.open('rb') as metadata:
        for row in recipe['segments']:
            if row['kind'] == 'metadata':
                metadata.seek(row['offset'])
                source, remaining = metadata, row['bytes']
            else:
                source = Path(row['source']).open('rb')
                remaining = row['bytes']
            check = hashlib.sha256()
            try:
                while remaining:
                    chunk = source.read(min(BLOCK, remaining))
                    if not chunk:
                        raise ValueError('Truncated reconstruction source')
                    remaining -= len(chunk)
                    check.update(chunk)
                    sink(chunk)
                if row['kind'] == 'file' and check.hexdigest() != row['sha256']:
                    raise ValueError(f'Changed bundle source: {row["source"]}')
            finally:
                if source is not metadata:
                    source.close()


def archive_recipe(archive, bundle, output):
    archive, bundle, output = map(Path, (archive, bundle, output))
    output.mkdir(parents=True, exist_ok=False)
    payloads = []
    with tarfile.open(archive, 'r:') as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if name.parts[0] != 'robot_exam_training' or '..' in name.parts:
                raise ValueError('Unexpected archive member path')
            if not member.isfile() or not member.size:
                continue
            source = bundle.joinpath(*name.parts[1:])
            if source.is_symlink() or not source.is_file() or source.stat().st_size != member.size:
                raise ValueError(f'No byte-equivalent retained source: {source}')
            payloads.append(dict(offset=member.offset_data, bytes=member.size,
                                 source=str(source.resolve()), sha256=digest(source)))
    skeleton = output/'tar_nonpayload_bytes.bin'
    segments = []
    with archive.open('rb') as source, skeleton.open('xb') as target:
        cursor = 0
        for row in payloads:
            gap = row['offset']-cursor
            if gap < 0:
                raise ValueError('Overlapping archive payloads')
            if gap:
                offset = target.tell()
                source.seek(cursor)
                while gap:
                    chunk = source.read(min(BLOCK, gap))
                    if not chunk:
                        raise ValueError('Truncated tar metadata')
                    target.write(chunk); gap -= len(chunk)
                segments.append(dict(kind='metadata', offset=offset, bytes=row['offset']-cursor))
            segments.append(dict(kind='file', **row))
            cursor = row['offset']+row['bytes']
        source.seek(cursor)
        offset = target.tell()
        shutil.copyfileobj(source, target, BLOCK)
        segments.append(dict(kind='metadata', offset=offset, bytes=target.tell()-offset))
        target.flush(); os.fsync(target.fileno())
    recipe = dict(archive=str(archive.resolve()), bytes=archive.stat().st_size,
                  sha256=digest(archive), skeleton=str(skeleton.resolve()),
                  skeleton_sha256=digest(skeleton), segments=segments,
                  retained_bundle=str(bundle.resolve()))
    reconstructed = hashlib.sha256()
    stream_recipe(recipe, reconstructed.update)
    if reconstructed.hexdigest() != recipe['sha256']:
        raise ValueError('Retained bundle cannot reconstruct the original tar byte for byte')
    recipe['streamed_byte_exact_reconstruction'] = 'PASS'
    write_json(output/'recipe.json', recipe)
    return recipe


def compact_log(path):
    path = Path(path)
    before = path.stat()
    if path.is_symlink() or before.st_nlink != 1:
        raise ValueError('Only an independent regular log may be compacted')
    expected = digest(path)
    target = path.with_suffix(path.suffix+'.gz')
    if target.exists():
        return dict(original=str(path.resolve()), status='SKIPPED_EXISTING_GZIP')
    temp = Path(str(target)+'.pending')
    with path.open('rb') as source, temp.open('xb') as output:
        with gzip.GzipFile(filename='', fileobj=output, mode='wb', compresslevel=3, mtime=0) as gz:
            shutil.copyfileobj(source, gz, BLOCK)
        output.flush(); os.fsync(output.fileno())
    check = hashlib.sha256()
    with gzip.open(temp, 'rb') as source:
        for chunk in iter(lambda: source.read(BLOCK), b''):
            check.update(chunk)
    if check.hexdigest() != expected or digest(path) != expected:
        raise ValueError('Lossless verification failed or source log changed')
    row = dict(original=str(path.resolve()), compressed=str(target.resolve()),
               original_bytes=before.st_size, compressed_bytes=temp.stat().st_size,
               sha256=expected, gzip_sha256=digest(temp), mode=stat.S_IMODE(before.st_mode),
               mtime_ns=before.st_mtime_ns, status='VERIFIED_GZIP_READY')
    os.replace(temp, target)
    return row


def restore_log(row):
    path, source = Path(row['original']), Path(row['compressed'])
    if path.exists():
        raise FileExistsError('Never overwrite an existing log')
    if shutil.disk_usage(path.parent).free < row['original_bytes']+25*1024**3:
        raise RuntimeError('Restore would cross the25GiB reserve')
    if digest(source) != row['gzip_sha256']:
        raise ValueError('Compressed log checksum mismatch')
    temp = Path(str(path)+'.restore_pending')
    with gzip.open(source, 'rb') as stream, temp.open('xb') as target:
        shutil.copyfileobj(stream, target, BLOCK)
        target.flush(); os.fsync(target.fileno())
    if digest(temp) != row['sha256']:
        raise ValueError('Restored log checksum mismatch')
    os.chmod(temp, row['mode']); os.utime(temp, ns=(row['mtime_ns'], row['mtime_ns']))
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path)
    p.add_argument('--restore-manifest', type=Path)
    p.add_argument('--restore-log', type=Path)
    p.add_argument('--restore-archive-recipe', type=Path)
    args = p.parse_args()
    if args.restore_manifest:
        manifest = json.loads(args.restore_manifest.read_text())
        original = str(args.restore_log.resolve())
        row = next(r for r in manifest['logs'] if r['original'] == original and r['status'] == 'COMPRESSED_ORIGINAL_REMOVED')
        restore_log(row); print('Restored byte-exact log:', original); return
    if args.restore_archive_recipe:
        recipe = json.loads(args.restore_archive_recipe.read_text())
        path = Path(recipe['archive'])
        if shutil.disk_usage(path.parent).free < recipe['bytes']+25*1024**3:
            raise RuntimeError('Archive restore would cross the25GiB reserve')
        with path.open('xb') as target:
            stream_recipe(recipe, target.write)
            target.flush(); os.fsync(target.fileno())
        if digest(path) != recipe['sha256']:
            raise ValueError('Restored archive checksum mismatch')
        print('Restored byte-exact upload archive:', path); return
    if args.output is None:
        p.error('--output is required for compaction')
    # Avoid racing any simulation/dataset writer. The current process and
    # read-only shell inspection are not matched by these script basenames.
    processes = subprocess.check_output(['ps', '-eo', 'comm,args'], text=True).splitlines()
    for line in processes:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and 'python' in parts[0] and any(name in parts[1] for name in
                ('scripts/run_task.py', 'scripts/collect_multifruit_parent.py', 'pytest', 'scripts/probe_')):
            raise RuntimeError(f'Active project writer: {line}')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    result = dict(status='RUNNING', started_at=time.time(),
                  authorization='User requested local space reclamation; unique evidence retained losslessly',
                  free_bytes_before=shutil.disk_usage(ROOT).free, archives=[], logs=[],
                  unchanged='Scores/results, scene/config files, videos, raw episodes, environments, source and models')
    journal = out/'cleanup_manifest.json'; write_json(journal, result)
    for version in (1, 2):
        folder = ROOT/f'outputs/stage12/server_training_ready_v{version}'
        archive = folder/f'upload_bundle_v{version}.tar'
        expected = json.loads((folder/f'upload_bundle_v{version}_result.json').read_text())['archive']
        if digest(archive) != expected['sha256']:
            raise ValueError('Old upload archive differs from original validation')
        recipe = archive_recipe(archive, archive.with_suffix(''), out/f'archive_v{version}')
        result['archives'].append(dict(path=str(archive), bytes=recipe['bytes'], sha256=recipe['sha256'],
            restore_recipe=str(out/f'archive_v{version}/recipe.json'), status='RECONSTRUCTION_VERIFIED'))
        write_json(journal, result)
        archive.unlink()
        result['archives'][-1]['status'] = 'REDUNDANT_ARCHIVE_REMOVED'
        write_json(journal, result)
        print('Removed verified reconstructable archive:', archive.name, flush=True)
    candidates = sorted(path for path in (ROOT/'outputs').rglob('*.json')
                        if path.name in BULK_NAMES and path.stat().st_size >= 1024**2
                        and not path.is_symlink() and path.stat().st_nlink == 1)
    for i, path in enumerate(candidates):
        row = compact_log(path)
        result['logs'].append(row)
        write_json(journal, result)
        if row['status'] == 'VERIFIED_GZIP_READY':
            path.unlink()
            row['status'] = 'COMPRESSED_ORIGINAL_REMOVED'
            write_json(journal, result)
        if i % 50 == 0:
            print('Verified bulk logs:', i+1, '/', len(candidates), flush=True)
    result.update(status='COMPLETE', finished_at=time.time(), free_bytes_after=shutil.disk_usage(ROOT).free)
    result['observed_free_bytes_increase'] = result['free_bytes_after']-result['free_bytes_before']
    write_json(journal, result)
    print('Observed free GiB increase:', result['observed_free_bytes_increase']/1024**3, flush=True)


if __name__ == '__main__':
    main()

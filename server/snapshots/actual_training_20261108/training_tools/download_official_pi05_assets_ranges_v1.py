#!/usr/bin/env python3
"""Authorized, streaming curl downloads checked against frozen official GCS metadata.

No model is imported. Incomplete files stay as .partial; a directory is never
declared ready just because it exists. Uses curl and google-crc32c, not a model
downloader that may buffer a multi-gigabyte object in memory.
"""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from urllib.parse import quote, urlencode


def asset_relative_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe asset object name')
    return Path(*path.parts)


def verify_file(path, metadata):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != int(metadata['size']):
        raise ValueError(f'Incomplete or missing asset: {path.name}')
    md5 = hashlib.md5() if metadata.get('md5Hash') else None
    crc = None
    if metadata.get('crc32c'):
        import google_crc32c
        crc = google_crc32c.Checksum()
    if md5 is None and crc is None:
        raise ValueError('Official checksum is required')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            if md5 is not None:
                md5.update(block)
            if crc is not None:
                crc.update(block)
    result = {'bytes': path.stat().st_size}
    for key, checksum in (('md5Hash', md5), ('crc32c', crc)):
        if checksum is not None:
            value = base64.b64encode(checksum.digest()).decode()
            if value != metadata[key]:
                raise ValueError(f'Official {key} mismatch: {path.name}')
            result[key] = value
    return result


def validate_range_response(headers, start, end, total, actual_bytes):
    ranges = re.findall(r'(?im)^content-range:\s*bytes (\d+)-(\d+)/(\d+)\s*$', headers)
    statuses = re.findall(r'ROBOT_EXAM_HTTP_STATUS=(\d+)', headers)
    if (not statuses or statuses[-1] != '206' or not ranges
            or tuple(map(int, ranges[-1])) != (start, end, total)
            or actual_bytes != end-start+1):
        raise ValueError('Server range response does not match the requested bytes')


def download_in_ranges(partial, url, total, deadline, chunk_bytes):
    """Bound long HTTP streams; append only exact 206 ranges, then hash full file."""
    while (partial.stat().st_size if partial.exists() else 0) < total:
        start = partial.stat().st_size if partial.exists() else 0
        end = min(start+chunk_bytes, total)-1
        remaining = int(deadline-time.monotonic())
        if remaining <= 0:
            raise TimeoutError('Download budget exhausted')
        chunk = partial.with_name(partial.name+'.range-'+uuid.uuid4().hex)
        response = subprocess.run(
            ['curl', '--fail', '--location', '--silent', '--show-error',
             '--connect-timeout', '20', '--retry', '2', '--retry-delay', '2',
             '--retry-max-time', str(remaining), '--max-time', str(min(180, remaining)),
             '--range', f'{start}-{end}', '--dump-header', '-',
             '--write-out', '\nROBOT_EXAM_HTTP_STATUS=%{http_code}\n',
             '--output', str(chunk), url],
            check=True, timeout=remaining, capture_output=True, text=True)
        validate_range_response(response.stdout, start, end, total, chunk.stat().st_size)
        with partial.open('ab') as output, chunk.open('rb') as source:
            shutil.copyfileobj(source, output, length=1024**2)
            output.flush()
            os.fsync(output.fileno())
        chunk.unlink()  # Only this attempt's successfully appended scratch file.
        print('RANGE_APPENDED', partial.name, end+1, '/', total, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extra-deps', type=Path)
    parser.add_argument('--authorized-download', action='store_true')
    parser.add_argument('--max-total-seconds', type=int, default=7200)
    parser.add_argument('--chunk-mib', type=int, default=0,
                        help='Optional bounded HTTP ranges (1..64MiB); 0 uses a single curl stream')
    args = parser.parse_args()
    if not args.authorized_download:
        parser.error('Explicit user download authorization required; training is never started')
    if args.output.exists():
        raise FileExistsError('Preserve previous download outcomes')
    if not 60 <= args.max_total_seconds <= 43200:
        parser.error('Use a bounded download budget of 60..43200 seconds')
    if not 0 <= args.chunk_mib <= 64:
        parser.error('Chunk size must be 0..64MiB')
    if args.extra_deps:
        sys.path.insert(0, str(args.extra_deps.resolve()))
    import google_crc32c  # noqa: F401 - fail before any download if verifier unavailable
    manifest = json.loads(args.manifest.read_text())
    if manifest['weights_bucket'] != 'openpi-assets' or manifest['prefix'] != 'checkpoints/pi05_base/params/' or manifest['tokenizer_bucket'] != 'big_vision':
        raise ValueError('Only official pi05_base and PaliGemma assets supported')
    # Finish the small tokenizer first so a slow checkpoint does not block it.
    rows = [(manifest['tokenizer_bucket'], manifest['tokenizer'])]
    rows.extend((manifest['weights_bucket'], row) for row in manifest['objects'])
    for bucket, item in rows:
        asset_relative_path(item['name'])
        if bucket == 'openpi-assets' and not item['name'].startswith(manifest['prefix']):
            raise ValueError('Unexpected checkpoint prefix')
        if bucket == 'big_vision' and item['name'] != 'paligemma_tokenizer.model':
            raise ValueError('Unexpected tokenizer object')
        if not item.get('generation') or not item.get('crc32c'):
            raise ValueError('Freeze official object generation and CRC32C before downloading')
    args.cache.mkdir(parents=True, exist_ok=True)
    # Different attempt names still share the same .partial files.
    lock = (args.cache/'.official_pi05_download.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('An official asset download already owns this cache') from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'objects': [], 'manifest': str(args.manifest.resolve()),
              'model_loaded': False, 'training_performed': False, 'user_authorized_download': True,
              'chunk_mib': args.chunk_mib, 'max_total_seconds': args.max_total_seconds}
    log = args.output.with_suffix('.progress.jsonl')
    if log.exists():
        raise FileExistsError('Use a new download attempt/log')
    started = time.monotonic()
    def interrupted(signum, frame):
        raise InterruptedError(f'Download interrupted by signal {signum}; partials preserved')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for bucket, item in rows:
            relative = asset_relative_path(item['name'])
            if item['name'].endswith('/') and int(item['size']) == 0:
                continue
            target = args.cache/bucket/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            reused = target.exists()
            if not reused:
                partial = target.with_name(target.name+'.partial')
                current = partial.stat().st_size if partial.exists() else 0
                if current > int(item['size']):
                    raise ValueError('Oversized partial file; preserve and inspect before retry')
                needed = int(item['size'])-current+25*1024**3
                if shutil.disk_usage(args.cache).free < needed:
                    raise RuntimeError('Insufficient disk for next object plus 25GiB reserve')
                remaining = int(args.max_total_seconds-(time.monotonic()-started))
                if remaining <= 0:
                    raise TimeoutError('Download budget exhausted')
                if int(item['size']) == 0 and not partial.exists():
                    partial.touch(exist_ok=False)
                if current != int(item['size']):
                    url = 'https://storage.googleapis.com/'+bucket+'/'+quote(item['name'], safe='/')+'?'+urlencode({'generation': item['generation']})
                    print('DOWNLOADING', item['name'], item['size'], 'resume_bytes', current, flush=True)
                    if args.chunk_mib:
                        download_in_ranges(partial, url, int(item['size']),
                                           started+args.max_total_seconds, args.chunk_mib*1024**2)
                    else:
                        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                                        '--connect-timeout', '20', '--retry', '2', '--retry-delay', '2',
                                        '--retry-max-time', str(remaining), '--max-time', str(remaining),
                                        '--continue-at', '-', '--output', str(partial), url],
                                       check=True, timeout=remaining)
                checked = verify_file(partial, item)
                if target.exists():
                    raise FileExistsError('Asset appeared concurrently; do not replace it')
                partial.rename(target)
            else:
                checked = verify_file(target, item)
            row = {'name': item['name'], 'bucket': bucket, 'generation': item['generation'],
                   'reused': reused, 'path': str(target), **checked}
            report['objects'].append(row)
            with log.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print('VERIFIED', item['name'], flush=True)
        report['status'] = 'PASS_OFFICIAL_ASSETS_VERIFIED_NOT_LOADED'
        report['weights_path'] = str(args.cache/'openpi-assets/checkpoints/pi05_base/params')
        report['tokenizer_path'] = str(args.cache/'big_vision/paligemma_tokenizer.model')
    except BaseException as error:
        report.update(status='FAILED_OR_INTERRUPTED_PARTIALS_PRESERVED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        report['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'objects'}, indent=2), flush=True)


if __name__ == '__main__':
    main()

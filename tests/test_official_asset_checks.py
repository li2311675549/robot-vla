import base64
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import time

import pytest

spec = importlib.util.spec_from_file_location('asset_check', Path(__file__).resolve().parents[1]/'scripts/download_official_pi05_assets.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_same_size_corruption_does_not_pass_asset_integrity(tmp_path):
    path = tmp_path/'asset'
    content = b'original checkpoint bytes'
    metadata = {'size': str(len(content)), 'md5Hash': base64.b64encode(hashlib.md5(content).digest()).decode()}
    path.write_bytes(content)
    assert module.verify_file(path, metadata)['bytes'] == len(content)
    path.write_bytes(b'x'*len(content))
    with pytest.raises(ValueError, match='mismatch'):
        module.verify_file(path, metadata)


def test_partial_and_missing_checksums_are_rejected(tmp_path):
    path = tmp_path/'asset'
    path.write_bytes(b'abc')
    with pytest.raises(ValueError, match='Incomplete'):
        module.verify_file(path, {'size': '10', 'md5Hash': 'unused'})
    with pytest.raises(ValueError, match='checksum'):
        module.verify_file(path, {'size': '3'})


def test_official_empty_commit_marker_is_a_valid_file(tmp_path):
    path = tmp_path/'commit_success.txt'
    path.touch()
    expected = base64.b64encode(hashlib.md5(b'').digest()).decode()
    assert module.verify_file(path, {'size': '0', 'md5Hash': expected})['bytes'] == 0


@pytest.mark.parametrize('name', ['../escape', '/absolute/file', 'params/../../escape'])
def test_asset_names_cannot_escape_cache(name):
    with pytest.raises(ValueError, match='Unsafe'):
        module.asset_relative_path(name)


@pytest.mark.parametrize('header,size', [
    ('Content-Range: bytes 0-3/8\nROBOT_EXAM_HTTP_STATUS=200', 4),
    ('Content-Range: bytes 4-7/8\nROBOT_EXAM_HTTP_STATUS=206', 4),
    ('Content-Range: bytes 0-3/8\nROBOT_EXAM_HTTP_STATUS=206', 3),
])
def test_ignored_or_incorrect_range_is_rejected(header, size):
    with pytest.raises(ValueError, match='range response'):
        module.validate_range_response(header, 0, 3, 8, size)


def test_range_resume_preserves_prefix_and_assembles_exact_content(tmp_path, monkeypatch):
    original = b'original official checkpoint content'
    partial = tmp_path/'checkpoint.partial'
    partial.write_bytes(original[:5])
    requested = []

    def curl(args, **kwargs):
        start, end = map(int, args[args.index('--range')+1].split('-'))
        requested.append((start, end))
        Path(args[args.index('--output')+1]).write_bytes(original[start:end+1])
        headers = f'Content-Range: bytes {start}-{end}/{len(original)}\r\nROBOT_EXAM_HTTP_STATUS=206\n'
        return subprocess.CompletedProcess(args, 0, stdout=headers)

    monkeypatch.setattr(module.subprocess, 'run', curl)
    module.download_in_ranges(partial, 'https://official.invalid/object', len(original), time.monotonic()+60, 7)
    assert requested[0][0] == 5
    assert partial.read_bytes() == original
    assert list(tmp_path.iterdir()) == [partial]


def test_bad_range_does_not_append_to_checkpoint(tmp_path, monkeypatch):
    partial = tmp_path/'checkpoint.partial'
    partial.write_bytes(b'valid')

    def curl(args, **kwargs):
        Path(args[args.index('--output')+1]).write_bytes(b'wrong')
        return subprocess.CompletedProcess(args, 0, stdout='ROBOT_EXAM_HTTP_STATUS=200\n')

    monkeypatch.setattr(module.subprocess, 'run', curl)
    with pytest.raises(ValueError, match='range response'):
        module.download_in_ranges(partial, 'https://official.invalid/object', 10, time.monotonic()+60, 5)
    assert partial.read_bytes() == b'valid'
    assert len(list(tmp_path.iterdir())) == 2  # Failed scratch kept for diagnosis.

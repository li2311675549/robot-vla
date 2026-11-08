"""Small fixtures prove byte preservation before compacting real evidence."""
import hashlib
import importlib.util
import os
from pathlib import Path
import tarfile

import pytest

spec = importlib.util.spec_from_file_location('compact_storage', Path(__file__).parents[1]/'scripts/compact_local_storage.py')
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


def test_upload_archive_exact_recipe_preserves_pax_and_hardlinks(tmp_path):
    bundle = tmp_path/'bundle'
    (bundle/'nested').mkdir(parents=True)
    (bundle/'nested'/'数据.txt').write_bytes(b'\x00sample\xff'*1000)
    (bundle/'empty').touch()
    os.link(bundle/'nested'/'数据.txt', bundle/'alias')
    archive = tmp_path/'old.tar'
    with tarfile.open(archive, 'w') as stream:
        stream.add(bundle, arcname='robot_exam_training')
    recipe = storage.archive_recipe(archive, bundle, tmp_path/'recipe')
    reconstructed = hashlib.sha256()
    storage.stream_recipe(recipe, reconstructed.update)
    assert reconstructed.hexdigest() == storage.digest(archive)
    assert recipe['streamed_byte_exact_reconstruction'] == 'PASS'
    (bundle/'alias').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source|Truncated'):
        storage.stream_recipe(recipe, hashlib.sha256().update)


def test_log_lossless_restore_and_existing_gzip_protection(tmp_path):
    path = tmp_path/'trajectory.json'
    payload = b'{"literal_space":  1, "rows": [1, 2, 3]}\n'*200
    path.write_bytes(payload)
    row = storage.compact_log(path)
    assert path.read_bytes() == payload  # Caller journals before removing it.
    assert row['compressed_bytes'] < row['original_bytes']
    assert storage.compact_log(path)['status'] == 'SKIPPED_EXISTING_GZIP'
    path.unlink()
    storage.restore_log(row)
    assert path.read_bytes() == payload
    with pytest.raises(FileExistsError):
        storage.restore_log(row)


def test_corrupt_compressed_log_cannot_be_restored(tmp_path):
    path = tmp_path/'payload_traces.json'
    path.write_bytes(b'[true, false]\n'*100)
    row = storage.compact_log(path)
    path.unlink()
    Path(row['compressed']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        storage.restore_log(row)
    assert not path.exists()

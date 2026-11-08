import hashlib
import pytest

from robot_exam.vla.checkpoint_cache import advise_checkpoint_cache,checkpoint_disk_budget


def test_cache_advice_preserves_payload_metadata_and_skips_external_symlinks(tmp_path):
    root=tmp_path/'checkpoint';root.mkdir();payload=root/'params';payload.write_bytes(b'checkpoint payload'*1000)
    outside=tmp_path/'outside';outside.write_bytes(b'not a checkpoint');(root/'symlink').symlink_to(outside)
    before=payload.stat();digest=hashlib.sha256(payload.read_bytes()).hexdigest()
    result=advise_checkpoint_cache([root],tmp_path/'audit.jsonl')
    assert result['files']==1 and result['files_deleted']==0 and result['metadata_unchanged']
    assert hashlib.sha256(payload.read_bytes()).hexdigest()==digest
    assert (before.st_ino,before.st_mtime_ns,before.st_size)==(payload.stat().st_ino,payload.stat().st_mtime_ns,payload.stat().st_size)
    assert outside.read_bytes()==b'not a checkpoint'


def test_disk_budget_ignores_incomplete_or_tiny_checkpoints(tmp_path):
    path=tmp_path/'model/run/29';path.mkdir(parents=True)
    (path/'_CHECKPOINT_METADATA').write_text('{}')
    assert checkpoint_disk_budget(tmp_path)['required_free_gib']==65
    for name in ('params/_METADATA','train_state/_METADATA'):
        p=path/name;p.parent.mkdir(exist_ok=True);p.write_text('{}')
    assert checkpoint_disk_budget(tmp_path)['required_free_gib']==65


def test_resume_disk_budget_counts_preallocated_active_checkpoint_once(tmp_path):
    active=tmp_path/'model/run';folder=active/'5000';folder.mkdir(parents=True)
    for name in ('_CHECKPOINT_METADATA','params/_METADATA','train_state/_METADATA'):
        p=folder/name;p.parent.mkdir(exist_ok=True);p.write_text('{}')
    # Sparse fixture:9GiB logical checkpoint size, no9GiB data allocation/write.
    with (folder/'params/sparse_fixture').open('wb') as stream:stream.truncate(9*1024**3)
    fresh=checkpoint_disk_budget(tmp_path,active_checkpoint_dir=tmp_path/'model/new_run')
    resumed=checkpoint_disk_budget(tmp_path,active_checkpoint_dir=active)
    assert fresh['required_free_gib']==pytest.approx(57.,abs=1e-5)
    assert resumed['required_free_gib']==pytest.approx(48.,abs=1e-5)
    assert resumed['additional_checkpoint_slots']==2 and resumed['existing_complete_in_active_run']==1
    assert resumed['minimum_free_gib']==25 and resumed['transient_gib']==5

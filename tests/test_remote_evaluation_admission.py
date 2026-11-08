"""The larger evaluation preserves disk safety and rejects the wrong model."""
import importlib.util
from pathlib import Path
import sys

import pytest


def script(name):
    directory = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, directory / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disk_admission_keeps_old_default_and_allows_safe_frozen_budget():
    limits = script('evaluate_remote_fruit_development').resource_limits
    assert limits({}) == (26 * 1024**3, None)
    assert limits({'minimum_free_gib': 25.5, 'maximum_output_bytes': 600 * 1024**2}) == (
        25.5 * 1024**3, 600 * 1024**2)
    for bad in (24.9, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            limits({'minimum_free_gib': bad})
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            limits({'maximum_output_bytes': bad})


def test_checkpoint_mismatch_refused_before_learned_action():
    require = script('evaluate_stationary_dual_skill').require_checkpoint
    require({'checkpoint_step_label': '9999'}, '9999')
    require({}, None)  # Historical default remains compatible.
    for metadata in ({}, {'checkpoint_step_label': '999'}):
        with pytest.raises(ValueError, match='Checkpoint mismatch'):
            require(metadata, '9999')


def test_shared_lock_refuses_duplicate_runner_and_allows_after_release(tmp_path):
    acquire=script('evaluate_remote_fruit_development').acquire_evaluation_lock
    first=acquire(tmp_path/'lock')
    with pytest.raises(RuntimeError,match='duplicate batch'):
        acquire(tmp_path/'lock')
    first.close()
    second=acquire(tmp_path/'lock')
    second.close()


def test_inference_seed_is_explicit_bounded_and_not_a_scene_change():
    seed=script('serve_openpi_original_ur5').evaluation_noise_seed
    assert seed({},False)==0
    assert seed({'X-Robot-Exam-Evaluation-Seed':'7'},True)==7
    for value in ('-1','2147483648','invalid'):
        with pytest.raises(ValueError):seed({'X-Robot-Exam-Evaluation-Seed':value},True)
    with pytest.raises(ValueError):seed({'X-Robot-Exam-Evaluation-Seed':'1'},False)


def test_exported_source_revision_requires_real_file_hashes(tmp_path):
    import hashlib,json
    revision=script('evaluate_remote_fruit_development').source_revision
    source=tmp_path/'code.py';source.write_text('actual source\n')
    manifest={'git_commit':'a'*40,'files':{'code.py':hashlib.sha256(source.read_bytes()).hexdigest()}}
    (tmp_path/'source_revision.json').write_text(json.dumps(manifest))
    assert revision(tmp_path)==('a'*40,'verified_source_export_no_git_checkout')
    source.write_text('changed source\n')
    with pytest.raises(ValueError,match='manifest mismatch'):revision(tmp_path)


def test_readonly_vision_queries_refuse_incomplete_or_changed_batch(tmp_path):
    import json
    require=script('probe_fixed_policy_vision').require_complete_matrix
    (tmp_path/'summary.json').write_text(json.dumps({'completed_cases':59,'expected_cases':60}))
    (tmp_path/'integrity.json').write_text(json.dumps({'source_hash_mismatches':[]}))
    with pytest.raises(RuntimeError,match='completed source-frozen'):require(tmp_path)
    (tmp_path/'summary.json').write_text(json.dumps({'completed_cases':60,'expected_cases':60}))
    (tmp_path/'integrity.json').write_text(json.dumps({'source_hash_mismatches':['changed']}))
    with pytest.raises(RuntimeError):require(tmp_path)
    (tmp_path/'integrity.json').write_text(json.dumps({'source_hash_mismatches':[]}))
    require(tmp_path)

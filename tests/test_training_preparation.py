"""Training configuration/asset isolation and portable dataset layout regressions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_exam.vla.training_plan import TrainingPlan, episode_sample_indices
from robot_exam.vla.offline_assets import local_assets_only, TOKENIZER_URI, verify_checkpoint_statistics

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('settings', ['pi05_dual_fruit_smoke_v1.json', 'pi05_dual_fruit_pilot_v1.json'])
def test_declared_warmup_finishes_before_training(settings):
    plan = TrainingPlan.load(ROOT/'configs'/settings)
    assert plan.warmup_steps < plan.steps
    assert plan.validation_repo_id != 'local/robot_exam_dual_release_train'


@pytest.mark.parametrize('values', [
    {'run_name': '../overwrite'}, {'steps': 100, 'warmup_steps': 100},
    {'peak_lr': float('nan')}, {'batch_size': 0}, {'num_workers': -1},
])
def test_unsafe_experiments_are_rejected(values):
    with pytest.raises(ValueError):
        TrainingPlan(**values)


def test_validation_sampling_is_episode_balanced_and_deterministic():
    episodes = [{'length': 102}, {'length': 60}]
    assert episode_sample_indices(episodes, 1) == [51, 132]
    assert episode_sample_indices(episodes, 3) == [17, 51, 85, 112, 132, 152]


def test_asset_guard_disallows_network_and_restores_after_exception(tmp_path):
    tokenizer = tmp_path/'tokenizer.model'
    tokenizer.write_bytes(b'unit fixture, not a real tokenizer')
    original = lambda path: (_ for _ in ()).throw(AssertionError('No download allowed'))
    download = SimpleNamespace(maybe_download=original)
    with pytest.raises(RuntimeError, match='network'):
        with local_assets_only(tokenizer, download):
            assert download.maybe_download(TOKENIZER_URI) == tokenizer.resolve()
            assert download.maybe_download(str(tokenizer)) == tokenizer.resolve()
            download.maybe_download('gs://unapproved/weights')
    assert download.maybe_download is original


def test_checkpoint_statistics_allow_serialization_changes_but_not_value_changes(tmp_path):
    expected, restored = tmp_path/'original.json', tmp_path/'checkpoint.json'
    stats = {'norm_stats': {'state': {'mean': [0., 1.], 'std': [1., 2.]}}}
    expected.write_text(json.dumps(stats))
    restored.write_text(json.dumps(stats, indent=2))
    verify_checkpoint_statistics(expected, restored)
    stats['norm_stats']['state']['mean'][0] = .1
    restored.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match='normalization'):
        verify_checkpoint_statistics(expected, restored)


def test_portable_layout_uses_copied_exports_and_refuses_redirection(tmp_path):
    spec = importlib.util.spec_from_file_location('layout', ROOT/'scripts/prepare_autodl_layout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    accepted = []
    for seed, split in enumerate(('train', 'validation', 'test')):
        meta = tmp_path/'datasets'/split/'meta'
        meta.mkdir(parents=True)
        info = {'fps': 5, 'total_frames': 10,
                'features': {'action': {'shape': [7]}, 'observation.images.front': {}, 'observation.images.wrist': {}}}
        (meta/'info.json').write_text(json.dumps(info))
        (meta/'episodes.jsonl').write_text(json.dumps({'length': 10})+'\n')
        accepted.append({'seed': seed, 'split': split})
    (tmp_path/'dataset_manifest').mkdir()
    (tmp_path/'dataset_manifest/manifest.json').write_text(json.dumps({'accepted': accepted}))
    cache = tmp_path/'cache'
    report = module.prepare_layout(tmp_path, cache)
    assert report['scene_split_isolation']
    assert module.prepare_layout(tmp_path, cache)['splits']['test']['frames'] == 10
    with pytest.raises(FileExistsError):
        other = tmp_path/'wrong_cache/local/robot_exam_dual_release_train'
        other.mkdir(parents=True)
        module.prepare_layout(tmp_path, tmp_path/'wrong_cache')

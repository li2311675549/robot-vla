"""Pinned OpenPI trainer adapter for the qualified old/new LeRobot release.

This module does not import a model, allocate a GPU, or download anything.
The installed factory uses the normal tokenizer/normalization/model transforms.
"""
from contextlib import contextmanager
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_REPOS = {'old': 'local/robot_exam_old_train',
                'new': 'local/robot_exam_stage21_train'}


def read_mixture(root):
    root = Path(root).resolve()
    prep = json.loads((root/'mixture/preparation.json').read_text())
    if prep['action_semantics'] != 'ur5_absolute_joint_goal_v1' or prep['fps'] != 5 or prep['action_horizon'] != 10:
        raise ValueError('Mixture action semantics/frequency/horizon mismatch')
    episodes = prep['episodes']
    if not episodes or any(e['split'] != 'train' for e in episodes):
        raise ValueError('Held-out episodes cannot enter the training sampler')
    # Ordering matters: ConcatDataset is old then new, each in episode order.
    ordered = sorted(episodes, key=lambda e: (list(SOURCE_REPOS).index(e['source']), e['episode']))
    if episodes != ordered:
        raise ValueError('Mixture index ordering differs from old/new LeRobot concatenation')
    from robot_exam.data.mixing import frame_weights
    expected, _, _ = frame_weights(episodes, {'old': .2, 'new': .8})
    with np.load(root/'mixture/sampling_weights.npz') as data:
        weights = data['weights'].copy()
    if weights.shape != expected.shape or not np.allclose(weights, expected, rtol=1e-12, atol=1e-16):
        raise ValueError('Frozen sampling weights do not match training episode identities')
    for source, repo in SOURCE_REPOS.items():
        subset = [e for e in episodes if e['source'] == source]
        if [e['episode'] for e in subset] != list(range(len(subset))):
            raise ValueError('Training episode indices must be consecutive within each source')
        info = json.loads((root/'lerobot'/repo/'meta/info.json').read_text())
        if info['total_frames'] != sum(e['frames'] for e in subset) or info['total_episodes'] != len(subset):
            raise ValueError(f'{source} metadata/frame count mismatch')
        if info['fps'] != 5:
            raise ValueError('Training source frequency mismatch')
    stats = root/'mixture/assets/original_ur5/norm_stats.json'
    if stats.read_bytes() != (root/'assets/original_ur5/norm_stats.json').read_bytes():
        raise ValueError('Mixture and exported normalization assets differ')
    return prep, weights


def resume_offset(checkpoint_dir, resume):
    """Pinned train.py saves directory i after optimizer step i+1."""
    if not resume:
        return 0
    completed = [int(p.name) + 1 for p in Path(checkpoint_dir).iterdir()
                 if p.name.isdigit() and (p/'_CHECKPOINT_METADATA').is_file()
                 and (p/'params/_METADATA').is_file() and (p/'train_state/_METADATA').is_file()]
    if not completed:
        raise FileNotFoundError('No completed checkpoint for deterministic sampler resume')
    return max(completed)


@contextmanager
def install_mixed_loader(root, audit_path=None):
    """Replace only the declared train repo's loader; restore upstream on exit.

    Single process / zero workers keeps sample provenance exact. A deterministic
    finite draw stream can resume at the completed checkpoint's optimizer step;
    no claim of bit-identical device stochastic state beyond the pinned trainer.
    """
    import torch
    from openpi.training import data_loader

    root = Path(root).resolve()
    prep, weights = read_mixture(root)
    original = data_loader.create_data_loader
    audit = {'source_frames': {}, 'draw_counts': {'old': 0, 'new': 0},
             'first_indices': [], 'tokenizer_bypassed': False,
             'normalization_sha256': hashlib.sha256((root/'assets/original_ur5/norm_stats.json').read_bytes()).hexdigest()}

    def factory(cfg, *, sharding=None, shuffle=False, num_batches=None,
                skip_norm_stats=False, framework='jax'):
        if cfg.data.repo_id != SOURCE_REPOS['new']:
            return original(cfg, sharding=sharding, shuffle=shuffle, num_batches=num_batches,
                            skip_norm_stats=skip_norm_stats, framework=framework)
        if skip_norm_stats or cfg.num_workers != 0 or cfg.model.action_horizon != 10:
            raise ValueError('Qualified mixed loader requires normalization, horizon10 and zero workers')
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        if not dc.model_transforms.inputs:
            raise ValueError('Real training requires model transforms and the real tokenizer')
        from openpi.shared import normalize
        expected_stats = normalize.load(root/'assets/original_ur5')
        for key in ('state', 'actions'):
            for field in ('mean', 'std', 'q01', 'q99'):
                if not np.array_equal(getattr(dc.norm_stats[key], field), getattr(expected_stats[key], field)):
                    raise ValueError('Training factory loaded a different normalization asset')
        datasets = []
        for source, repo in SOURCE_REPOS.items():
            native = data_loader.create_torch_dataset(dataclasses.replace(dc, repo_id=repo), 10, cfg.model)
            datasets.append(data_loader.transform_dataset(native, dc))
            audit['source_frames'][source] = len(native)
        combined = torch.utils.data.ConcatDataset(datasets)
        if len(combined) != len(weights):
            raise ValueError('Actual dataset length differs from frozen sampling weights')
        split_at = len(datasets[0])

        class TracedDataset:
            def __len__(self):
                return len(combined)

            def __getitem__(self, index):
                item = combined[index]
                if 'prompt' in item or 'tokenized_prompt' not in item or 'tokenized_prompt_mask' not in item:
                    raise ValueError('Real prompt tokenization was not executed')
                audit['draw_counts']['old' if index < split_at else 'new'] += 1
                if len(audit['first_indices']) < 64:
                    audit['first_indices'].append(int(index))
                return item

        start = resume_offset(cfg.checkpoint_dir, cfg.resume)
        # One extra prefetched batch is consumed by pinned train.py after its
        # final update. Keep draw stream bounded and deterministic across resume.
        draws = torch.multinomial(torch.from_numpy(weights), (cfg.num_train_steps + 1)*cfg.batch_size,
                                  replacement=True, generator=torch.Generator().manual_seed(cfg.seed))
        draws = draws[start*cfg.batch_size:].tolist()
        if not draws:
            # A completed run may reload its final checkpoint for verification.
            draws = [0]*cfg.batch_size
        audit.update(resume_optimizer_step=start, sampler_seed=cfg.seed,
                     expected_source_mass=prep['source_mass'], framework=framework)
        loader = data_loader.TorchDataLoader(TracedDataset(), cfg.batch_size,
            sampler=draws, sharding=None if framework == 'pytorch' else sharding,
            num_batches=num_batches, num_workers=0, seed=cfg.seed, framework=framework)
        return data_loader.DataLoaderImpl(dc, loader)

    data_loader.create_data_loader = factory
    try:
        yield audit
    finally:
        data_loader.create_data_loader = original
        if audit_path:
            path = Path(audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(audit, indent=2)+'\n')

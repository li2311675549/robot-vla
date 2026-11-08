#!/usr/bin/env python3
"""No-GPU actual trainer loader/tokenizer preflight; never loads model weights."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

from _bootstrap import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--openpi-root', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--settings', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous preflight attempts')
    root = args.dataset_root.resolve()
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', JAX_PLATFORMS='cpu',
                     CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled',
                     HF_LEROBOT_HOME=str(root/'lerobot'))
    sys.path.insert(0, str(args.openpi_root/'src'))
    from robot_exam.vla.lerobot_compat import enable_list_feature_compat
    enable_list_feature_compat()
    from robot_exam.vla.offline_assets import local_assets_only
    from robot_exam.vla.training_plan import TrainingPlan
    from robot_exam.vla.mixed_training import install_mixed_loader, read_mixture
    from robot_exam.vla.openpi_config import make_config
    import numpy as np
    import pyarrow.parquet as pq
    import jax
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training import data_loader

    started = time.monotonic()
    report = {'status': 'RUNNING', 'model_loaded': False, 'training_started': False,
              'tokenizer_bypassed': False, 'numeric_only_probe': False,
              'dataset_root': str(root), 'partitions': {}, 'checks': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report['wall_seconds'] = time.monotonic()-started
        report['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    save()
    try:
        prep, weights = read_mixture(root)
        plan = TrainingPlan.load(args.settings)
        report['context_split_isolation'] = {}
        for source, filename in [('old', 'manifest_old_portable.json'), ('new', 'manifest_new_portable.json')]:
            rows = json.loads((root/filename).read_text())['accepted']
            contexts = {split: {r.get('scene_context_id', str(r['seed'])) for r in rows if r['split'] == split}
                        for split in ('train', 'validation', 'test')}
            if any(contexts[a] & contexts[b] for a, b in
                   [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
                raise ValueError('A complete scene context appears in multiple partitions')
            report['context_split_isolation'][source] = {k: len(v) for k, v in contexts.items()}
        with local_assets_only(args.tokenizer):
            cfg = make_config('local/robot_exam_stage21_train', root/'assets', args.weights,
                              root.parent/'checkpoints', wrist_camera=True, training_plan=plan)
            # Import exactly the upstream module used by the production entrypoint.
            spec = importlib.util.spec_from_file_location('stage22_actual_train', args.openpi_root/'scripts/train.py')
            trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(trainer)
            tokenizer = PaligemmaTokenizer(cfg.model.max_token_len)
            maximum_tokens = 0
            prompt_counts = {}
            for source, prefix in [('old', 'robot_exam_old'), ('new', 'robot_exam_stage21')]:
                for split in ('train', 'validation', 'test'):
                    repo = f'local/{prefix}_{split}'
                    dataset = root/'lerobot'/repo
                    info = json.loads((dataset/'meta/info.json').read_text())
                    episodes = [json.loads(x) for x in (dataset/'meta/episodes.jsonl').read_text().splitlines()]
                    tasks = {r['task_index']: r['task'] for r in
                             map(json.loads, (dataset/'meta/tasks.jsonl').read_text().splitlines())}
                    total = 0
                    for ep in episodes:
                        path = dataset/info['data_path'].format(episode_chunk=ep['episode_index']//info['chunks_size'],
                                                                episode_index=ep['episode_index'])
                        table = pq.read_table(path, columns=['observation.state', 'action', 'task_index'])
                        if len(table) != ep['length']:
                            raise ValueError('Episode length mismatch')
                        state = np.asarray(table['observation.state'].to_pylist())
                        actions = np.asarray(table['action'].to_pylist())
                        if state.shape != (len(table), 7) or actions.shape != state.shape:
                            raise ValueError('Measured state/high-level action dimensions mismatch')
                        if not np.isfinite(state).all() or not np.isfinite(actions).all():
                            raise ValueError('Nonfinite data')
                        if np.any((actions[:, 6] < 0) | (actions[:, 6] > 1)):
                            raise ValueError('Illegal normalized gripper action')
                        prompt = tasks[int(table['task_index'][0].as_py())]
                        # Use real quantile-normalized states, matching PI05 transforms.
                        stats = json.loads((root/'assets/original_ur5/norm_stats.json').read_text())['norm_stats']['state']
                        norm = (state - np.asarray(stats['q01']))/(np.asarray(stats['q99'])-np.asarray(stats['q01'])+1e-6)*2-1
                        # All distinct states/prompts, no fake IDs or hidden truth.
                        for st in norm:
                            cleaned = prompt.strip().replace('_', ' ').replace('\n', ' ')
                            discrete = np.digitize(st, bins=np.linspace(-1, 1, 257)[:-1])-1
                            text = f'Task: {cleaned}, State: '+ ' '.join(map(str, discrete))+';\nAction: '
                            n = len(tokenizer._tokenizer.encode(text, add_bos=True))
                            maximum_tokens = max(maximum_tokens, n)
                            if n > cfg.model.max_token_len:
                                raise ValueError('Prompt/state truncated before the action prefix')
                        total += len(table)
                        prompt_counts.setdefault(f'{source}/{split}', set()).add(prompt)
                    if total != info['total_frames']:
                        raise ValueError('Split frame count mismatch')
                    report['partitions'][f'{source}/{split}'] = {'episodes': len(episodes), 'frames': total,
                        'unique_instructions': len(prompt_counts[f'{source}/{split}'])}
                    save()
                    gc.collect()
            report['max_actual_prompt_tokens'] = maximum_tokens
            report['max_token_len'] = cfg.model.max_token_len
            with install_mixed_loader(root, args.output.with_suffix('.sampler.json')) as audit:
                # Invoke trainer's actual factory, not a standalone NumericOnly loader.
                loader = trainer._data_loader.create_data_loader(cfg, shuffle=True, num_batches=4)
                for observation, actions in loader:
                    assert tuple(actions.shape) == (cfg.batch_size, 10, 32)
                    assert tuple(observation.state.shape) == (cfg.batch_size, 32)
                    assert observation.tokenized_prompt is not None
                    assert observation.tokenized_prompt_mask.any()
                    assert observation.image_masks['base_0_rgb'].all()
                    assert observation.image_masks['left_wrist_0_rgb'].all()
                    assert not observation.image_masks['right_wrist_0_rgb'].any()
                    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((observation, actions)))
                    report['checks'].append({'state': list(observation.state.shape),
                        'actions': list(actions.shape), 'tokens': list(observation.tokenized_prompt.shape)})
                report['actual_draw_counts'] = audit['draw_counts']
            # Both train sources and both native held-out sources use the real transforms.
            dc = cfg.data.create(cfg.assets_dirs, cfg.model)
            import dataclasses
            report['native_partition_samples'] = {}
            for source, prefix in [('old', 'robot_exam_old'), ('new', 'robot_exam_stage21')]:
                for split in ('train', 'validation', 'test'):
                    local = dataclasses.replace(dc, repo_id=f'local/{prefix}_{split}')
                    native = data_loader.create_torch_dataset(local, 10, cfg.model)
                    mapped = data_loader.transform_dataset(native, local)
                    first, last = mapped[0], mapped[len(mapped)-1]
                    assert last['actions'].shape == (10, 32)
                    np.testing.assert_allclose(last['actions'], np.repeat(last['actions'][:1], 10, axis=0))
                    report['native_partition_samples'][f'{source}/{split}'] = {
                        'frames': len(mapped), 'first_tokens': int(first['tokenized_prompt_mask'].sum()),
                        'last_tokens': int(last['tokenized_prompt_mask'].sum()), 'end_chunk_clamps': True}
                    del native, mapped, first, last
                    gc.collect()
                    save()
            report.update(status='PASS_REAL_TOKENIZER_TRAINER_LOADER_NO_MODEL',
                train_frames=len(weights), source_mass=prep['source_mass'], batch_size=cfg.batch_size,
                settings_sha256=hashlib.sha256(args.settings.read_bytes()).hexdigest(),
                tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                upstream_train_sha256=hashlib.sha256((args.openpi_root/'scripts/train.py').read_bytes()).hexdigest(),
                ready_for_gpu_preflight=True, gpu_model_checks_pending=True)
    except BaseException as error:
        report.update(status='FAILED_OR_INTERRUPTED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

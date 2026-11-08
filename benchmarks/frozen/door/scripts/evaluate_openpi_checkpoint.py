#!/usr/bin/env python3
"""Offline held-out action diagnostics; never a substitute for physical rollouts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from _bootstrap import ROOT
from robot_exam.vla.training_plan import TrainingPlan, episode_sample_indices
from robot_exam.vla.offline_assets import local_assets_only, verify_checkpoint_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openpi-root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--lerobot-cache', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authorized-model-load', action='store_true')
    args = parser.parse_args()
    if not args.authorized_model_load:
        parser.error('Real checkpoint evaluation requires --authorized-model-load on the server')
    if args.output.exists():
        raise FileExistsError('Do not overwrite checkpoint diagnostics')
    if not (args.checkpoint/'params').is_dir():
        raise FileNotFoundError('Provide a saved JAX checkpoint step containing params/')
    os.environ['HF_LEROBOT_HOME'] = str(args.lerobot_cache.resolve())
    os.environ.setdefault('HF_DATASETS_CACHE', str(args.lerobot_cache.resolve().parent/'hf_datasets_cache'))
    os.environ['HF_HUB_OFFLINE'] = os.environ['HF_DATASETS_OFFLINE'] = '1'
    sys.path.insert(0, str(args.openpi_root/'src'))
    import jax
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('Use the GPU server for real model evaluation')
    from robot_exam.vla.lerobot_compat import enable_list_feature_compat
    enable_list_feature_compat()
    from robot_exam.vla.openpi_config import make_config
    from openpi.training import data_loader
    from openpi.policies.policy_config import create_trained_policy
    plan = TrainingPlan.load(args.settings)
    cfg = make_config('local/robot_exam_dual_release_train', args.assets, args.checkpoint,
                      ROOT/'checkpoints', dry_check=True, wrist_camera=True, training_plan=plan)
    import dataclasses
    data = dataclasses.replace(cfg.data, repo_id=plan.validation_repo_id).create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
    episodes = [json.loads(x) for x in (args.lerobot_cache/plan.validation_repo_id/'meta/episodes.jsonl').read_text().splitlines() if x]
    indices = episode_sample_indices(episodes, 3)
    report = {'status': 'RUNNING', 'scope': 'validation action prediction, not physical success',
              'checkpoint': str(args.checkpoint.resolve()), 'validation_repo': plan.validation_repo_id,
              'model_loaded': False, 'training_performed': False, 'samples': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with local_assets_only(args.tokenizer):
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, dry_check=False))
        # create_trained_policy loads checkpoint-owned statistics, not base-model robot stats.
        policy = create_trained_policy(cfg, args.checkpoint)
        report['model_loaded'] = True
        expected = args.assets/'original_ur5/norm_stats.json'
        restored = args.checkpoint/'assets/original_ur5/norm_stats.json'
        verify_checkpoint_statistics(expected, restored)
        for index in indices:
            sample = dict(dataset[index])
            obs = {'state': np.asarray(sample['observation.state']),
                   'front_rgb': np.asarray(sample['observation.images.front']),
                   'wrist_rgb': np.asarray(sample['observation.images.wrist']), 'prompt': str(sample['prompt'])}
            row = {'frame_index': index, 'prompt': obs['prompt']}
            start = time.monotonic()
            try:
                actions = np.asarray(policy.infer(obs)['actions'])
                target = np.asarray(sample['action'])
                if actions.shape != (10, 7) or not np.isfinite(actions).all():
                    raise ValueError('Invalid prediction shape or nonfinite output')
                row.update(joint_rmse_rad=float(np.sqrt(np.mean((actions[:, :6]-target[:, :6])**2))),
                           gripper_mae=float(np.abs(actions[:, 6]-target[:, 6]).mean()),
                           first_joint_rmse_rad=float(np.sqrt(np.mean((actions[0, :6]-target[0, :6])**2))))
            except Exception as error:
                row['error'] = f'{type(error).__name__}: {error}'
            row['wall_seconds'] = time.monotonic()-start
            report['samples'].append(row)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
    report['invalid_predictions'] = sum('error' in row for row in report['samples'])
    report['status'] = 'FAIL_PREDICTIONS' if report['invalid_predictions'] else 'PASS_OFFLINE_DIAGNOSTICS_ONLY'
    for key in ('joint_rmse_rad', 'gripper_mae', 'first_joint_rmse_rad'):
        values = [row[key] for row in report['samples'] if key in row]
        report[key+'_mean_valid_only'] = float(np.mean(values)) if values else None
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'samples'}, indent=2))
    if report['invalid_predictions']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Offline configuration check by default; real training requires explicit authorization."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from _bootstrap import ROOT
from robot_exam.vla.training_plan import TrainingPlan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openpi-root', type=Path, default=Path('/home/li/vla/openpi'))
    parser.add_argument('--repo-id', default='local/robot_exam_dual_release_train')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, default=ROOT/'checkpoints/openpi')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--lerobot-cache', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--authorized-training', action='store_true')
    parser.add_argument('--wrist-camera', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new result path, including for resume attempts')
    plan = TrainingPlan.load(args.settings) if args.settings else TrainingPlan()
    if args.authorized_training and (not args.wrist_camera or not args.settings):
        parser.error('This pilot requires explicit --settings and --wrist-camera')
    if args.authorized_training and (not args.tokenizer or not args.lerobot_cache):
        parser.error('Supply local --tokenizer and --lerobot-cache before training')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    if args.lerobot_cache:
        os.environ['HF_LEROBOT_HOME'] = str(args.lerobot_cache.resolve())
        os.environ.setdefault('HF_DATASETS_CACHE', str(args.lerobot_cache.resolve().parent/'hf_datasets_cache'))
    sys.path.insert(0, str(args.openpi_root.resolve()/'src'))
    from robot_exam.vla.lerobot_compat import enable_list_feature_compat
    enable_list_feature_compat()
    from robot_exam.vla.openpi_config import make_config
    # Reports do not instantiate a tokenizer/model.
    cfg = make_config(args.repo_id, args.assets, args.weights, args.checkpoints,
                      dry_check=True, wrist_camera=args.wrist_camera,
                      training_plan=plan, resume=args.resume)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    report = {'status': 'CONFIGURATION_ONLY', 'config': cfg.name, 'robot': 'original_ur5',
              'action_dim_padded': cfg.model.action_dim, 'action_horizon': cfg.model.action_horizon,
              'policy_hz': 5, 'actual_action_dim': 7, 'lora': True,
              'training_performed': False, 'training_started': False,
              'norm_stats_loaded': data.norm_stats is not None,
              'weights_exist': args.weights.is_dir(), 'actual_camera_count': 2 if args.wrist_camera else 1,
              'delta_transform': False, 'settings': plan.to_dict(), 'resume': args.resume,
              'checkpoint_dir': str(cfg.checkpoint_dir), 'wandb_enabled': cfg.wandb_enabled,
              'weights_loaded': False, 'uploads_performed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save_report():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    save_report()
    if not args.authorized_training:
        print(json.dumps(report, indent=2))
        return
    started = time.monotonic()
    try:
        import jax
        devices = jax.devices()
        if not devices or any(device.platform != 'gpu' for device in devices):
            raise RuntimeError('Real training requires server GPUs; CPU fallback is forbidden')
        if len(devices) != 1:
            raise RuntimeError('This pilot is single-GPU: select one device with CUDA_VISIBLE_DEVICES')
        if not args.weights.is_dir() or not any(args.weights.iterdir()) or data.norm_stats is None:
            raise RuntimeError('Local base params directory and train-only statistics are required')
        for repo in (args.repo_id, plan.validation_repo_id):
            if not (args.lerobot_cache/repo/'meta/info.json').is_file():
                raise FileNotFoundError(f'Exported local dataset missing: {repo}')
        if (args.lerobot_cache/args.repo_id).resolve() == (args.lerobot_cache/plan.validation_repo_id).resolve():
            raise ValueError('Validation must not point to the training directory')
        if cfg.checkpoint_dir.exists() and not args.resume:
            raise FileExistsError('Experiment exists: use a new run_name, or explicit --resume')
        if args.resume and not cfg.checkpoint_dir.is_dir():
            raise FileNotFoundError('Cannot resume a nonexistent experiment')
        if args.resume and not any((path/'_CHECKPOINT_METADATA').is_file()
                                   and (path/'params/_METADATA').is_file()
                                   and (path/'train_state/_METADATA').is_file()
                                   for path in cfg.checkpoint_dir.iterdir() if path.name.isdigit()):
            raise FileNotFoundError('No completed checkpoint; refusing silent restart')
        args.checkpoints.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.checkpoints).free < 25 * 1024**3:
            raise RuntimeError('Keep at least 25GiB free before training; size real checkpoints in smoke test')
        from robot_exam.vla.offline_assets import local_assets_only
        from robot_exam.vla.openpi_validation import run_pinned_training
        with local_assets_only(args.tokenizer):
            cfg = make_config(args.repo_id, args.assets, args.weights, args.checkpoints,
                              wrist_camera=True, training_plan=plan, resume=args.resume)
            identity = {'settings': plan.to_dict(), 'repo_id': args.repo_id,
                        'norm_stats_sha256': hashlib.sha256((args.assets/'original_ur5/norm_stats.json').read_bytes()).hexdigest(),
                        'tokenizer_sha256': hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()}
            identity_path = args.checkpoints/'experiment_specs'/cfg.name/(plan.run_name+'.json')
            if identity_path.exists():
                if not args.resume or json.loads(identity_path.read_text()) != identity:
                    raise ValueError('Experiment identity differs or exists; do not resume with changed settings/data stats')
            else:
                if args.resume:
                    raise FileNotFoundError('Resume requires the original experiment identity')
                identity_path.parent.mkdir(parents=True, exist_ok=True)
                identity_path.write_text(json.dumps(identity, indent=2)+'\n')
            spec = importlib.util.spec_from_file_location('robot_exam_openpi_train', args.openpi_root/'scripts/train.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            import logging
            logging.basicConfig(level=logging.INFO)
            audit = args.output.parent/(args.output.stem+'_metrics')
            audit.mkdir(exist_ok=False)
            report.update(status='RUNNING', training_started=True,
                          devices=[str(device) for device in devices], metrics_directory=str(audit))
            report['tokenizer_sha256'] = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
            report['norm_stats_sha256'] = hashlib.sha256((args.assets/'original_ur5/norm_stats.json').read_bytes()).hexdigest()
            if (ROOT/'.git').exists():
                report['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
            elif (ROOT.parent/'transfer_manifest.json').is_file():
                report['source_commit'] = json.loads((ROOT.parent/'transfer_manifest.json').read_text())['git_commit']
            else:
                raise FileNotFoundError('Code provenance missing: preserve transfer_manifest.json next to robot_exam/')
            save_report()
            run_pinned_training(module, cfg, plan, args.lerobot_cache, audit)
            metrics = audit/'train_metrics.jsonl'
            updates = len(metrics.read_text().splitlines()) if metrics.exists() else 0
            report.update(status='TRAINING_COMPLETED_RELOAD_PENDING' if updates else 'CHECKPOINT_RELOAD_COMPLETED_NO_UPDATES',
                          training_performed=updates > 0, optimizer_updates_this_attempt=updates,
                          weights_loaded=True, checkpoint_restored=args.resume)
    except BaseException as error:
        report.update(status='FAILED_OR_INTERRUPTED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        save_report()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

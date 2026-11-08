#!/usr/bin/env python3
"""Prepare by default; explicit GPU-only teacher-forced action diagnostics.

This never sends actuator commands, updates weights or claims task success.
The public policy.infer(noise=...) interface controls stochastic comparisons.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import re
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--robot-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--openpi-root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--lerobot-cache',type=Path,required=True)
    p.add_argument('--repo-id',default='local/robot_exam_dual_release_train')
    p.add_argument('--frames',type=int,nargs='+',default=[0,30,60])
    p.add_argument('--noise-seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--authorized-gpu-diagnostic',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve diagnostics')
    if any(f<0 for f in a.frames) or len(a.noise_seeds)!=len(set(a.noise_seeds)) or any(s<0 for s in a.noise_seeds):p.error('Invalid diagnostic frames or noise seeds')
    meta=a.lerobot_cache/a.repo_id/'meta/episodes.jsonl'
    episodes=[json.loads(s) for s in meta.read_text().splitlines() if s]
    selection=[];seen=set();offset=0
    for episode in episodes:
        match=re.search(r'Pick up the (apple|banana|orange)',episode['tasks'][0])
        if match and match[1] not in seen:
            for frame in a.frames:
                if frame>=episode['length']:p.error('Frame outside selected episode')
                selection.append({'skill':match[1],'dataset_index':offset+frame,'frame':frame,'dataset_prompt':episode['tasks'][0]})
            seen.add(match[1])
        offset+=int(episode['length'])
    if seen!={'apple','banana','orange'}:p.error('Require all three training classes')
    queries=len(selection)*len(a.noise_seeds)*2
    if queries>144:p.error('Bound diagnostic to144 requests')
    report={'status':'PLAN_ONLY_NO_WEIGHTS_LOADED','checkpoint':str(a.checkpoint),'repo_id':a.repo_id,
            'selection':selection,'noise_seeds':a.noise_seeds,'expected_queries':queries,
            'prompt_conditions':['legacy','green_explicit'],'raw_output_semantics':'10x32, first7 unnormalized robot actions; last25 padded normalized channels',
            'scope':'Teacher-forced training-frame diagnostics, not physical task success or generalization.',
            'model_loaded':False,'training_updates':0,'actuator_commands':0}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
    if not a.authorized_gpu_diagnostic:save();print(json.dumps(report,indent=2));return
    if not glob.glob('/dev/nvidia[0-9]*'):raise RuntimeError('Refuse model loading without GPU; remain plan-only in no-card mode')
    limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    available=int(limit) if limit!='max' else os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    if available<60*1024**3:raise RuntimeError('GPU diagnostic requires the prepared server memory budget, not local8GB/no-card')
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_LEROBOT_HOME=str(a.lerobot_cache.resolve()))
    os.environ.setdefault('HF_DATASETS_CACHE',str(a.lerobot_cache.resolve().parent/'hf_datasets_cache'))
    sys.path.insert(0,str(a.robot_root/'src'));sys.path.insert(0,str(a.openpi_root/'src'))
    import dataclasses
    import numpy as np
    import jax
    from openpi import transforms
    from openpi.training import data_loader
    from openpi.policies.policy_config import create_trained_policy
    from robot_exam.vla.openpi_config import make_config,OriginalUR5DataConfig
    from robot_exam.vla.offline_assets import local_assets_only,verify_checkpoint_statistics
    from robot_exam.vla.lerobot_compat import enable_list_feature_compat
    if not all(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('CPU fallback forbidden')
    verify_checkpoint_statistics(a.assets/'original_ur5/norm_stats.json',a.checkpoint/'assets/original_ur5/norm_stats.json')
    cfg=make_config(a.repo_id,a.assets,a.checkpoint,a.output.parent/'unused_checkpoints',wrist_camera=True)
    @dataclasses.dataclass(frozen=True)
    class RawDiagnosticData(OriginalUR5DataConfig):
        def create(self,assets_dirs,model_config):
            data=super().create(assets_dirs,model_config)
            # Keep native inputs/normalization; inspect raw gripper errors
            # before strict deployment rejection or explicit bounded projection.
            return dataclasses.replace(data,data_transforms=transforms.Group(inputs=data.data_transforms.inputs,outputs=[]))
    factory=RawDiagnosticData(**{f.name:getattr(cfg.data,f.name) for f in dataclasses.fields(cfg.data)})
    cfg=dataclasses.replace(cfg,data=factory)
    enable_list_feature_compat()
    raw_factory=dataclasses.replace(cfg.data,dry_check=True)
    dataset=data_loader.create_torch_dataset(raw_factory.create(cfg.assets_dirs,cfg.model),cfg.model.action_horizon,cfg.model)
    report.update(status='RUNNING',rows=[]);save();started=time.monotonic()
    try:
        with local_assets_only(a.tokenizer):
            policy=create_trained_policy(cfg,a.checkpoint);report['model_loaded']=True
            for sample_spec in selection:
                sample=dict(dataset[sample_spec['dataset_index']]);truth=np.asarray(sample['action'])[:,:7]
                for seed in a.noise_seeds:
                    noise=np.random.default_rng(seed).standard_normal((10,32)).astype(np.float32)
                    for prompt_style in ('legacy','green_explicit'):
                        if time.monotonic()-started>900:raise TimeoutError('Bound diagnostic wall time')
                        skill=sample_spec['skill']
                        prompt=(f'Pick up the {skill} and place it at the target.' if prompt_style=='legacy' else
                                f'Pick up the {skill} and place it on the green circular target marker on the table.')
                        obs={'state':np.asarray(sample['observation.state']),'front_rgb':np.asarray(sample['observation.images.front']),
                             'wrist_rgb':np.asarray(sample['observation.images.wrist']),'prompt':prompt}
                        out=policy.infer(obs,noise=noise);actions=np.asarray(out['actions']);assert actions.shape==(10,32)
                        predicted=actions[:,:7];assert np.isfinite(predicted).all()
                        report['rows'].append({**sample_spec,'prompt_style':prompt_style,'prompt':prompt,'noise_seed':seed,
                            'arm_mae_rad_per_joint':np.mean(np.abs(predicted[:,:6]-truth[:,:6]),axis=0).tolist(),
                            'first_arm_absolute_error_rad':np.abs(predicted[0,:6]-truth[0,:6]).tolist(),
                            'gripper_mae':float(np.mean(np.abs(predicted[:,6]-truth[:,6]))),
                            'gripper_binary_accuracy':float(np.mean((predicted[:,6]>.5)==(truth[:,6]>.5))),
                            'raw_gripper_outside_fraction':float(np.mean((predicted[:,6]<0)|(predicted[:,6]>1))),
                            'predicted_first7':predicted.tolist(),'teacher_actions':truth.tolist(),
                            'padded25_absolute_mean':float(np.mean(np.abs(actions[:,7:])))})
                        save()
        report.update(status='COMPLETED_OFFLINE_DIAGNOSTICS_NOT_PHYSICAL_SUCCESS',wall_seconds=time.monotonic()-started)
    except Exception as exc:
        report.update(status='FAIL',error={'type':type(exc).__name__,'message':str(exc)})
        save();raise
    save();print(json.dumps({'status':report['status'],'queries':len(report['rows']),'training_updates':0}))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Prepared inference server; requires a real user-provided trained checkpoint."""
import argparse
import os
import json
import time
from pathlib import Path
import sys
from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--openpi-root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--port',type=int,default=8000);p.add_argument('--serve',action='store_true')
    p.add_argument('--repo-id',default='local/robot_exam_dual_release_train');p.add_argument('--wrist-camera',action='store_true')
    p.add_argument('--tokenizer',type=Path);p.add_argument('--lerobot-cache',type=Path)
    a=p.parse_args()
    os.environ['HF_HUB_OFFLINE']=os.environ['HF_DATASETS_OFFLINE']='1'
    if a.lerobot_cache:
        os.environ['HF_LEROBOT_HOME']=str(a.lerobot_cache.resolve())
        os.environ.setdefault('HF_DATASETS_CACHE',str(a.lerobot_cache.resolve().parent/'hf_datasets_cache'))
    sys.path.insert(0,str(a.openpi_root/'src'))
    from robot_exam.vla.openpi_config import make_config
    cfg=make_config(a.repo_id,a.assets,a.checkpoint,ROOT/'checkpoints',dry_check=not a.serve,wrist_camera=a.wrist_camera)
    if not a.serve:
        print({'status':'configuration_only','model_loaded':False,'action_semantics':'ur5_absolute_joint_goal_v1'})
        return
    if not a.checkpoint.is_dir():raise FileNotFoundError('A local trained checkpoint is required; no automatic download')
    if not a.tokenizer or not a.lerobot_cache:p.error('Serving requires local --tokenizer and --lerobot-cache for real-data warmup')
    from robot_exam.vla.lerobot_compat import enable_list_feature_compat
    enable_list_feature_compat()
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from robot_exam.vla.offline_assets import local_assets_only,verify_checkpoint_statistics
    verify_checkpoint_statistics(a.assets/'original_ur5/norm_stats.json',a.checkpoint/'assets/original_ur5/norm_stats.json')
    with local_assets_only(a.tokenizer):
        policy=create_trained_policy(cfg,a.checkpoint)
        # Compile using a real train observation before admitting timed control requests.
        import dataclasses
        import numpy as np
        from openpi.training import data_loader
        factory=dataclasses.replace(cfg.data,dry_check=True)
        dataset=data_loader.create_torch_dataset(factory.create(cfg.assets_dirs,cfg.model),cfg.model.action_horizon,cfg.model)
        sample=dict(dataset[0]);obs={'state':np.asarray(sample['observation.state']),
            'front_rgb':np.asarray(sample['observation.images.front']),'prompt':str(sample['prompt'])}
        if a.wrist_camera:obs['wrist_rgb']=np.asarray(sample['observation.images.wrist'])
        start=time.monotonic();prediction=policy.infer(obs)
        print(json.dumps({'status':'REAL_MODEL_WARMUP_COMPLETE','seconds':time.monotonic()-start,
            'action_shape':list(prediction['actions'].shape),'physical_success':'NOT_EVALUATED'}),flush=True)
        WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=cfg.policy_metadata).serve_forever()


if __name__=='__main__':main()

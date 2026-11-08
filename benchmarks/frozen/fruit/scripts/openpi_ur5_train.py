#!/usr/bin/env python3
"""Check local config by default; training requires a deliberate explicit flag."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--openpi-root',type=Path,default=Path('/home/li/vla/openpi'))
    p.add_argument('--repo-id',default='local/robot_exam_schema3_train')
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints/openpi')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--authorized-training',action='store_true')
    p.add_argument('--wrist-camera',action='store_true')
    a=p.parse_args()
    sys.path.insert(0,str(a.openpi_root/'src'))
    from robot_exam.vla.lerobot_compat import enable_list_feature_compat
    enable_list_feature_compat()
    from robot_exam.vla.openpi_config import make_config
    cfg=make_config(a.repo_id,a.assets,a.weights,a.checkpoints,dry_check=not a.authorized_training,wrist_camera=a.wrist_camera)
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    report={'config':cfg.name,'robot':'original_ur5','action_dim_padded':cfg.model.action_dim,
            'action_horizon':cfg.model.action_horizon,'policy_hz':5,
            'actual_action_dim':7,'lora':True,'training_performed':False,
            'norm_stats_loaded':data.norm_stats is not None,'weights_exist':a.weights.exists(),
            'actual_camera_count':2 if a.wrist_camera else 1,'delta_transform':False,'wandb_enabled':cfg.wandb_enabled}
    if a.authorized_training:
        if not a.weights.exists() or data.norm_stats is None:
            raise RuntimeError('Local weights and original-UR5 training-only statistics are required')
        spec=importlib.util.spec_from_file_location('robot_exam_openpi_train',a.openpi_root/'scripts/train.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.main(cfg);report['training_performed']=True
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

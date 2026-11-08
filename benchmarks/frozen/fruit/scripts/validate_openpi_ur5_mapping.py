#!/usr/bin/env python3
"""Native OpenPI repack/normalize/pad/output checks, with no model allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image

from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--openpi-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lerobot-cache',type=Path);p.add_argument('--repo-id',default='local/robot_exam_schema3_train')
    p.add_argument('--wrist-camera',action='store_true');a=p.parse_args()
    if a.lerobot_cache:os.environ['HF_LEROBOT_HOME']=str(a.lerobot_cache.resolve())
    os.environ.setdefault('HF_DATASETS_CACHE','/tmp/robot_exam_openpi_native_cache')
    sys.path.insert(0,str(a.openpi_root/'src'))
    from openpi import transforms
    from robot_exam.vla.openpi_config import make_config
    manifest=json.loads((a.manifest/'manifest.json').read_text())
    episode=Path(next(x['path'] for x in manifest['accepted'] if x['split']=='train'))
    frames=[json.loads(line) for line in (episode/'frames.jsonl').read_text().splitlines()]
    cfg=make_config(a.repo_id,a.manifest/'assets',Path('/tmp/no_model_weights'),
                    ROOT/'checkpoints',dry_check=True,wrist_camera=a.wrist_camera)
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    sample={'observation.state':np.asarray(frames[0]['observation.state']),
            'observation.images.front':np.asarray(Image.open(episode/frames[0]['rgb'])),
            'action':np.asarray([r['action'] for r in frames[:10]]),'prompt':frames[0]['task']}
    if a.wrist_camera:sample['observation.images.wrist']=np.asarray(Image.open(episode/frames[0]['rgb_wrist']))
    for transform in (*data.repack_transforms.inputs,*data.data_transforms.inputs):sample=transform(sample)
    original=sample['actions'].copy()
    normalized=transforms.Normalize(data.norm_stats,use_quantiles=data.use_quantile_norm)(sample)
    padded=transforms.PadStatesAndActions(cfg.model.action_dim)(normalized)
    assert padded['actions'].shape==(10,32) and padded['state'].shape==(32,)
    assert padded['image_mask']=={'base_0_rgb':True,'left_wrist_0_rgb':a.wrist_camera,'right_wrist_0_rgb':False}
    restored=transforms.Unnormalize(data.norm_stats,use_quantiles=data.use_quantile_norm)(padded)
    for transform in data.data_transforms.outputs:restored=transform(restored)
    error=float(np.max(np.abs(restored['actions']-original)))
    assert error<1e-5 and data.norm_stats is not None
    files=['examples/ur5/README.md','src/openpi/training/config.py','src/openpi/models/pi0_config.py',
           'src/openpi/transforms.py','src/openpi/policies/policy_config.py','pyproject.toml']
    report={'status':'PASS','scope':'native transforms and LoRA config only; no model inference/training/tokenizer fetch',
            'action_semantics':'ur5_absolute_joint_goal_v1','roundtrip_error':error,
            'padded_state_shape':[32],'padded_action_shape':[10,32],'actual_action_shape':[10,7],
            'norm_stats_source':'training split only, asset original_ur5','delta_transform':False,
            'camera_mask':{k:bool(v) for k,v in padded['image_mask'].items()},
            'openpi_source_root':str(a.openpi_root.resolve()),'upstream_git_commit':None,
            'snapshot_note':'Local source has no .git; copy and verify these file hashes on AutoDL',
            'source_hashes':{name:hashlib.sha256((a.openpi_root/name).read_bytes()).hexdigest() for name in files},
            'training_performed':False,'weights_loaded':False}
    if a.lerobot_cache:
        from robot_exam.vla.lerobot_compat import enable_list_feature_compat
        report['hf_feature_compatibility']=enable_list_feature_compat()
        from openpi.training import data_loader
        dataset=data_loader.create_torch_dataset(data,cfg.model.action_horizon,cfg.model)
        transformed=data_loader.transform_dataset(dataset,data)
        sample=transformed[0]
        assert sample['actions'].shape==(10,7)
        assert sample['image']['base_0_rgb'].shape[-1]==3
        report['native_lerobot_training_loader']={'status':'PASS','frames':len(dataset),
            'action_chunk_shape':list(sample['actions'].shape),'actual_model_weights':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

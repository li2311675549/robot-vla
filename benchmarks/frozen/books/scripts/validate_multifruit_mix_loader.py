#!/usr/bin/env python3
"""Native two-source OpenPI loader/sampler probe without tokenizer or weights."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--new-dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve native loader probes')
    a.output.mkdir(parents=True);cache=a.output.resolve()/'cache_aliases/local';cache.mkdir(parents=True)
    sources={'old':ROOT/'outputs/stage15/data_preparation/green_target_v2/full/train','new':a.new_dataset.resolve()}
    for source,path in sources.items():(cache/source).symlink_to(path,target_is_directory=True)
    os.environ['HF_LEROBOT_HOME']=str(cache.parent)
    os.environ.setdefault('HF_DATASETS_CACHE','/tmp/robot_exam_mix_native_cache')
    sys.path.insert(0,'/home/li/vla/openpi/src')
    import torch
    from openpi import transforms
    from openpi.training import data_loader
    from robot_exam.vla.lerobot_compat import enable_list_feature_compat
    from robot_exam.vla.openpi_config import make_config
    enable_list_feature_compat()
    prep=json.loads((a.preparation/'preparation.json').read_text());raw=np.load(a.preparation/'sampling_weights.npz')
    datasets=[];prompts={};checks=[]
    for source in sources:
        cfg=make_config(f'local/{source}',a.preparation/'assets','/tmp/no_weights','/tmp/no_checkpoints',dry_check=True,wrist_camera=True)
        dc=cfg.data.create(cfg.assets_dirs,cfg.model)
        native=data_loader.create_torch_dataset(dc,10,cfg.model)
        mapped=data_loader.transform_dataset(native,dc)
        prompts[source]=[mapped[0]['prompt'],mapped[len(mapped)-1]['prompt']]
        datasets.append(mapped)
    combined=torch.utils.data.ConcatDataset(datasets)
    if len(combined)!=len(raw['weights']):raise ValueError('Dataset index/weight mapping mismatch')
    # Preserve natural language outside this numeric batching check. Real
    # training still uses the real authorized tokenizer; no fake token IDs.
    class NumericOnly:
        def __len__(self):return len(combined)
        def __getitem__(self,index):
            row=dict(combined[index]);row.pop('prompt',None)
            row=transforms.ResizeImages(224,224)(row)
            return transforms.PadStatesAndActions(32)(row)
    sampler=torch.utils.data.WeightedRandomSampler(torch.from_numpy(raw['weights']),8,replacement=True,
        generator=torch.Generator().manual_seed(51))
    loader=data_loader.TorchDataLoader(NumericOnly(),2,sampler=sampler,num_batches=4,num_workers=0,framework='pytorch')
    for batch in loader:
        assert tuple(batch['state'].shape)==(2,32) and tuple(batch['actions'].shape)==(2,10,32)
        assert torch.isfinite(batch['state']).all() and torch.isfinite(batch['actions']).all()
        assert batch['image_mask']['base_0_rgb'].all() and batch['image_mask']['left_wrist_0_rgb'].all()
        assert not batch['image_mask']['right_wrist_0_rgb'].any()
        checks.append({'state':list(batch['state'].shape),'actions':list(batch['actions'].shape)})
    report={'status':'PASS','scope':'Native OpenPI LeRobot+UR5 transforms+TorchDataLoader+weighted sampler; no model or tokenizer',
        'frames':len(combined),'batches':checks,'source_frames':dict(zip(sources,map(len,datasets))),
        'prompt_examples':prompts,'action_semantics':'ur5_absolute_joint_goal_v1',
        'norm_stats':str((a.preparation/'assets/original_ur5/norm_stats.json').resolve()),
        'tokenization_tested':False,'weights_loaded':False,'training':False,'training_ready':False}
    (a.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()

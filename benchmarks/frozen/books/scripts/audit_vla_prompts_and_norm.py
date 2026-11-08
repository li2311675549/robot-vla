#!/usr/bin/env python3
"""Independently verify every released language label and train-only norm asset.

Uses the existing OpenPI environment for pyarrow/native transforms; no images,
model weights, training, dataset mutation or network access are required.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from _bootstrap import ROOT


def load(path):
    return json.loads(path.read_text())


def statistics(values):
    return {'mean':values.mean(0),'std':values.std(0),
            'q01':np.quantile(values,.01,axis=0),'q99':np.quantile(values,.99,axis=0)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--openpi-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing audits')
    m=load(a.release/'manifest/manifest.json')
    stats_path=a.release/'manifest/assets/original_ur5/norm_stats.json'
    saved=load(stats_path)['norm_stats']
    raw_train={'state':[],'actions':[]};export_train={'state':[],'actions':[]}
    counts={};prompts=Counter();sources={e['path']:e for e in m['accepted']}
    maximum_export_difference=0.;verified_frames=0
    for split in ('train','validation','test'):
        root=a.release/f'lerobot_dual_{split}_v1'
        conversion=load(root/'conversion_result.json')
        source_map={s['episode']:s for s in conversion['sources']}
        tasks={t['task_index']:t['task'] for t in map(json.loads,(root/'meta/tasks.jsonl').read_text().splitlines())}
        counts[split]={'episodes':0,'frames':0,'by_class':Counter()}
        for file in sorted(root.glob('data/**/*.parquet')):
            table=pq.read_table(file,columns=['observation.state','action','task_index','episode_index',
                                             'frame_index','episode_success'])
            episode_ids=table['episode_index'].to_pylist()
            assert len(set(episode_ids))==1
            source=source_map[episode_ids[0]];entry=sources[source['raw_path']]
            assert entry['split']==split and entry['seed'] in m['protocol'][split]
            folder=Path(source['raw_path']);meta=load(folder/'episode.json')
            expected=f"Pick up the {entry['skill']} and place it at the target."
            assert meta['task']==meta['language_instruction']==expected
            assert meta['success'] and meta['schema_version']==3
            assert meta['action_semantics']=='ur5_absolute_joint_goal_v1'
            frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
            assert len(frames)==len(table)==entry['frames']
            assert table['frame_index'].to_pylist()==list(range(len(frames)))
            assert all(table['episode_success'].to_pylist())
            assert all(tasks[k]==expected for k in table['task_index'].to_pylist())
            assert all(row['task']==expected for row in frames)
            raw={key:np.asarray([f[field] for f in frames],dtype=float)
                 for key,field in [('state','observation.state'),('actions','action')]}
            exported={key:np.asarray(table[field].to_pylist(),dtype=float)
                      for key,field in [('state','observation.state'),('actions','action')]}
            for key in raw:
                assert raw[key].shape==(len(frames),7) and np.isfinite(raw[key]).all()
                assert np.array_equal(raw[key].astype(np.float32),exported[key].astype(np.float32))
                maximum_export_difference=max(maximum_export_difference,float(np.max(np.abs(raw[key]-exported[key]))))
                if split=='train':
                    raw_train[key].append(raw[key]);export_train[key].append(exported[key])
            prompts[expected]+=len(frames);verified_frames+=len(frames)
            counts[split]['episodes']+=1;counts[split]['frames']+=len(frames)
            counts[split]['by_class'][entry['skill']]+=1
    raw_errors={};export_errors={};ranges={}
    for key in raw_train:
        raw_train[key]=np.concatenate(raw_train[key]);export_train[key]=np.concatenate(export_train[key])
        raw_errors[key]={name:float(np.max(np.abs(value-saved[key][name])))
                         for name,value in statistics(raw_train[key]).items()}
        export_errors[key]={name:float(np.max(np.abs(value-saved[key][name])))
                            for name,value in statistics(export_train[key]).items()}
        assert max(raw_errors[key].values())<1e-12
        assert min(np.asarray(saved[key]['q99'])-saved[key]['q01'])>0
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',JAX_PLATFORMS='cpu')
    sys.path.insert(0,str(a.openpi_root/'src'))
    from openpi import transforms
    from robot_exam.vla.openpi_config import make_config
    cfg=make_config('local/robot_exam_dual_release_train',a.release/'manifest/assets',Path('/tmp/no_weights'),
                    Path('/tmp/no_checkpoints'),dry_check=True,wrist_camera=True)
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    assert data.use_quantile_norm and data.prompt_from_task
    normalized=transforms.Normalize(data.norm_stats,use_quantiles=data.use_quantile_norm)(export_train)
    restored=transforms.Unnormalize(data.norm_stats,use_quantiles=data.use_quantile_norm)(normalized)
    roundtrip={key:float(np.max(np.abs(restored[key]-export_train[key]))) for key in export_train}
    assert max(roundtrip.values())<1e-6
    for key in normalized:
        assert np.isfinite(normalized[key]).all()
        ranges[key]={'min_per_dimension':normalized[key].min(0).tolist(),'max_per_dimension':normalized[key].max(0).tolist(),
                     'q99_minus_q01':(np.asarray(saved[key]['q99'])-saved[key]['q01']).tolist()}
    report={'status':'PASS_DATA_INTEGRITY_NOT_POLICY_SUCCESS','episodes':len(m['accepted']),
            'frames_verified':verified_frames,'counts':counts,'prompt_frame_counts':dict(prompts),
            'prompt_mismatches':0,'all_exported_state_actions_exact_after_float32_cast':True,
            'maximum_float64_to_export_float32_difference':maximum_export_difference,
            'train_only_stats_raw_recomputation_errors':raw_errors,
            'train_only_export_float32_stats_differences':export_errors,
            'norm_stats_sha256':hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            'norm_stats_content_sha256':hashlib.sha256(json.dumps(load(stats_path),sort_keys=True).encode()).hexdigest(),
            'native_openpi_quantile_normalization':bool(data.use_quantile_norm),
            'native_prompt_from_task':bool(data.prompt_from_task),'native_roundtrip_max_errors':roundtrip,
            'normalized_ranges':ranges,'statistics_used_validation_or_test':False,
            'limitations':['Three repeated task templates; no linguistic variation test.',
                           'The target wording assumes the single visible designated target in these scenes.',
                           'Near-fixed wrist_2 has narrow quantiles; this is limited pose coverage, not zero-width norm.',
                           'Integrity/replay are necessary but do not prove sufficient learning coverage or trained policy success.'],
            'training_updates':0,'weights_loaded':False,'dataset_modified':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='normalized_ranges'},indent=2))


if __name__=='__main__':main()

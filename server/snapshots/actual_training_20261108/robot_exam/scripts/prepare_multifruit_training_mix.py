#!/usr/bin/env python3
"""Offline actual-data sampler/statistics pilot, with no training/model fetch.

Run in the existing OpenPI environment with pyarrow/torch installed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from _bootstrap import ROOT
from robot_exam.data.mixing import frame_weights,chunk_statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--new-manifest',type=Path,required=True)
    p.add_argument('--new-dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-dataset',type=Path,default=ROOT/'outputs/stage15/data_preparation/green_target_v2/full/train')
    p.add_argument('--old-mapping',type=Path,default=ROOT/'outputs/stage11/dual_expansion_release_v1/lerobot_dual_train_v1/conversion_result.json')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior mixture preparations')
    manifest=json.loads(a.new_manifest.read_text())
    if manifest.get('non_target_preservation_required') is not True:
        raise ValueError('Require strict clean-parent manifest')
    records=manifest['accepted']
    if any(r['split']!='train' for r in records):raise ValueError('Train-only preparation')
    old_mapping=json.loads(a.old_mapping.read_text())['sources']
    new_mapping=json.loads((a.new_dataset/'conversion_result.json').read_text())['sources']
    by_path={str(Path(r['path']).resolve()):r for r in records}
    episodes=[];states=[];actions=[];files=[]
    for source,root,mapping in [('old',a.old_dataset,old_mapping),('new',a.new_dataset,new_mapping)]:
        info=json.loads((root/'meta/info.json').read_text())
        if info['fps']!=5 or info['total_episodes']!=len(mapping):raise ValueError('Unexpected dataset frequency/episode count')
        for row in mapping:
            episode=row['episode'];q=root/info['data_path'].format(episode_chunk=episode//info['chunks_size'],episode_index=episode)
            table=pq.read_table(q,columns=['observation.state','action','episode_index','timestamp'])
            state=np.asarray(table['observation.state'].to_pylist());action=np.asarray(table['action'].to_pylist())
            if not np.all(np.asarray(table['episode_index'])==episode):raise ValueError('Episode mapping mismatch')
            if not np.allclose(np.diff(np.asarray(table['timestamp'])),.2,atol=1e-5):raise ValueError('Action frequency mismatch')
            if row['action_semantics']!='ur5_absolute_joint_goal_v1':raise ValueError('Action semantics mismatch')
            if not np.isfinite(action).all() or np.any(action[:,6]<0) or np.any(action[:,6]>1):raise ValueError('Invalid action')
            record=by_path[str(Path(row['raw_path']).resolve())] if source=='new' else None
            episodes.append(dict(source=source,split='train',episode=episode,
                scene_context_id=record['scene_context_id'] if record else f'old_{row["seed"]}',
                seed=row['seed'],category=record['skill'] if record else row['skill'],phase=record['phase'] if record else 'whole_pick_place',
                frames=len(state),parquet=str(q.resolve())))
            states.append(state);actions.append(action)
            files.append({'path':str(q.resolve()),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
    weights,offsets,masses=frame_weights(episodes)
    stats=chunk_statistics(episodes,np.concatenate(states),np.concatenate(actions),horizon=10)
    sampler=torch.utils.data.WeightedRandomSampler(torch.from_numpy(weights),20000,replacement=True,
                                                  generator=torch.Generator().manual_seed(20261105))
    draws=np.asarray(list(sampler));which=np.searchsorted(offsets[1:],draws,side='right')
    sources=Counter(episodes[i]['source'] for i in which)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'sampling_weights.npz',weights=weights,episode_offsets=offsets)
    assets=a.output/'assets/original_ur5';assets.mkdir(parents=True)
    (assets/'norm_stats.json').write_text(json.dumps({'norm_stats':stats},indent=2))
    for i,e in enumerate(episodes):e['sampling_probability']=masses[i]
    report={'status':'LOCAL_MIXTURE_PILOT_PASS_NOT_TRAINING_RELEASE','training_ready':False,
        'weights_loaded':False,'training_performed':False,'server_contacted':False,
        'sampling':'Torch WeightedRandomSampler; source20/80 then class/phase/episode/frame balanced',
        'action_semantics':'ur5_absolute_joint_goal_v1','fps':5,'action_horizon':10,
        'normalization':'Training partitions only; state by sampled anchors, action by matching10-step LeRobot within-episode clamped chunks',
        'source_mass':{s:sum(masses[i] for i,e in enumerate(episodes) if e['source']==s) for s in ('old','new')},
        'sampled20000_source_counts':dict(sources),'episodes':episodes,'frames':len(weights),
        'input_files':files,'old_prompt':'Explicit green circular target on table',
        'new_prompts':'Specified fruit relation grasp, or named category-bin placement; not mixed with green-target instruction',
        'pending':['Qualified new validation/ID-test/OOD coverage','Larger independent training-scene coverage',
                   'Weighted native-loader integration check','Future explicit upload/GPU/training authorization']}
    (a.output/'preparation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('status','frames','source_mass','sampled20000_source_counts')},indent=2))


if __name__=='__main__':main()

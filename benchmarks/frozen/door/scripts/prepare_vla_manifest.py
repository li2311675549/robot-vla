#!/usr/bin/env python3
"""Split only schema3 positive episodes and compute training-only statistics."""
import argparse
import json
from pathlib import Path
import numpy as np

from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--splits',type=Path,default=ROOT/'configs/stage9_data_splits.json')
    p.add_argument('--additional-source',type=Path,action='append',default=[])
    p.add_argument('--skills',nargs='+');p.add_argument('--require-executor-success',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new manifest directory')
    splits=json.loads(a.splits.read_text());assignment={}
    for split in ('train','validation','test'):
        for seed in splits[split]:
            if seed in assignment:raise ValueError('Seed leaks across splits')
            assignment[seed]=split
    accepted=[];excluded=[];states=[];actions=[]
    files={file.resolve() for root in [a.source,*a.additional_source] for file in root.rglob('episode.json')}
    for file in sorted(files):
        meta=json.loads(file.read_text());folder=file.parent
        if a.skills and meta.get('skill') not in a.skills:continue
        validation_path=folder/'validation.json'
        validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
        reasons=[]
        if not meta.get('success'):reasons.append('physical_failure')
        if meta.get('schema_version')!=3 or meta.get('action_semantics')!=splits['action_semantics']:reasons.append('action_semantics')
        if splits.get('sensor_schema') and meta.get('sensor_schema')!=splits['sensor_schema']:
            reasons.append('sensor_schema')
        if splits.get('base_control') and meta.get('base_control')!=splits['base_control']:
            reasons.append('base_control')
        if splits.get('wrist_mount_position_tool_m'):
            if not np.allclose(meta.get('wrist_mount_position_tool_m',[999,999,999]),splits['wrist_mount_position_tool_m']):
                reasons.append('wrist_mount_version')
        if validation.get('status')!='PASS' or validation.get('max_replay_qpos_error') is None:reasons.append('quality_or_full_rate_replay')
        low=folder/'low_rate_replay.json'
        low_result=json.loads(low.read_text()) if low.exists() else {}
        if not low_result.get('low_rate_physical_placement_pass'):reasons.append('low_rate_physical_replay')
        if a.require_executor_success:
            executed=folder/'vla_executor_replay.json'
            if not executed.exists() or not json.loads(executed.read_text()).get('success'):
                reasons.append('deployment_executor_physical_replay')
        if meta['seed'] not in assignment:reasons.append('seed_not_in_protocol')
        entry={'path':str(folder.resolve()),'seed':meta['seed'],'skill':meta.get('skill'),'frames':meta['frames'],
               'scale':meta.get('fixture_scale'),'expert_source':meta['expert_source']}
        entry.update(sensor_schema=meta.get('sensor_schema','front_rgbd_v1'),base_control=meta.get('base_control'))
        if reasons:excluded.append({**entry,'reasons':reasons});continue
        entry['split']=assignment[meta['seed']]
        entry['low_rate_physical_placement_pass']=low_result.get('low_rate_physical_placement_pass')
        accepted.append(entry)
        if entry['split']=='train':
            rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
            states.extend(row['observation.state'] for row in rows);actions.extend(row['action'] for row in rows)
    if not states:raise RuntimeError('No valid training episodes')
    stats={}
    for name,values in [('state',states),('actions',actions)]:
        values=np.asarray(values,dtype=float)
        stats[name]={'mean':values.mean(axis=0).tolist(),'std':values.std(axis=0).tolist(),
                     'q01':np.quantile(values,.01,axis=0).tolist(),'q99':np.quantile(values,.99,axis=0).tolist()}
    asset=a.output/'assets/original_ur5';asset.mkdir(parents=True)
    (asset/'norm_stats.json').write_text(json.dumps({'norm_stats':stats},indent=2)+'\n')
    report={'protocol':splits,'accepted':accepted,'excluded':excluded,'normalization_split':'train',
            'successful_episodes':len(accepted),'frames':sum(x['frames'] for x in accepted),
            'split_counts':{k:sum(x['split']==k for x in accepted) for k in ('train','validation','test')},
            'training_performed':False,'model_weights_downloaded':False}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('successful_episodes','frames','split_counts')},indent=2))


if __name__=='__main__':main()

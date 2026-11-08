#!/usr/bin/env python3
"""Freeze class-balanced held-out phase tests before loading any learned model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def balanced(rows, counts):
    selected=[]
    for count in counts:
        candidates=[r for r in rows if r['scene_object_count']==count and r not in selected]
        if not candidates:
            raise ValueError(f'Insufficient independent fixtures for object_count={count}')
        selected.append(sorted(candidates,key=lambda r:(r['seed'],r['fixture']))[0])
    return selected


def prepare(root):
    root=Path(root).resolve();candidates=[];hashes={}
    def sha(path):
        stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in hashes:
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
            hashes[key]=h.hexdigest()
        return hashes[key]
    for split in ('validation','test'):
        # Include saved initializations from expert failures, not just the
        # successful-demonstration manifest. Never select by expert outcome.
        for meta_path in sorted((root/f'evaluation_fixtures/{split}').glob('*/*/grasp_initialization/episode.json')):
            meta=json.loads(meta_path.read_text());directory=meta_path.parent
            result=json.loads((directory.parent/'result.json').read_text())
            if not meta.get('parking_ready_at_start') or meta['parking_stable_seconds']<2-1e-6:
                raise ValueError('Unqualified initial parking in frozen fixture registry')
            candidates.append({'split':split,'phase':'grasp','category':meta['category'],
                'seed':meta['seed'],'scene_object_count':meta['scene_object_count'],
                'fixture':str(directory.relative_to(root)), 'expert_parent_status':result['status'],
                'sha256':{name:sha(directory/name) for name in ('episode.json','model.mjb','replay.npz')}})
    manifest=json.loads((root/'manifest_new_portable.json').read_text())
    for row in manifest['accepted']:
        if row['split'] not in ('validation','test') or row['phase']!='place':continue
        directory=root/row['path'];meta=json.loads((directory/'episode.json').read_text())
        candidates.append({'split':row['split'],'phase':'place','category':meta['category'],
            'seed':meta['seed'],'scene_object_count':meta['scene_object_count'],
            'fixture':str(directory.relative_to(root)),
            'scope':'Conditional release from physically held expert initialization, not autonomous grasp+place',
            'sha256':{name:sha(directory/name) for name in ('episode.json','model.mjb','replay.npz')}})
    matrices={}
    for split,counts in [('validation',[3,5,10]),('test',[3,5,10,10,3,5,10,10,10,10])]:
        for phase in ('grasp','place'):
            rows=[]
            for category in ('apple','banana','orange'):
                pool=[r for r in candidates if r['split']==split and r['phase']==phase and r['category']==category]
                rows+=balanced(pool,counts)
            matrices[f'{split}/{phase}']=rows
    return {'protocol':'parked_multifruit_vla_class_grasp_v1',
        'scope':'One model, three fruit classes, real parked mixed scenes; not complete10fruit task',
        'phase_budget_seconds':30.,'policy_noise_seed':0,'execute_prefix_actions':1,
        'action_horizon':10,'policy_hz':5,'network_image_size':224,'max_response_seconds':2.,
        'normalization_sha256':sha(root/'assets/original_ur5/norm_stats.json'),
        'selection':'Predeclared count strata then increasing seed; NEVER by expert/model outcome',
        'test_locked_until_validation_checkpoint_selection':True,
        'model_inference_performed':False,'expert_failure_initializations_in_candidate_pool':True,
        'candidate_counts':dict(Counter(f"{r['split']}/{r['phase']}/{r['category']}" for r in candidates)),
        'candidate_registry':candidates,'matrices':matrices}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Freeze a new protocol version instead of replacing one')
    report=prepare(a.dataset_root);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:len(v) for k,v in report['matrices'].items()}))

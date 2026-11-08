#!/usr/bin/env python3
"""Export through the already-installed LeRobot v2.1 API (separate openpi env).

Run with ~/anaconda3/envs/openpi/bin/python. No installation, upload, or network.
Depth and full-rate replay remain in raw episodes as sidecars.
"""
import argparse,json,os
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--validate-only',action='store_true');p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--require-schema3',action='store_true');p.add_argument('--require-lowrate-success',action='store_true');p.add_argument('--repo-id',default='local/robot_exam_stage34')
    p.add_argument('--manifest',type=Path);p.add_argument('--split',choices=['train','validation','test']);a=p.parse_args()
    a.source=a.source.resolve();a.output=a.output.resolve()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    # Some shared machines expose the user's HF cache read-only. Keep format
    # validation local/offline and place the transient lock/cache under /tmp.
    os.environ.setdefault('HF_DATASETS_CACHE','/tmp/robot_exam_hf_datasets_cache')
    Path(os.environ['HF_DATASETS_CACHE']).mkdir(parents=True,exist_ok=True)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset,CODEBASE_VERSION
    if CODEBASE_VERSION!='v2.1':raise RuntimeError(f'Expected tested local v2.1, got {CODEBASE_VERSION}')
    if a.output.exists() and not a.validate_only:raise RuntimeError('Refusing to overwrite an existing exported dataset')
    eligible=[];semantics=set();sensors=set()
    if a.manifest:
        manifest=json.loads(a.manifest.read_text())
        metadata_paths=[Path(row['path'])/'episode.json' for row in manifest['accepted']
                        if a.split is None or row['split']==a.split]
    else:metadata_paths=sorted(a.source.rglob('episode.json'))
    for metadata_path in ([] if a.validate_only else metadata_paths):
        folder=metadata_path.parent;meta=json.loads(metadata_path.read_text())
        if a.seeds is not None and meta['seed'] not in a.seeds:continue
        if a.require_schema3 and (meta.get('schema_version')!=3 or meta.get('action_semantics')!='ur5_absolute_joint_goal_v1'):continue
        validation_path=folder/'validation.json'
        if not validation_path.exists():continue
        validation=json.loads(validation_path.read_text())
        if not meta['success'] or validation['status']!='PASS':continue
        if a.require_schema3 and validation.get('max_replay_qpos_error') is None:continue
        if a.require_lowrate_success:
            lowrate=folder/'low_rate_replay.json'
            if not lowrate.exists() or not json.loads(lowrate.read_text()).get('low_rate_physical_placement_pass'):continue
        eligible.append((folder,meta));semantics.add(meta.get('action_semantics','legacy_action_semantics_unresolved'))
        sensors.add(meta.get('sensor_schema','front_rgbd_v1'))
    if len(semantics)>1:raise RuntimeError('Refusing to mix action semantics in a dataset')
    if len(sensors)>1:raise RuntimeError('Refusing to mix single/dual camera schemas in a dataset')
    if not a.validate_only and not eligible:raise RuntimeError('No validated physically successful episodes')
    names=['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3','gripper']
    features={'observation.images.front':{'dtype':'image','shape':(480,640,3),'names':['height','width','channels']},
                  'observation.state':{'dtype':'float32','shape':(7,),'names':names},
                  'action':{'dtype':'float32','shape':(7,),'names':names},
                  'episode_success':{'dtype':'bool','shape':(1,),'names':['success']}}
    dual=sensors=={'front_wrist_rgb_v1'}
    if dual:features['observation.images.wrist']={'dtype':'image','shape':(480,640,3),'names':['height','width','channels']}
    dataset=None if a.validate_only else LeRobotDataset.create(a.repo_id,root=a.output,fps=5,
        robot_type='husky_original_ur5_2f85',use_videos=False,features=features)
    sources=[]
    for folder,meta in eligible:
        frames=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        for row in frames:
            frame={'observation.images.front':np.array(Image.open(folder/row['rgb'])),
                               'observation.state':np.asarray(row['observation.state'],dtype=np.float32),
                               'action':np.asarray(row['action'],dtype=np.float32),
                               'episode_success':np.asarray([True]),'task':row['task']}
            if dual:frame['observation.images.wrist']=np.array(Image.open(folder/row['rgb_wrist']))
            dataset.add_frame(frame)
        dataset.save_episode();sources.append({'episode':len(sources),'raw_path':str(folder),'expert_source':meta['expert_source'],'seed':meta['seed'],
            'action_semantics':meta.get('action_semantics','legacy_action_semantics_unresolved'),'skill':meta.get('skill'),
            'action_execution_contract':meta.get('action_execution_contract','legacy_arm_slew_v1')})
    # Local LeRobot v2.1 predates datasets.Column: adapt string-column access in
    # this process only. No installed environment/library files are modified.
    original_loader=LeRobotDataset.load_hf_dataset
    class ColumnCompat:
        def __init__(self,dataset):self.dataset=dataset
        def __getattr__(self,key):return getattr(self.dataset,key)
        def __len__(self):return len(self.dataset)
        def __getitem__(self,key):
            value=self.dataset[key]
            return list(value) if isinstance(key,str) else value
    def load_compatible(self):return ColumnCompat(original_loader(self))
    LeRobotDataset.load_hf_dataset=load_compatible
    loaded=LeRobotDataset(a.repo_id,root=a.output,download_videos=False)
    sample=loaded[0]
    assert tuple(sample['action'].shape)==(7,) and tuple(sample['observation.images.front'].shape)==(3,480,640)
    if dual:assert tuple(sample['observation.images.wrist'].shape)==(3,480,640)
    result={'status':'PASS','codebase_version':CODEBASE_VERSION,'episodes':loaded.num_episodes,'frames':len(loaded),'sample_shapes':{k:list(v.shape) for k,v in sample.items() if hasattr(v,'shape')},
            'local_compatibility_adapter':'datasets.Column -> list for LeRobot v2.1 column access',
            'sources':sources,'action_semantics':next(iter(semantics),None),'depth':'kept in raw sidecars; not assumed native pi0.5 input','uploads':False,'training_performed':False}
    result['sensor_schema']=next(iter(sensors),None)
    if a.validate_only and (a.output/'conversion_result.json').exists():
        result['sources']=json.loads((a.output/'conversion_result.json').read_text()).get('sources',[])
    (a.output/'conversion_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

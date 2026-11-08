#!/usr/bin/env python3
"""Bounded native-GPU training-scene pilot with isolated logs and all failures.

Run in the already-working project environment; no upload or model training.
Large batches remain blocked until the separately tracked quality gates pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from _bootstrap import ROOT
from robot_exam.data.multifruit_plan import collection_registry


def source_hashes():
    paths=sorted((ROOT/'src/robot_exam').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/build_tasks.py',ROOT/'scripts/collect_multifruit_parent.py',ROOT/'scripts/replay_multifruit_phase.py',ROOT/'scripts/calibrate_fruit_rest_pose.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def allocated_bytes(root):
    inodes=set();total=0
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        st=path.stat();key=st.st_dev,st.st_ino
        if key not in inodes:total+=st.st_blocks*512;inodes.add(key)
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--maximum-raw-gib',type=float,default=4.)
    p.add_argument('--target-selector',choices=['leftmost','rightmost'],default='leftmost')
    p.add_argument('--config',type=Path,default=ROOT/'configs/stage20_fruit_round_route_dev_v7.json')
    p.add_argument('--passive-rest-init',action='store_true',help='Prepare exact600s passive rest poses before scene initialization')
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a new batch directory; no retry/overwrite of prior attempts')
    cfg=json.loads((ROOT/'configs/stage20_multifruit_dataset_expansion_proposal_v1.json').read_text())
    if len(a.seeds)!=len(set(a.seeds)):raise ValueError('Do not count duplicated scene seeds')
    if len(a.seeds)>6 and not cfg['resource_gate']['large_collection_ready']:
        raise ValueError('Large collection gates not passed; maximum6pilot contexts')
    if not 0<a.maximum_raw_gib<=4.:raise ValueError('Local pilot budget is bounded to4GiB')
    registry={x['seed']:x for x in collection_registry(cfg)['collection']}
    scenes=[registry[s] for s in a.seeds]
    if any(s['split']!='train' for s in scenes):raise ValueError('This development collector cannot expose heldout seeds')
    out.mkdir(parents=True);frozen=source_hashes();(out/'source_freeze.json').write_text(json.dumps(frozen,indent=2))
    cache=ROOT/'outputs/stage20/multifruit_shared_models_v1';cache_before=allocated_bytes(cache) if cache.exists() else 0
    result={'status':'RUNNING','scope':'Oracle multiobject parked skills, not full-task/formal/model evaluation',
            'seeds':a.seeds,'maximum_raw_gib':a.maximum_raw_gib,'attempts':[],'training':False,'uploads':False}
    def save():
        (out/'progress.json').write_text(json.dumps(result,indent=2))
    save()
    for scene in scenes:
        settings=json.loads(a.config.read_text())
        if a.passive_rest_init:
            from build_tasks import build
            from calibrate_fruit_rest_pose import calibrate
            prep=out/f'seed_{scene["seed"]}'/'preparation';prep.mkdir(parents=True)
            reference,_=build('fruit',scene['seed'],prep/'reference.xml',wrist_camera=True,
                              layout_profile='bounded_scatter_v3',bin_label_profile='upright_v2',fruit_count=scene['object_count'])
            calibration=calibrate(reference,prep/'calibration',seconds=600.)
            if calibration['status']!='PASS_ALL_EXACT_FRUITS_STABLE':
                result['status']='STOPPED_PASSIVE_CALIBRATION_REJECTED';save();return
            settings.update(fruit_spawn_profile='passive_rest_init_v2',
                            fruit_rest_calibration=str(prep/'calibration/calibration.json'),
                            require_non_target_scene_preservation=True)
        for category in ('apple','banana','orange'):
            used=allocated_bytes(out)+max(0,allocated_bytes(cache)-cache_before)
            if shutil.disk_usage(out).free<25.5*2**30 or used+.15*2**30>a.maximum_raw_gib*2**30:
                result.update(status='STOPPED_STORAGE_GATE',used_bytes=used);save();return
            if source_hashes()!=frozen:raise RuntimeError('Source changed; stop instead of mixing controllers')
            attempt=out/f'seed_{scene["seed"]}'/category;attempt.mkdir(parents=True)
            raw=attempt/'raw';start=time.monotonic()
            config_path=attempt/'settings.json';config_path.write_text(json.dumps(settings,indent=2))
            command=[sys.executable,str(ROOT/'scripts/collect_multifruit_parent.py'),'--output',str(raw),
                     '--config',str(config_path),
                     '--seed',str(scene['seed']),'--category',category,'--fruit-count',str(scene['object_count']),
                     '--precision-dock-strategy','direct','--twist-braking-all','--guarded-dock',
                     '--target-selector',a.target_selector,'--require-hardware-renderer']
            row={'scene_context_id':scene['scene_context_id'],'seed':scene['seed'],'category':category,
                 'scene_object_count':scene['object_count'],'split':'train','path':str(raw)}
            with (attempt/'capture.log').open('x') as log:
                try:
                    cp=subprocess.run(command,cwd=attempt,stdout=log,stderr=subprocess.STDOUT,timeout=1000)
                    row['returncode']=cp.returncode
                except subprocess.TimeoutExpired:row['capture_timeout']=True
            path=raw/'result.json';physical=json.loads(path.read_text()) if path.exists() else {}
            row.update(physical_status=physical.get('status','INFRASTRUCTURE_FAILURE'),error=physical.get('error'),
                       non_target_preserved=physical.get('scene_preservation',{}).get('success'),
                       source_changed=source_hashes()!=frozen or physical.get('source_changed_during_attempt',False),
                       wall_seconds=time.monotonic()-start,replays=[])
            # Working directory is unique. Text warnings no longer mingle
            # with unrelated project processes; capture/full-rate scopes are
            # still explicitly combined inside the collector's one process.
            logs=list(attempt.glob('MUJOCO_LOG*'))
            row['capture_or_fullrate_text_warning_logs']=[str(x) for x in logs]
            row['text_warning_count']=sum(x.read_text(errors='replace').count('WARNING') for x in logs)
            if row['physical_status']=='PHYSICAL_PARENT_PASS_PENDING_REPLAY_AND_EXPORT' and not row['source_changed']:
                for phase in ('grasp','place'):
                    dest=raw/phase/'action_chunk_replay_v2.json'
                    with (raw/phase/'action_chunk_replay_v2.log').open('x') as log:
                        cp=subprocess.run([sys.executable,str(ROOT/'scripts/replay_multifruit_phase.py'),str(raw/phase),
                                           '--output',str(dest)],cwd=raw/phase,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                    row['replays'].append(json.loads(dest.read_text())['status'] if dest.exists() else 'INFRASTRUCTURE_FAILURE')
            row['quality']='PHYSICAL_REPLAY_CANDIDATE_PENDING_FORMAT_AND_LANGUAGE_QA' if row['replays']==['PASS','PASS'] and row['text_warning_count']==0 else 'NOT_ADMITTED'
            result['attempts'].append(row);save();print(scene['seed'],category,row['physical_status'],row['replays'],flush=True)
            if row['source_changed']:
                result['status']='STOPPED_SOURCE_CHANGED';save();return
    result['status']='COMPLETE_ALL_ATTEMPTS_RETAINED';save()


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Bounded teacher-frame camera-occlusion probe AFTER completed physical tests.

Zeroed RGB is an out-of-distribution diagnostic, not missing-camera training,
not a new policy, and never evidence of physical success without a camera.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from _bootstrap import ROOT


def require_complete_matrix(matrix):
    summary=json.loads((matrix/'summary.json').read_text())
    integrity=json.loads((matrix/'integrity.json').read_text())
    if summary['completed_cases']!=summary['expected_cases'] or integrity['source_hash_mismatches']:
        raise RuntimeError('Require completed source-frozen physical batch before any extra policy queries')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/stage15/data_preparation/executor_matched6_release_v4/fit_manifest/manifest.json')
    a=p.parse_args()
    require_complete_matrix(a.matrix)
    if a.output.exists():raise FileExistsError('Preserve probe results')
    a.output.mkdir(parents=True)
    sys.path.insert(0,'/home/li/vla/openpi/packages/openpi-client/src')
    from openpi_client.image_tools import resize_with_pad
    from robot_exam.vla.client import connect_openpi_policy
    entries=[e for e in json.loads(a.manifest.read_text())['accepted'] if e['seed']==15001]
    assert len(entries)==3 and all(e['split']=='train' for e in entries)
    report=dict(status='RUNNING',queries=[],training_updates=0,actuator_commands=0,
        scope='Seen train teacher frames, explicit RGB occlusion; not physical success, camera removal, or a causal proof of task improvement')
    start=time.monotonic()
    def save():(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for entry in entries:
            source=Path(entry['path']);frames=[json.loads(x) for x in (source/'frames.jsonl').read_text().splitlines()]
            for index in (0,30,60):
                row=frames[index];truth=np.asarray([x['action'] for x in frames[index:index+10]])
                original={'state':np.asarray(row['observation.state'],dtype=np.float32),'prompt':row['task']}
                for field,key in [('front_rgb','rgb'),('wrist_rgb','rgb_wrist')]:
                    with Image.open(source/row[key]) as image:original[field]=resize_with_pad(np.asarray(image.convert('RGB')),224,224)
                for seed in (0,1,2):
                    baseline=None
                    for condition in ('dual','zero_front','zero_wrist','zero_both'):
                        if time.monotonic()-start>120:raise TimeoutError('Bounded120s read-only probe budget')
                        obs=dict(original)
                        if condition in ('zero_front','zero_both'):obs['front_rgb']=np.zeros_like(original['front_rgb'])
                        if condition in ('zero_wrist','zero_both'):obs['wrist_rgb']=np.zeros_like(original['wrist_rgb'])
                        policy=connect_openpi_policy('127.0.0.1',a.port,compression='deflate',noise_seed=seed)
                        try:
                            meta=policy.get_server_metadata()
                            assert meta['checkpoint_step_label']=='9999' and meta['episode_policy_noise_seed']==seed
                            before=time.monotonic();out=policy.infer(obs);elapsed=time.monotonic()-before
                        finally:policy._ws.close()
                        actions=np.asarray(out['actions']);assert actions.shape==(10,7) and np.isfinite(actions).all()
                        if baseline is None:baseline=actions
                        report['queries'].append(dict(skill=entry['skill'],scene_seed=15001,frame=index,
                            policy_noise_seed=seed,condition=condition,wall_seconds=elapsed,
                            arm_mae_rad=float(np.mean(np.abs(actions[:,:6]-truth[:,:6]))),
                            maximum_arm_change_from_dual_rad=float(np.max(np.abs(actions[:,:6]-baseline[:,:6]))),
                            gripper_mae=float(np.mean(np.abs(actions[:,6]-truth[:,6]))),
                            predicted_actions=actions.tolist(),input_hashes={k:hashlib.sha256(obs[k].tobytes()).hexdigest() for k in ('front_rgb','wrist_rgb')}))
                        save()
        report.update(status='COMPLETE_READONLY_NOT_PHYSICAL_SUCCESS',expected_queries=108)
    except Exception as exc:
        report.update(status='INCOMPLETE_READONLY_PROBE',error=dict(type=type(exc).__name__,message=str(exc)))
    report['wall_seconds']=time.monotonic()-start;save()
    print(json.dumps({k:v for k,v in report.items() if k!='queries'}),flush=True)


if __name__=='__main__':main()

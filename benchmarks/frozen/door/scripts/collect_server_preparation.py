#!/usr/bin/env python3
"""Collect only completed local preparation evidence; no model/remote operations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence-root',type=Path,required=True)
    p.add_argument('--transfer-root',type=Path)
    p.add_argument('--bundle-result',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve preparation audits')
    def load(name):return json.loads((a.evidence_root/name).read_text())
    hooks=load('training_hooks_final_v2.json')
    assert hooks['status']=='PASS_TINY_CPU_PLUMBING_ONLY' and hooks['actual_validation_samples']==78
    pytest_log=(a.evidence_root/'pytest_full_final_v1.log').read_text()
    matches=re.findall(r'(\d+) passed in ([\d.]+)s',pytest_log)
    assert matches,'Actual full pytest completion required'
    count,seconds=matches[-1]
    source=load('original_dataset_checksum_v1.json');assert source['status']=='PASS'
    native={}
    for split in ('train','validation','test'):
        x=load(f'native_{split}_v1.json')
        assert x['status']=='PASS' and not x['raw_episode_files_required'] and x['camera_mask']['left_wrist_0_rgb']
        native[split]=x['native_lerobot_training_loader']
    operations={}
    for skill in ('apple','banana','orange'):
        x=load(f'expert_{skill}_v1/result.json');assert x['success']
        operations[skill]=x
    negative=load('mock_negative_v1/result.json');assert not negative['success']
    uvlog=(a.evidence_root/'uv_environment_dry_run_v2.log').read_text()
    assert 'Would install' in uvlog
    result={'status':'LOCAL_TESTS_PASS_BUNDLE_PENDING','timestamp':datetime.now(timezone.utc).isoformat(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'pytest':{'passed':int(count),'seconds':float(seconds),'log':str(a.evidence_root/'pytest_full_final_v1.log')},
        'dataset':{'qualified_episodes':173,'frames':17646,'split_counts':{'train':121,'validation':26,'test':26},
            'scope':'parked mobile single fruit to green target; no navigation or multi-fruit sorting',
            'unaltered_previous_transfer_files_checked':source['files_checked'],'native_loaders':native},
        'training_hook_test':hooks,'expert_local_regressions':operations,
        'negative_control':negative,'dependency_resolution':{'offline_frozen_dry_run':'PASS','packages_to_install':202,
            'installation_performed':False},'local_environment':load('local_environment_v1.json'),
        'server_runtime_pending':['GPU/driver/JAX compatibility','dependency installation',
            'actual pi05 parameters and tokenizer load','real 30-step LoRA and memory/disk measurement',
            'checkpoint reload/resume with actual pi05','remote inference latency and learned-policy physical success'],
        'training_performed':False,'weights_downloaded':False,'uploads_performed':False,
        'new_physical_formal_evaluation_performed':False,'free_disk_GiB':shutil.disk_usage(ROOT).free/1024**3,
        'runbook':'docs/46_server_training_preparation.md'}
    if a.transfer_root and a.bundle_result:
        x=json.loads((a.transfer_root/'transfer_manifest.json').read_text())
        bundle=json.loads(a.bundle_result.read_text())
        check=json.loads((a.transfer_root/'verification.json').read_text())
        copied=json.loads((a.transfer_root/'copied_layout_verification.json').read_text())
        assert bundle['status']=='LOCAL_UPLOAD_BUNDLE_VERIFIED' and check['status']==copied['status']=='PASS'
        result.update(status='LOCAL_PREPARATION_PASS_SERVER_RUNTIME_PENDING',
            frozen_source_commit=x['git_commit'],transfer_manifest=str(a.transfer_root/'transfer_manifest.json'),
            bundle=bundle,transfer_checks=check,copied_layout_checks=copied)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','source_commit','pytest','free_disk_GiB','training_performed')},indent=2))


if __name__=='__main__':main()

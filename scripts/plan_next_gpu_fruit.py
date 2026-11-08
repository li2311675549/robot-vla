#!/usr/bin/env python3
"""No-card readiness/command manifest; training is NEVER auto-started here."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def read_model_paths(file):
    allowed={'ROBOT_EXAM_WEIGHTS','ROBOT_EXAM_TOKENIZER','ROBOT_EXAM_PYTHON','ROBOT_EXAM_OPENPI_ROOT'}
    dependencies=allowed|{'ROBOT_EXAM_BUNDLE','OPENPI_DATA_HOME'};raw={}
    for line in file.read_text().splitlines():
        if not line.strip().startswith('export '):continue
        parts=shlex.split(line,comments=True)
        if len(parts)==2 and '=' in parts[1]:
            key,value=parts[1].split('=',1)
            if key in dependencies:raw[key]=value
    def expand(key,seen=()):
        if key in seen or key not in raw:raise ValueError('Unknown/cyclic model path variable')
        value=re.sub(r'\$(?:\{([A-Za-z_][A-Za-z_0-9]*)\}|([A-Za-z_][A-Za-z_0-9]*))',
                     lambda match:expand(match[1] or match[2],(*seen,key)),raw[key])
        if '$' in value or '`' in value:raise ValueError('Executable shell syntax forbidden in model paths')
        return value
    return {key:expand(key) for key in allowed}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--matched-settings',type=Path,help='Versioned matched-six fit settings; old3000-step default is retained')
    p.add_argument('--run-gpu-diagnostics',action='store_true',help='Explicitly run only read-only model diagnostics after GPU startup')
    a=p.parse_args();r=a.root
    if a.output.exists():raise FileExistsError('Preserve readiness records')
    paths=read_model_paths(r/'server_paths_v2.sh')
    assert all(Path(value).exists() for value in paths.values())
    checkpoint=r/'checkpoints/pilot_20261101_v1/pi05_original_ur5_lora_v1/dual_fruit_pilot_v1/999'
    assert (checkpoint/'_CHECKPOINT_METADATA').is_file() and (checkpoint/'params/_METADATA').is_file()
    base=r/'data_preparation/green_target_v2';merged=r/'data_preparation/green_timing_merged_v3'
    matched=r/'data_preparation/executor_matched6_v4'
    for folder in (base,merged,matched):assert (folder/'preparation.json').is_file()
    old_stats=json.loads((r/'dataset_manifest/assets/original_ur5/norm_stats.json').read_text())
    assert old_stats==json.loads((checkpoint/'assets/original_ur5/norm_stats.json').read_text())
    assert old_stats==json.loads((matched/'fit_manifest/assets/original_ur5/norm_stats.json').read_text())
    size=sum(f.stat().st_size for f in checkpoint.rglob('*') if f.is_file())
    data_free=shutil.disk_usage(r).free;root_free=shutil.disk_usage('/').free
    data_after=data_free-2*size-int(.5*1024**3);root_after=root_free-4*1024**3
    budget_ok=min(data_after,root_after)>=25*1024**3
    memory=Path('/sys/fs/cgroup/memory.max').read_text().strip();devices=glob.glob('/dev/nvidia*')
    settings_overfit=r/'tools/green_target_prepare_v2/pi05_green_target_overfit6_proposed_v2.json'
    settings_full=r/'tools/timing12_payload_v1/pi05_green_timing_full_proposed_v3.json'
    settings_matched=a.matched_settings or r/'tools/executor_matched6_payload_v4/pi05_executor_matched6_overfit_proposed_v4.json'
    for file in (settings_overfit,settings_full,settings_matched):
        plan=json.loads(file.read_text());assert plan['save_interval']>=plan['steps']
    matched_plan=json.loads(settings_matched.read_text())
    matched_name=matched_plan['run_name'];matched_steps=matched_plan['steps']
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',matched_name)
    assert type(matched_steps) is int and matched_steps>0
    matched_label=str(matched_steps-1)
    frozen_files=[settings_overfit,settings_full,settings_matched,
        Path(__file__).resolve(),r/'tools/green_target_prepare_v2/run_training_disk_guard.py',
        r/'tools/green_target_prepare_v2/diagnose_pi05_conditions_v1.py',
        r/'tools/training_ready_v1/run_resource_guarded_cgroup_v2.py',
        r/'robot_exam/scripts/openpi_ur5_train.py',
        r/'robot_exam/src/robot_exam/vla/openpi_config.py',
        r/'robot_exam/src/robot_exam/vla/openpi_validation.py',
        r/'runtime/fruit_eval_v4/scripts/serve_openpi_original_ur5.py',
        r/'dataset_manifest/assets/original_ur5/norm_stats.json',
        checkpoint/'assets/original_ur5/norm_stats.json',
        base/'manifest/assets/original_ur5/norm_stats.json',
        merged/'manifest/assets/original_ur5/norm_stats.json',
        matched/'fit_manifest/assets/original_ur5/norm_stats.json',
        matched/'preparation.json']
    common=[paths['ROBOT_EXAM_PYTHON'],str(r/'robot_exam/scripts/openpi_ur5_train.py'),
            '--openpi-root',paths['ROBOT_EXAM_OPENPI_ROOT'],'--weights',paths['ROBOT_EXAM_WEIGHTS'],
            '--tokenizer',paths['ROBOT_EXAM_TOKENIZER'],'--wrist-camera']
    def train_command(repo,folder,settings,name,manifest='manifest'):
        return ['/usr/bin/env','HF_DATASETS_CACHE=/tmp/robot_exam_next_gpu_hf_cache_v2',
            'HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1','XLA_PYTHON_CLIENT_MEM_FRACTION=.75',
            *common,'--repo-id',repo,'--assets',str(folder/manifest/'assets'),'--lerobot-cache',str(folder/'cache'),
            '--checkpoints',str(r/'checkpoints/next_stage16'),'--settings',str(settings),
            '--output',str(r/'runs'/name/'training_result.json'),'--authorized-training']
    diagnostic=[paths['ROBOT_EXAM_PYTHON'],str(r/'tools/green_target_prepare_v2/diagnose_pi05_conditions_v1.py'),
        '--robot-root',str(r/'robot_exam'),'--openpi-root',paths['ROBOT_EXAM_OPENPI_ROOT'],
        '--assets',str(r/'dataset_manifest/assets'),'--checkpoint',str(checkpoint),'--tokenizer',paths['ROBOT_EXAM_TOKENIZER'],
        '--lerobot-cache',str(r/'lerobot_cache'),'--output',str(r/'runs/next_gpu_conditions_v1.json'),'--authorized-gpu-diagnostic']
    def guarded(command,name,seconds):
        prefix=r/'setup_logs'/name
        return [paths['ROBOT_EXAM_PYTHON'],str(r/'tools/green_target_prepare_v2/run_training_disk_guard.py'),
            '--base-guard',str(r/'tools/training_ready_v1/run_resource_guarded_cgroup_v2.py'),
            '--disk-root',str(r),'--disk-root','/','--min-free-gib','25','--',
            '--memory-accounting','cgroup','--max-rss-mib','81920','--timeout',str(seconds),
            '--output',str(prefix)+'_guard.json','--log',str(prefix)+'.log','--',*command]
    def serving_command(repo,assets,cache,ckpt,old_cache=False):
        hf_cache=r/'hf_datasets_cache' if old_cache else Path('/tmp/robot_exam_next_gpu_hf_cache_v2')
        return ['/usr/bin/env',f'HF_DATASETS_CACHE={hf_cache}','HF_HUB_OFFLINE=1',
            'HF_DATASETS_OFFLINE=1','XLA_PYTHON_CLIENT_MEM_FRACTION=.75',
            paths['ROBOT_EXAM_PYTHON'],str(r/'runtime/fruit_eval_v4/scripts/serve_openpi_original_ur5.py'),
            '--openpi-root',paths['ROBOT_EXAM_OPENPI_ROOT'],'--assets',str(assets),
            '--checkpoint',str(ckpt),'--repo-id',repo,'--lerobot-cache',str(cache),
            '--tokenizer',paths['ROBOT_EXAM_TOKENIZER'],'--wrist-camera','--project-gripper',
            '--websocket-compression','deflate','--port','8000','--serve']
    report={'status':'READY_NO_TRAINING_STARTED' if budget_ok else 'BLOCKED_PROJECTED_STORAGE',
        'memory_max':memory,'gpu_device_files':devices,'checkpoint_label':'999','checkpoint_optimizer_step':1000,
        'oldnorm_matches_checkpoint':True,'checkpoint_total_bytes':size,'data_free_GiB':data_free/2**30,'root_free_GiB':root_free/2**30,
        'projected_data_free_after_two_final_checkpoints_and_0_5GiB_jit':data_after/2**30,
        'projected_root_free_after_4GiB_temporary_HF_cache':root_after/2**30,
        'budgets_above25GiB':budget_ok,'future_training_HF_DATASETS_CACHE':'/tmp/robot_exam_next_gpu_hf_cache_v2',
        'diagnostic_command':diagnostic,'overfit6_candidate_command':train_command('local/robot_exam_green_target_overfit6_train',base,settings_overfit,'green_target_overfit6_proposed_v2'),
        'full185_candidate_command':train_command('local/robot_exam_green_timing_full_train',merged,settings_full,'green_timing_full_proposed_v3'),
        'executor_matched6_alternative_command':train_command('local/robot_exam_executor_matched6_train',matched,settings_matched,matched_name,'fit_manifest'),
        'selected_matched_fit':{'settings':str(settings_matched),'run_name':matched_name,
            'optimizer_steps':matched_steps,'final_checkpoint_label':matched_label,
            'training_requires_user_confirmed_GPU_startup':True},
        'training_commands_executed':False,'model_loaded':False,
        'readiness_inputs_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in frozen_files},
        'rules':['Run actual GPU/JAX/BF16 preflight after paid GPU startup.',
                 'Read-only54-query diagnosis and limited training-scene physical tests precede any further training.',
                 'Train6 diagnostic first; choose at most one full-data candidate after results, no blind sequential full runs.',
                 'Choose ONE six-scene diagnostic: old-executor v2 or actual-deployment-matched v4; do not run both blindly.',
                 'Matched6 reuses old training scenes and frozen121train norm; it is not six new random scenes.',
                 'Full185 uses NEW127train norm and a NEW training run from official base, never resume999 with swapped norm.',
                 'All generated training commands require the established80GiB cgroup/time guard and runtime disk checks.',
                 'Old999 read-only diagnostics use original assets, not full185 newnorm.']}
    report['guarded_overfit6_candidate_command']=guarded(report['overfit6_candidate_command'],'next_overfit6_v2',3600)
    report['guarded_full185_candidate_command']=guarded(report['full185_candidate_command'],'next_full185_v3',10800)
    report['guarded_executor_matched6_alternative_command']=guarded(report['executor_matched6_alternative_command'],
        'next_'+matched_name,10800 if matched_steps>3000 else 3600)
    report['guarded_old999_inference_command']=guarded(serving_command(
        'local/robot_exam_dual_release_train',r/'dataset_manifest/assets',r/'lerobot_cache',checkpoint,True),
        'next_old999_inference_v1',1800)
    report['guarded_matched6_inference_after_training_command']=guarded(serving_command(
        'local/robot_exam_executor_matched6_train',matched/'fit_manifest/assets',matched/'cache',
        r/'checkpoints/next_stage16/pi05_original_ur5_lora_v1'/matched_name/matched_label),
        'next_'+matched_name+'_inference_v1',1800)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.run_gpu_diagnostics:
        if not devices or memory=='max' or int(memory)<60*1024**3 or not budget_ok:
            report['status']='REFUSED_GPU_DIAGNOSTIC_PRECONDITION';a.output.write_text(json.dumps(report,indent=2)+'\n')
            raise RuntimeError('GPU/memory/storage preconditions unmet; no model started')
        memory_mib=[[int(v.strip()) for v in s.split(',')] for s in subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
        if len(memory_mib)!=1 or memory_mib[0][0]<80*1024 or memory_mib[0][1]<70*1024:raise RuntimeError('Reuse verified single80+GiB total/70+GiB free server configuration; no model started')
        prefix=r/'setup_logs/next_gpu_conditions_v1'
        cmd=[paths['ROBOT_EXAM_PYTHON'],str(r/'tools/training_ready_v1/run_resource_guarded_cgroup_v2.py'),
             '--memory-accounting','cgroup','--max-rss-mib','81920','--timeout','900',
             '--output',str(prefix)+'_guard.json','--log',str(prefix)+'.log','--',*diagnostic]
        if Path(str(prefix)+'_launcher.log').exists():raise FileExistsError('Preserve previous model diagnostic')
        with Path(str(prefix)+'_launcher.log').open('x') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,XLA_PYTHON_CLIENT_MEM_FRACTION='.75',HF_DATASETS_CACHE=str(r/'hf_datasets_cache'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1'))
        report.update(status='READONLY_GPU_DIAGNOSTIC_LAUNCHED',guard_pid=proc.pid)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

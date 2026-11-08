#!/usr/bin/env python3
"""Bounded inference-only video collection; retain every attempted outcome."""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from _bootstrap import ROOT
from robot_exam.evaluation.video_collection import CATEGORIES, select_grasp_videos


def atomic(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(4*1024**2),b''):sha.update(part)
    return sha.hexdigest()


def describe(case_id, result_path, spec):
    r=json.loads(result_path.read_text()) if result_path.exists() else {
        'success':False,'status':'FAIL','failure_category':'process_or_initialization_error',
        'error':{'message':'Child did not leave a result; preserved process log'}}
    video=result_path.parent/'video/demo.mp4';meta=r.get('server_metadata') or {}
    eligible=(video.is_file() and r.get('video',{}).get('full_decode')=='PASS'
              and r.get('phase')=='grasp' and r.get('queries',0)>0
              and r.get('scene_object_count')==spec['scene_fruit_count']
              and r.get('category')==spec['category'] and r.get('seed')==spec['scene_seed']
              and str(meta.get('checkpoint_step_label'))=='9999'
              and meta.get('repo_id')=='local/robot_exam_stage21_train'
              and meta.get('episode_policy_noise_seed')==spec['policy_noise_seed'])
    if r.get('success'):
        score=r['score']
        if not (eligible and score['lift_m']>=.055 and
                score['last_second_bilateral_fraction']>=.9 and score['final_bilateral']
                and not r.get('other_fruit_lifted') and r['non_target_preservation']['success']):
            raise RuntimeError('Success receipt inconsistent with unchanged physical evaluator')
    return dict(id=case_id,category=spec['category'],scene_seed=spec['scene_seed'],
        scene_fruit_count=spec['scene_fruit_count'],policy_noise_seed=spec['policy_noise_seed'],
        condition=spec['condition'],success=bool(r.get('success')),exhibit_eligible=bool(eligible),
        failure_category=r.get('failure_category'),error=r.get('error'),queries=r.get('queries',0),
        result=str(result_path.resolve()),video=str(video.resolve()),sim_seconds=r.get('sim_seconds'),
        wall_seconds=r.get('wall_seconds'),score=r.get('score'))


def summarize(rows):
    per_class={}
    for category in CATEGORIES:
        group=[r for r in rows if r['category']==category]
        successful=[r for r in group if r['success'] and r['exhibit_eligible']]
        per_class[category]={'attempts':len(group),'successful_decoded_videos':len(successful),
            'decoded_videos':sum(r['exhibit_eligible'] for r in group),
            'observed_success_fraction_in_adaptive_collection':len(successful)/len(group) if group else None,
            'unique_scene_seeds':len(set(r['scene_seed'] for r in group)),
            'unique_scene_noise_pairs':len(set((r['scene_seed'],r['policy_noise_seed']) for r in group)),
            'failure_categories':dict(Counter(r['failure_category'] for r in group if not r['success']))}
    return {'attempts':len(rows),'decoded_videos':sum(r['exhibit_eligible'] for r in rows),
            'by_class':per_class,'scope':'Adaptive repetition/noise showcase, not independent formal test or generalization',
            'all_attempts_preserved':True,'test_partition_used':False}


def publish(root, rows, chosen, protocol):
    target_success=int(protocol.get('goal',{}).get('minimum_success_each_class',8))
    out=root/'collection40';out.mkdir()
    videos=out/'videos';records=out/'records';videos.mkdir();records.mkdir()
    entries=[];by_id={r['id']:r for r in rows}
    for index,case_id in enumerate(chosen,1):
        row=by_id[case_id];outcome='success' if row['success'] else 'failure'
        name=f"{index:02d}_{row['category']}_{outcome}_scene{row['scene_seed']}_rng{row['policy_noise_seed']}_{case_id}"
        video=videos/(name+'.mp4');source=Path(row['video'])
        try:os.link(source,video)
        except OSError:shutil.copy2(source,video)
        shutil.copy2(row['result'],records/(name+'.json'))
        entries.append({**row,'collection_video':str(video.relative_to(out)),
                        'video_sha256':digest(video),'result_sha256':digest(Path(row['result']))})
    report={'status':'DISPLAY_QUOTA_MET_NOT_FORMAL_SUCCESS_RATE','count':40,
        'successes':sum(r['success'] for r in entries),'failures':sum(not r['success'] for r in entries),
        'per_class':{c:{'success':sum(r['success'] for r in entries if r['category']==c),
                       'failure':sum(not r['success'] for r in entries if r['category']==c)} for c in CATEGORIES},
        'selection':f'At least{target_success} decoded successes per class and4 real failures; prefer different scene/noise before exact repetitions, then balance classes. No re-scoring, copied duplicate trials, or edited video.',
        'original_validation_result':protocol.get('original_validation_result','3/9, separately frozen and unchanged'),
        'scope':protocol['scope'],'all_attempt_statistics':summarize(rows),'videos':entries}
    atomic(out/'manifest.json',report)
    with (out/'all_attempts.csv').open('w') as stream:
        fields=['id','category','scene_seed','scene_fruit_count','policy_noise_seed','condition',
                'success','exhibit_eligible','failure_category','sim_seconds','wall_seconds','result','video']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    original_validation=protocol.get('original_validation_result','3/9')
    scene_note=(f"每回合初始桌面有{protocol['required_scene_fruit_count']}个水果。\n\n"
                if protocol.get('required_scene_fruit_count') else '')
    (out/'README.md').write_text('# 按指令抓取三类水果：40段本地视频\n\n'+scene_note+
        '真实模型checkpoint9999，不再训练。双视角输入、驻车、指定类别/最左目标抓起保持；不包含分类入筐或完整十果任务。\n\n'
        '成功沿用原评分：目标抬升>=55mm，最后1s双侧接触>=90%且结束仍接触，驻车、旁物保护、碰撞检查均通过。\n\n'
        f'本目录按每类至少{target_success}次成功及适量失败整理。为补齐录像，包含已成功初态的真实重新执行和推理采样变化；不是独立随机场景泛化评估，40段展示构成不能作为模型成功率。\n\n'
        '每段对应一次单独实际执行，未复制视频充当额外尝试、未剪辑拼接。相同初态/RNG可能产生相同视频字节，来源attempt不同且明确标记。全部尝试及失败见all_attempts.csv，详细物理评分见records/。'
        f'原固定验证结果{original_validation}保留，不用此展示集替代。\n')
    return str(out.resolve())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    a=parser.parse_args();protocol=json.loads(a.protocol.read_text());root=a.output.resolve()
    target_success=int(protocol.get('goal',{}).get('minimum_success_each_class',8))
    required=protocol.get('required_scene_fruit_count')
    if required is not None and any(s['scene_fruit_count']!=required for s in protocol['baseline']+protocol['plan']):
        raise ValueError('Collection includes a scene outside the required fruit count')
    if root.exists() and not a.resume:raise FileExistsError('Preserve previous collection')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=root/'status.json';phash=digest(a.protocol)
        state=json.loads(receipt.read_text()) if receipt.exists() else {'protocol_sha256':phash,'next_index':0,'active':None}
        if state['protocol_sha256']!=phash:raise ValueError('Do not change a running collection protocol')
        if state.get('status')=='QUOTA_COMPLETE':raise RuntimeError('Collection already complete')
        ledger=root/'all_attempts.json';rows=json.loads(ledger.read_text()) if ledger.exists() else []
        if not rows:
            for i,spec in enumerate(protocol['baseline']):
                rows.append(describe(f'baseline{i:03d}',Path(spec['result']),spec))
            atomic(ledger,rows)
        if state.get('active'):
            active=state['active'];pid=active.get('pid');cmd=Path(f'/proc/{pid}/cmdline')
            if pid and cmd.exists() and str(root).encode() in cmd.read_bytes():
                raise RuntimeError('Previous physical rollout still active; do not duplicate it')
            index=active['index'];spec=protocol['plan'][index]
            case_id=f'new{index:03d}'
            existing=[r for r in rows if r['id']==case_id]
            if existing:
                if len(existing)!=1 or existing[0]['result']!=str(Path(active['result']).resolve()):
                    raise RuntimeError('Inconsistent recovery ledger; never duplicate a physical attempt')
            else:
                rows.append(describe(case_id,Path(active['result']),spec));atomic(ledger,rows)
            state.update(next_index=index+1,active=None)
        started=time.monotonic();state.update(status='RUNNING',runner_pid=os.getpid(),summary=summarize(rows));atomic(receipt,state)
        for index,spec in enumerate(protocol['plan']):
            if index<state['next_index']:continue
            selected=select_grasp_videos(rows,successes_per_category=target_success)
            if selected is not None:
                state.update(status='QUOTA_COMPLETE',collection=publish(root,rows,selected,protocol));break
            summary=summarize(rows)
            if spec['condition']=='exact_initialization_and_rng_repetition' and summary['by_class'][spec['category']]['successful_decoded_videos']>=target_success:
                state['next_index']=index+1;atomic(receipt,state);continue
            if shutil.disk_usage(root).free<25.5*1024**3:
                state['status']='STOP_DISK_RESERVE';break
            if time.monotonic()-started>protocol['maximum_wall_seconds']:
                state['status']='STOP_WALL_BUDGET';break
            deadline_file=root/'operational_deadline.json'
            if deadline_file.is_file():
                deadline=json.loads(deadline_file.read_text())
                state['operational_deadline']=deadline
                if time.time()>=deadline['no_new_trials_after_epoch']:
                    state['status']='STOP_USER_GPU_DEADLINE';break
            fixture=Path(protocol['dataset_root'])/spec['fixture']
            for name,expected in spec['fixture_sha256'].items():
                if digest(fixture/name)!=expected:raise ValueError('Registered fixture differs')
            case=root/'attempts'/f"{index:03d}_{spec['category']}_rng{spec['policy_noise_seed']}"
            work=root/'work'/str(index);work.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,str(ROOT/'scripts/evaluate_multifruit_policy_phase.py'),
                '--fixture',str(fixture),'--output',str(case),'--policy','openpi',
                '--host','127.0.0.1','--port',str(protocol['port']),
                '--expected-checkpoint-label','9999','--expected-normalization-sha256',protocol['normalization_sha256'],
                '--policy-noise-seed',str(spec['policy_noise_seed']),'--seconds','30','--record']
            state['active']={'index':index,'result':str(case/'result.json'),'pid':None}
            atomic(receipt,state)
            with (work/'process.log').open('x') as log:
                child=subprocess.Popen(command,cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['active']['pid']=child.pid;atomic(receipt,state)
                try:child.wait(timeout=650)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
            row=describe(f'new{index:03d}',case/'result.json',spec);rows.append(row);atomic(ledger,rows)
            state.update(next_index=index+1,active=None,summary=summarize(rows));atomic(receipt,state)
            print(row['id'],row['category'],row['success'],state['summary']['by_class'],flush=True)
            if len(rows)>=3 and all(not r['queries'] for r in rows[-3:]):
                state['status']='STOP_REPEATED_INFRASTRUCTURE_ERRORS';break
        else:state['status']='PLAN_EXHAUSTED'
        if state['status']!='QUOTA_COMPLETE':
            selected=select_grasp_videos(rows,successes_per_category=target_success)
            if selected is not None:state.update(status='QUOTA_COMPLETE',collection=publish(root,rows,selected,protocol))
        state.update(active=None,summary=summarize(rows),wall_seconds=time.monotonic()-started)
        atomic(receipt,state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

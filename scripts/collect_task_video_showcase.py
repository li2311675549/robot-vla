#!/usr/bin/env python3
"""Quota-driven full-task video collection, with every attempt retained.

This is explicitly a demonstration selection protocol, never an estimator of
independent success rate. Existing and repeated initial conditions are labeled.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from _bootstrap import ROOT


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def admit(folder,task):
    folder=Path(folder).resolve();r=json.loads((folder/'result.json').read_text())
    video=folder/'video/demo.mp4'
    if r.get('task')!=task or not video.is_file():raise ValueError('Missing full task/video result')
    if r.get('initial_base',[0])[0]>-4.5:raise ValueError('Not a full five-metre start')
    if 'blank video' in str(r.get('error','')).lower() or r.get('video_error'):
        raise ValueError('Video/infrastructure failure is not a physical task failure exemplar')
    if task=='door' and r.get('door_profile')!='aligned_handles_v3':raise ValueError('Old hinge-side handle scene excluded')
    if task=='drawer' and (r.get('book_profile')!='distinct_v2' or 'scatter_v3' not in r.get('scene_protocol','')):
        raise ValueError('Booklets/earlier table layouts excluded')
    if r.get('success'):
        if task=='fruit' and r.get('completed')!=10:raise ValueError('Single fruit does not count as full classification')
        if task=='drawer':
            score=r.get('score',{})
            if not score.get('returned') or len(score.get('books',[]))!=3 or not all(x['inside_correct_layer'] for x in score['books']):
                raise ValueError('Incomplete three-book task')
    stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(video)],text=True))['streams'][0]
    if int(stream.get('nb_read_frames',0))<2:raise ValueError('Empty/truncated recording')
    cp=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    if cp.returncode or cp.stderr.strip():raise ValueError('Full video decode failed: '+cp.stderr[-500:])
    return {'episode':str(folder),'task':task,'success':bool(r.get('success')),'seed':r.get('seed'),
        'sim_seconds':r.get('sim_seconds'),'wall_seconds':r.get('wall_seconds'),'scene_protocol':r.get('scene_protocol'),
        'completed_fruits':r.get('completed'),'error':r.get('error'),'video':str(video),'video_sha256':digest(video),
        'result_sha256':digest(folder/'result.json'),'stream':stream,'full_decode':'PASS',
        'warning_counts':r.get('warnings'),'recording_view':json.loads((folder/'recording_manifest.json').read_text())
            if (folder/'recording_manifest.json').exists() else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());out=a.output.resolve()
    if out.exists():raise FileExistsError('Preserve previous collection; no automatic replay of interrupted jobs')
    out.mkdir(parents=True)
    paths=list((ROOT/'src/robot_exam').rglob('*.py'))+[ROOT/'scripts/run_task.py',ROOT/'scripts/build_tasks.py',
        ROOT/'scripts/record_task_view.py',ROOT/'scripts/record_frozen_task_view.py',Path(__file__).resolve(),a.plan.resolve()]
    paths+=list({Path(j['config']) for g in plan['task_groups'].values() for j in g['jobs']})
    freeze={str(p):digest(p) for p in paths};(out/'source_freeze.json').write_text(json.dumps(freeze,indent=2))
    state={'status':'RUNNING','purpose':plan['purpose'],'not_a_formal_success_rate':True,
        'all_raw_attempts_retained':True,'groups':{task:{'status':'RUNNING','success_goal':g['success_goal'],
            'failure_goal':g['failure_goal'],'records':[],'excluded':[],'new_attempts':[]} for task,g in plan['task_groups'].items()}}
    lock=threading.Lock()
    def save():
        with lock:
            tmp=out/'progress.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'progress.json')
    def counts(task):
        records=state['groups'][task]['records'];return sum(r['success'] for r in records),sum(not r['success'] for r in records)
    def add(task,path,origin):
        group=state['groups'][task]
        try:
            row=admit(path,task);row['origin']=origin
            if any(r['video_sha256']==row['video_sha256'] for r in group['records']):
                raise ValueError('Identical video content retained in raw evidence but not counted twice in15')
            group['records'].append(row)
        except Exception as exc:group['excluded'].append({'episode':str(path),'reason':str(exc),'origin':origin})
        save()
    def execute_task(task,group_plan):
        group=state['groups'][task]
        for path in group_plan['existing']:add(task,ROOT/path,'historical_recording')
        for path in group_plan.get('wait_existing',[]):
            folder=ROOT/path;started=time.monotonic()
            while not(folder/'result.json').exists() and time.monotonic()-started<1500:time.sleep(5)
            add(task,folder,'already_started_before_batch')
        for index,job in enumerate(group_plan['jobs']):
            success,failure=counts(task)
            if success>=group_plan['success_goal'] and failure>=group_plan['failure_goal']:break
            if index>=group_plan['max_new_attempts']:break
            if shutil.disk_usage(out).free<26*2**30:
                group['status']='STOPPED_STORAGE_GATE';save();return
            if any(digest(Path(p))!=h for p,h in freeze.items()):raise RuntimeError('Source/config changed mid-collection')
            folder=(Path(job['code_root'])/'outputs'/'showcase_v2'/task/f'attempt_{index:03d}_seed{job["seed"]}'
                    if job.get('code_root') else out/task/f'attempt_{index:03d}_seed{job["seed"]}')
            folder.parent.mkdir(parents=True,exist_ok=True)
            wrapper='record_frozen_task_view.py' if job.get('code_root') else 'record_task_view.py'
            command=[sys.executable,str(ROOT/'scripts'/wrapper),'--task',task,'--seed',str(job['seed']),
                '--mode',job['mode'],'--config',job['config'],'--output',str(folder),
                '--view-azimuth',str(job['angle'])]
            if job.get('code_root'):
                command.remove('--mode');command.remove(job['mode'])
                command+=['--code-root',job['code_root']]
            log=folder.with_suffix('.log');wall=time.monotonic()
            with log.open('x') as stream:
                try:code=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=3900).returncode
                except subprocess.TimeoutExpired:code=None
            group['new_attempts'].append({'index':index,'job':job,'episode':str(folder),'returncode':code,
                'wall_seconds':time.monotonic()-wall,'log':str(log)})
            add(task,folder,'fresh_physical_recording')
            print(task,index,job['seed'],counts(task),flush=True)
        success,failure=counts(task)
        group['status']='QUOTA_MET' if success>=group_plan['success_goal'] and failure>=group_plan['failure_goal'] else 'QUOTA_INCOMPLETE_ALL_ATTEMPTS_RETAINED'
        # Explicit fixed composition, never interpreted as empirical rate.
        chosen=[r for r in group['records'] if r['success']][:group_plan['success_goal']]
        chosen+=[r for r in group['records'] if not r['success']][:group_plan['failure_goal']]
        group['selected']=chosen;group['counts']={'valid_success_recordings':success,'valid_failure_recordings':failure}
        save()
    save()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(execute_task,t,g) for t,g in plan['task_groups'].items()]
        for f in futures:f.result()
    state['status']='ALL_THREE_QUOTAS_MET' if all(g['status']=='QUOTA_MET' for g in state['groups'].values()) else 'INCOMPLETE_WITH_EVIDENCE'
    save()


if __name__=='__main__':main()

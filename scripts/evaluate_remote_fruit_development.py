#!/usr/bin/env python3
"""Frozen paired development rollouts; every attempted case is retained."""
import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys

from _bootstrap import ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resource_limits(protocol):
    """Version disk admission only; never change the physical success score."""
    reserve = float(protocol.get('minimum_free_gib', 26))
    cap = protocol.get('maximum_output_bytes')
    if not math.isfinite(reserve) or reserve < 25:
        raise ValueError('At least25GiB disk reserve is required')
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ValueError('Output cap must be a positive byte count')
    return reserve * 1024**3, cap


def acquire_evaluation_lock(path):
    handle=path.open('a+')
    try:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Another physical remote evaluation is active; refuse duplicate batch') from None
    handle.seek(0);handle.truncate();handle.write(str(os.getpid())+'\n');handle.flush()
    return handle


def source_revision(root):
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=root,text=True,capture_output=True)
    if git.returncode==0:return git.stdout.strip(),'git_worktree'
    manifest=json.loads((root/'source_revision.json').read_text())
    commit=manifest['git_commit']
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Invalid source commit')
    for name,expected in manifest['files'].items():
        file=(root/name).resolve()
        if root.resolve() not in file.parents or digest(file)!=expected:
            raise ValueError('Exported source manifest mismatch: '+name)
    return commit,'verified_source_export_no_git_checkout'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=18000)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();protocol=json.loads(a.protocol.read_text())
    evaluation_lock=acquire_evaluation_lock(ROOT/'outputs/.remote_fruit_evaluation.lock')
    assert protocol['split']=='development_not_formal'
    reserve, output_cap = resource_limits(protocol)
    noise_seeds=protocol.get('evaluation_noise_seeds',[None])
    if len(set(noise_seeds))!=len(noise_seeds) or not noise_seeds:raise ValueError('Use unique predeclared evaluation noise seeds')
    expected_cases=len(protocol['scene_seeds'])*len(protocol['skills'])*len(protocol['policies'])*len(noise_seeds)
    if a.output.exists() and not a.resume:raise FileExistsError('Preserve prior batch')
    a.output.mkdir(parents=True,exist_ok=True)
    paths=list((ROOT/'src/robot_exam').rglob('*.py'))+[Path(__file__),ROOT/'scripts/evaluate_stationary_dual_skill.py',ROOT/'scripts/serve_openpi_original_ur5.py',a.protocol]
    paths += [ROOT/'scripts/run_fruit_fifty_evaluation.py', ROOT/'scripts/analyze_remote_fruit_evaluation.py']
    if (ROOT/'scripts/probe_fixed_policy_vision.py').exists():paths.append(ROOT/'scripts/probe_fixed_policy_vision.py')
    if (ROOT/'source_revision.json').exists():paths.append(ROOT/'source_revision.json')
    paths+=list(a.fixtures.rglob('scene.xml'))+list(a.fixtures.rglob('fixture.json'))
    identity={str(x.resolve()):digest(x) for x in paths}
    lock=a.output/'freeze.json'
    if a.resume:
        assert json.loads(lock.read_text())['source_hashes']==identity,'Frozen inputs changed'
    else:
        commit,provenance=source_revision(ROOT)
        lock.write_text(json.dumps({'protocol':protocol,'source_hashes':identity,
            'git_commit':commit,'git_provenance':provenance},indent=2)+'\n')
    records=[];fresh_records=[]
    for seed,skill,policy,noise_seed in itertools.product(protocol['scene_seeds'],protocol['skills'],protocol['policies'],noise_seeds):
        if shutil.disk_usage(ROOT).free < reserve:raise RuntimeError('Frozen disk reserve reached; preserve unfinished cases')
        if output_cap is not None and sum(x.stat().st_size for x in a.output.rglob('*') if x.is_file()) >= output_cap:
            raise RuntimeError('Frozen output cap reached; preserve unfinished cases')
        if any(digest(Path(x))!=h for x,h in identity.items()):raise RuntimeError('Frozen evaluation changed')
        suffix=f'_rng{noise_seed:02d}' if noise_seed is not None else ''
        folder=(a.output/f'{seed}_{skill}_{policy}{suffix}').resolve();result_path=folder/'result.json'
        new_attempt=not folder.exists()
        if result_path.exists():result=json.loads(result_path.read_text())
        elif folder.exists():
            result={'success':False,'policy':policy,'seed':seed,'skill':skill,'error':{'category':'process_interrupted','message':'Preserved without free retry'}}
            result_path.write_text(json.dumps(result,indent=2)+'\n')
        else:
            cmd=[sys.executable,str(ROOT/'scripts/evaluate_stationary_dual_skill.py'),'--fixture',str((a.fixtures/skill/f'seed_{seed:04d}').resolve()),
                '--policy',policy,'--output',str(folder),'--seconds',str(protocol['sim_seconds_limit']),'--execute-actions',str(protocol['execute_actions']),'--record']
            if policy=='openpi':cmd+=['--port',str(a.port),'--network-image-size','224','--websocket-compression','deflate',
                                     '--max-response-seconds',str(protocol['maximum_response_wall_seconds'])]
            if policy=='openpi' and protocol.get('interpolate_lookahead'):cmd+=['--interpolate-lookahead']
            if policy=='openpi' and protocol.get('server_checkpoint_label'):
                cmd+=['--expected-checkpoint-label',str(protocol['server_checkpoint_label'])]
            if policy=='openpi' and noise_seed is not None:cmd+=['--policy-noise-seed',str(noise_seed)]
            if protocol.get('video_backend')=='pyav':cmd+=['--video-backend','pyav']
            # The evaluator owns its fresh output directory. Keep process logs
            # outside it, including admission failures before it is created.
            logs=a.output/'process_logs';logs.mkdir(exist_ok=True)
            with (logs/f'{folder.name}.log').open('w') as log:
                proc=subprocess.Popen(cmd,cwd=logs.resolve(),env=dict(os.environ,MUJOCO_GL='egl'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:proc.wait(timeout=protocol['wall_seconds_limit']+60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            if result_path.exists():result=json.loads(result_path.read_text())
            else:
                folder.mkdir(exist_ok=True)
                result={'success':False,'policy':policy,'seed':seed,'skill':skill,'error':{'category':'process_failed','message':f'Exit{proc.returncode}'}}
                result_path.write_text(json.dumps(result,indent=2)+'\n')
        result['evidence']=str(result_path);result['evaluation_noise_seed']=noise_seed;records.append(result)
        if new_attempt:fresh_records.append(result)
        summary={}
        for label in protocol['policies']:
            rows=[x for x in records if x['policy']==label]
            summary[label]={'attempts':len(rows),'successes':sum(x.get('success',False) for x in rows),
                'by_class':{s:{'attempts':sum(x['skill']==s for x in rows),'successes':sum(x['skill']==s and x.get('success',False) for x in rows)} for s in protocol['skills']}}
        (a.output/'episodes.json').write_text(json.dumps(records,indent=2)+'\n')
        (a.output/'summary.json').write_text(json.dumps({'protocol':protocol,'summary':summary,'completed_cases':len(records),'expected_cases':expected_cases},indent=2)+'\n')
        print(seed,skill,policy,result.get('success'),result.get('error'),flush=True)
        limit=protocol.get('maximum_consecutive_infrastructure_failures')
        if new_attempt and limit and len(fresh_records)>=limit:
            recent=fresh_records[-limit:]
            def infrastructure_failure(row):
                error=row.get('error') or {}
                message=error.get('message','').lower()
                return error.get('category') in ('process_failed','process_interrupted') or any(
                    token in message for token in ('connection refused','connection reset','timed out','checkpoint mismatch',
                                                    'valid http response','connection closed','no close frame','another episode is active'))
            if all(infrastructure_failure(row) for row in recent):
                raise RuntimeError('Repeated infrastructure failure; batch incomplete, all attempted failures retained')
    (a.output/'integrity.json').write_text(json.dumps({'source_hash_mismatches':[x for x,h in identity.items() if digest(Path(x))!=h],
        'all_cases_counted':len(records)==expected_cases,'not_formal':True},indent=2)+'\n')


if __name__=='__main__':main()

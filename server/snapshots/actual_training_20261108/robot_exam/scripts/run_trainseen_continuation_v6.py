#!/usr/bin/env python3
"""Continue unattempted draws; retain all original and supplemental outcomes."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from _bootstrap import ROOT
from run_fruit_fifty_evaluation import stop_own_server


def infrastructure_error(row):
    error=row.get('error') or {}
    message=error.get('message','').lower()
    return error.get('category') in ('process_failed','process_interrupted') or any(
        x in message for x in ('valid http response','timed out in','response too stale','connection refused',
                              'connection reset','no close frame','checkpoint mismatch','noise seed mismatch'))


def summarize(rows):
    result={}
    for skill in ('apple','banana','orange'):
        group=[r for r in rows if r['skill']==skill]
        original=[r for r in group if r.get('evaluation_noise_seed',-1)<20]
        extra=[r for r in group if r.get('evaluation_noise_seed',-1)>=20]
        valid=[r for r in group if not infrastructure_error(r)]
        success=sum(bool(r.get('success')) for r in group)
        result[skill]=dict(all_attempts=len(group),all_successes=success,
            all_attempt_success_rate=success/len(group) if group else None,
            original_predeclared_attempts=len(original),original_predeclared_successes=sum(bool(r.get('success')) for r in original),
            supplementary_attempts=len(extra),supplementary_successes=sum(bool(r.get('success')) for r in extra),
            infrastructure_failures=len(group)-len(valid),valid_model_evaluations=len(valid),
            conditional_model_success_rate=sum(bool(r.get('success')) for r in valid)/len(valid) if valid else None,
            lifted_over55mm=sum(r.get('maximum_lift_m',0)>.055 for r in valid),
            stable_lift=sum(r.get('maximum_lift_m',0)>.055 and r.get('continuous_bilateral_hold_seconds',0)>=1.5 for r in valid),
            fully_decoded_videos=sum((r.get('video') or {}).get('full_decode')=='PASS' for r in group))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--server-receipt',type=Path,required=True)
    p.add_argument('--ssh-socket',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--execution-prefix',type=int,choices=(1,5),default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    status=a.output/'run_status.json'
    if status.exists():raise FileExistsError('Do not duplicate continuation')
    old=ROOT/('outputs/stage17/prefix5_deployment_diagnostic_v1/matrix' if a.execution_prefix==5 else 'outputs/stage17/train_seen_20_each_local_v4/matrix')
    rows=json.loads((old/'episodes.json').read_text())
    if a.execution_prefix==5:
        assert len(rows)==6 and all(r['execute_actions']==5 for r in rows)
        assert {(r['skill'],r['evaluation_noise_seed']) for r in rows}=={(s,n) for s in ('apple','banana','orange') for n in (0,2)}
    else:
        assert len(rows)==9 and all(r['skill']=='apple' for r in rows)
        assert [r['evaluation_noise_seed'] for r in rows]==list(range(9))
    expected=60 if a.execution_prefix==5 else 63
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();returns={}
    def save(state):
        report=dict(status=state,started_utc=start,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            checkpoint='9999',training_updates=0,scene_seed=15001,geometry_randomization=False,
            retained_prior_attempts=6 if a.execution_prefix==5 else 9,expected_all_attempts=expected,
            execution_prefix=a.execution_prefix,all_attempts=len(rows),per_class=summarize(rows),
            source_scope='One fixed seen training condition per fruit, not generalization or complete examination tasks',
            note=('Prefix5 independent deployment protocol: pilot6 reused including failures, only unattempted noises1,3..19 continued. All60 outcomes kept; do not merge with prefix1.' if a.execution_prefix==5 else
                  'Original planned60 include3 startup failures. Supplementaryapple noise20..22 predeclared. Keep all63 attempts; report original, supplementary, infrastructure and conditional model outcomes separately.'),
            child_returncodes=returns)
        status.write_text(json.dumps(report,indent=2)+'\n')
        (a.output/'all_episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
        return report
    save('RUNNING')
    try:
        order=('apple','orange','banana') if a.execution_prefix==5 else ('banana','orange','apple')
        for skill in order:
            configuration=f'configs/pi05_trainseen_prefix5_continue_{skill}_v2.json' if a.execution_prefix==5 else f'configs/pi05_trainseen_continue_{skill}_v6.json'
            cmd=[sys.executable,str(ROOT/'scripts/run_fruit_fifty_evaluation.py'),
                 '--protocol',str(ROOT/configuration),
                 '--fixtures',str(ROOT/'outputs/stage15/data_preparation/known_train_fixtures_green_v5'),
                 '--output',str((a.output/skill).resolve()),'--port',str(a.port)]
            returns[skill]=subprocess.run(cmd,cwd=ROOT).returncode
            matrix=a.output/skill/'matrix'
            if (matrix/'episodes.json').exists():rows.extend(json.loads((matrix/'episodes.json').read_text()))
            save('RUNNING')
            if returns[skill]!=0:break
        report=save('COMPLETE' if len(rows)==expected and all(x==0 for x in returns.values()) else 'INCOMPLETE')
        deadline=datetime.datetime.fromisoformat('2026-11-02T19:48:00+00:00')
        if report['status']=='COMPLETE' and datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=150)<deadline:
            probe=[sys.executable,str(ROOT/'scripts/probe_fixed_policy_vision.py'),'--matrix',str((a.output/'orange/matrix').resolve()),
                   '--output',str((a.output/'vision_probe').resolve()),'--port',str(a.port)]
            try:subprocess.run(probe,cwd=ROOT,timeout=150)
            except subprocess.TimeoutExpired:(a.output/'vision_probe_watchdog.txt').write_text('150s budget; physical results preserved\n')
    finally:
        try:cleanup=stop_own_server(a)
        except Exception as exc:cleanup=dict(status='UNCONFIRMED',error=str(exc),fallback='Independent server guard and user shutdown timer')
        (a.output/'inference_cleanup.json').write_text(json.dumps(cleanup,indent=2)+'\n')
    index=['# 固定训练场景全部尝试与视频','','原计划与补充回合、基础设施错误均保留。','']
    for row in rows:
        directory=Path(row['evidence']).parent
        index.append(f"- {row['skill']} rng{row['evaluation_noise_seed']:02d}: {'PASS' if row.get('success') else 'FAIL'}, infrastructure={infrastructure_error(row)} — {directory/'video/demo.mp4'}")
    (a.output/'VIDEO_INDEX.md').write_text('\n'.join(index)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Serial fixed-condition deployment study with explicit retained pilot cohorts."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

from _bootstrap import ROOT
from run_trainseen_continuation_v6 import summarize
from run_fruit_fifty_evaluation import stop_own_server


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--server-receipt',type=Path,required=True)
    p.add_argument('--ssh-socket',required=True)
    p.add_argument('--port',type=int,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'run_status.json').exists():raise FileExistsError('Preserve existing study')
    rows=[];returns={};started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for source in plan['prior_cohorts']:
        cohort=json.loads((ROOT/source['matrix']/'episodes.json').read_text())
        selected=[r for r in cohort if r['skill'] in source['skills'] and r['execute_actions']==source['execution_prefix']]
        assert len(selected)==source['expected_retained'],'Retain all matching pilot outcomes, including failures'
        rows.extend(selected)
    def save(state):
        keys=[(r['skill'],r['evaluation_noise_seed']) for r in rows]
        assert len(keys)==len(set(keys)),'Never duplicate a pilot draw'
        result=dict(status=state,plan=plan,checkpoint='9999',training_updates=0,
            started_utc=started,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            attempts=len(rows),expected_attempts=plan['expected_attempts'],per_class=summarize(rows),
            child_returncodes=returns,scope='Declared class-conditioned development deployment on one fixed seen training scene; not uniform-prefix or novel-scene generalization')
        (a.output/'run_status.json').write_text(json.dumps(result,indent=2)+'\n')
        (a.output/'all_episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
        return result
    save('RUNNING')
    try:
        for phase in plan['phases']:
            configuration=ROOT/phase['config'];cfg=json.loads(configuration.read_text())
            assert cfg['execute_actions']==plan['execution_prefix_by_class'][phase['skill']]
            cmd=[sys.executable,str(ROOT/'scripts/run_fruit_fifty_evaluation.py'),'--protocol',str(configuration),
                '--fixtures',str(ROOT/'outputs/stage15/data_preparation/known_train_fixtures_green_v5'),
                '--output',str((a.output/phase['skill']).resolve()),'--port',str(a.port)]
            returns[phase['skill']]=subprocess.run(cmd,cwd=ROOT).returncode
            path=a.output/phase['skill']/'matrix/episodes.json'
            if path.exists():rows.extend(json.loads(path.read_text()))
            save('RUNNING')
            if returns[phase['skill']]!=0:break
        result=save('COMPLETE_ATTEMPT_SCHEDULE' if len(rows)==plan['expected_attempts'] and all(x==0 for x in returns.values()) else 'INCOMPLETE')
    finally:
        try:cleanup=stop_own_server(a)
        except Exception as exc:cleanup=dict(status='UNCONFIRMED',error=str(exc))
        (a.output/'inference_cleanup.json').write_text(json.dumps(cleanup,indent=2)+'\n')
    lines=['# 固定训练场景开发部署研究：全部原始回合','','配置按类别预先固定，不逐回合选择执行方式；原比较协议全部另存。','']
    for row in rows:
        path=Path(row['evidence']).parent/'video/demo.mp4'
        lines.append(f"- {row['skill']} rng{row['evaluation_noise_seed']:02d}, prefix{row['execute_actions']}, {'PASS' if row['success'] else 'FAIL'}: {path}")
    (a.output/'VIDEO_INDEX.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

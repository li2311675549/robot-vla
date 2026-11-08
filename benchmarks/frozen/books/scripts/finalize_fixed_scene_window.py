#!/usr/bin/env python3
"""Local-only postprocessing after the exact evaluation unit exits.

No inference, training, upload, resource purchase, or deletion is performed.
"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from _bootstrap import ROOT
from run_trainseen_continuation_v6 import summarize


def main():
    root=ROOT/'outputs/stage17/train_seen_20_each_continuation_v6'
    unit='robot-exam-trainseen-continuation-v6.service'
    start=time.monotonic()
    while time.monotonic()-start<4500:
        active=subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0
        if not active:break
        time.sleep(20)
    else:raise TimeoutError('Local postprocessing wait budget reached')
    matrices=[ROOT/'outputs/stage17/train_seen_20_each_local_v4/matrix']+[root/s/'matrix' for s in ('banana','orange','apple')]
    rows=[];audits=[];solver_logs=[]
    for matrix in matrices:
        if (matrix/'episodes.json').exists():rows.extend(json.loads((matrix/'episodes.json').read_text()))
        if (matrix/'freeze.json').exists():
            frozen=json.loads((matrix/'freeze.json').read_text())
            mismatch=[p for p,h in frozen['source_hashes'].items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
            audits.append(dict(matrix=str(matrix),git_commit=frozen['git_commit'],mismatches=mismatch))
        log=matrix/'process_logs/MUJOCO_LOG.TXT'
        if log.exists():solver_logs.append(dict(path=str(log),warning_lines=log.read_text(errors='replace').count('WARNING:')))
    per_class=summarize(rows)
    complete=len(rows)==63 and all(c['valid_model_evaluations']==20 for c in per_class.values()) and not any(a['mismatches'] for a in audits)
    test_log=root/'pytest_window_final.log'
    with test_log.open('x') as log:
        tests=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    text=test_log.read_text();passed=re.findall(r'(\d+) passed',text)
    replay=[]
    for path in sorted((root/'reproducibility').glob('*.json')):
        data=json.loads(path.read_text())
        if isinstance(data,dict) and data.get('status') in ('PASS_REPRODUCIBILITY_ONLY','FAIL_REPRODUCIBILITY'):
            replay.append(dict(path=str(path),status=data['status'],maximum_state_error=data['maximum_state_error'],
                               model_queries=data['model_queries']))
    result=dict(status='COMPLETE_FIXED_TRAINING_CONDITION_EVALUATION' if complete else 'INCOMPLETE_OR_INTEGRITY_FAILURE',
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint='9999',training_updates=0,
        new_qualified_training_demonstrations=0,expected_total_attempts=63,total_attempts=len(rows),per_class=per_class,
        source_audits=audits,pytest=dict(returncode=tests.returncode,passed=int(passed[-1]) if passed else None,log=str(test_log)),
        replay_diagnostics=replay,source_video_index=str(root/'VIDEO_INDEX.md'),
        data_warning_trials=sum(bool(any(r.get('warnings',[]))) for r in rows),
        unsafe_contact_trials=sum(bool(r.get('unexpected_contacts')) for r in rows),
        solver_text_logs=solver_logs,local_free_GiB=shutil.disk_usage(ROOT).free/1024**3,
        scope='One fixed seen training scene per fruit. Original20+supplement and infrastructure outcomes must stay separate. Not novel-scene generalization or original full examination tasks.',
        source_upload_status='Previously rejected by automatic review; separate explicit authorization was requested. This job never uploads anything.')
    output=ROOT/'docs/results/fixed_scene_window_final_v7.json'
    if output.exists():raise FileExistsError('Preserve prior final report')
    output.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 固定训练场景各20次评测终态','','自动从全部原始结果汇总，非新场景泛化或三个完整考核任务。',
        f"状态：{result['status']}；实际总尝试{len(rows)}/63；63为含预声明补充的计划数量，实际执行补充数量见表。",'',
        '| 类别 | 原定20尝试成功数 | 补充成功/次数 | 基础设施失败 | 有效模型评测成功率 |',
        '|---|---:|---:|---:|---:|']
    for skill,c in per_class.items():
        rate='未测试' if c['conditional_model_success_rate'] is None else f"{100*c['conditional_model_success_rate']:.1f}% ({c['all_successes']}/{c['valid_model_evaluations']})"
        lines.append(f"| {skill} | {c['original_predeclared_successes']}/{c['original_predeclared_attempts']} | {c['supplementary_successes']}/{c['supplementary_attempts']} | {c['infrastructure_failures']} | {rate} |")
    lines+=['',f"完整pytest返回码{tests.returncode}，通过{result['pytest']['passed']}；原始日志{test_log}。",
        f"动作重放诊断{len(replay)}次，仅验证复现，不增加模型评测分母。",
        f"全部视频索引：{root/'VIDEO_INDEX.md'}。机器摘要：{output}。",
        '单回合data.warning和进程文本日志分别保留；不能把前者为零解释为绝无求解器文本提示。',
        '未追加训练、未新发布示教数据、未修改用户云端关机设置。']
    (ROOT/'docs/62_fixed_scene_twenty_window_results.md').write_text('\n'.join(lines)+'\n')
    handoff=ROOT/'docs/CODEX_HANDOFF.md';content=handoff.read_text();title='# 项目开发与服务器准备交接\n'
    section=f"\n## 最新自动终态：固定场景评测与回归（{result['timestamp_utc']}）\n\n- {result['status']}，总尝试{len(rows)}/63。逐类原计划/补充/基础设施/有效模型分母见docs62和{output.relative_to(ROOT)}，不要混同正式考核或泛化。\n- 完整pytest返回{tests.returncode}、通过{result['pytest']['passed']}；日志{test_log.relative_to(ROOT)}。不自动重开推理或训练；先检查远端是否已按用户定时关机和本地任务终态。\n- 原始视频/失败/重放诊断全部保留。源码上传授权另行待核对；此自动收尾未进行任何外传。\n\n"
    assert content.startswith(title);handoff.write_text(title+section+content[len(title):])
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

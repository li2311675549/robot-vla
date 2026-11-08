#!/usr/bin/env python3
"""Organize requested presentation examples without changing the full experiment ledger."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import ROOT


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest()


def describe(path):
    result = json.loads(path.read_text())
    trajectory = json.loads((path.parent/'trajectory.json').read_text())
    queries = json.loads((path.parent/'queries.json').read_text())
    time = np.array([x['time'] for x in trajectory])
    xyz = np.array([x['position'] for x in trajectory])
    lift = np.array([x['lift_m'] for x in trajectory])
    u = np.linspace(0, 1, 24)
    progress = (time-time[0])/max(time[-1]-time[0], 1e-9)
    sampled = np.column_stack([np.interp(u, progress, xyz[:, j]-xyz[0,j]) for j in range(3)])
    path_length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    states = np.array([x['measured_state_before'][:6] for x in queries])
    joint_travel = float(np.linalg.norm(np.diff(states, axis=0), axis=1).sum())
    first_lift = float(time[np.flatnonzero(lift >= .055)[0]]-time[0]) if np.any(lift >= .055) else None
    metadata = result.get('server_metadata') or {}
    stats = dict(skill=result['skill'],noise_seed=metadata['episode_policy_noise_seed'],
                 success=result['success'],sim_seconds=result['sim_seconds'],wall_seconds=result['wall_seconds'],
                 max_lift_mm=1000*result['maximum_lift_m'],bilateral_hold_seconds=result['continuous_bilateral_hold_seconds'],
                 placement_error_mm=1000*result['placement_error_m'],final_speed_m_s=result['final_speed_m_s'],
                 object_path_length_m=path_length,joint_path_length_rad=joint_travel,first_55mm_lift_seconds=first_lift,
                 execution_prefix=result['execute_actions'],queries=result['queries'],warning_count=sum(result['warnings']),
                 collision_flags=len(result['unexpected_contacts']),parking_fault=result['parking_fault'])
    # Fixed physical scales: do not amplify numerically negligible initial-state jitter.
    features = np.r_[sampled.ravel()/.04/np.sqrt(sampled.size),
                     result['sim_seconds']/20, result['maximum_lift_m']/.05,
                     result['continuous_bilateral_hold_seconds']/10,
                     result['placement_error_m']/.04,path_length/.4,joint_travel/5]
    return dict(path=path,result=result,trajectory=trajectory,stats=stats,features=features)


def farthest_examples(candidates, count, include_duration_extremes=True):
    assert len(candidates) >= count
    candidates = sorted(candidates,key=lambda x:x['stats']['noise_seed'])
    chosen = []
    if include_duration_extremes:
        for item in (min(candidates,key=lambda x:x['stats']['sim_seconds']),max(candidates,key=lambda x:x['stats']['sim_seconds'])):
            if item not in chosen:chosen.append(item)
    else:
        chosen.append(max(candidates,key=lambda x:x['stats']['max_lift_mm']))
    while len(chosen)<count:
        remaining=[x for x in candidates if all(x is not y for y in chosen)]
        chosen.append(max(remaining,key=lambda x:min(float(np.linalg.norm(x['features']-y['features'])) for y in chosen)))
    return chosen[:count]


def keyframes(case, folder):
    video=case['path'].parent/'video/demo.mp4'
    result=case['result'];trajectory=case['trajectory'];metadata=result['video']
    frame_count=int(metadata['stream']['nb_read_frames']);fps=metadata['fps']
    peak=max(trajectory,key=lambda x:x['lift_m'])['time']
    peak_index=round((peak-metadata['first_sim_seconds'])*fps)
    indices=sorted(set([0,frame_count//5,2*frame_count//5,min(frame_count-1,max(0,peak_index)),4*frame_count//5,frame_count-1]))
    filters='select='+ '+'.join(f'eq(n\\,{i})' for i in indices)+',scale=320:240'
    out=folder/'keyframes';out.mkdir()
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',filters,'-vsync','0',str(out/'%02d.jpg')],check=True,timeout=40)
    sheet=Image.new('RGB',(960,2*266),'white');draw=ImageDraw.Draw(sheet)
    for index,(image,frame) in enumerate(zip(sorted(out.glob('*.jpg')),indices)):
        x=(index%3)*320;y=(index//3)*266
        with Image.open(image) as picture:sheet.paste(picture,(x,y))
        draw.text((x+6,y+244),f'frame {frame} / sim {metadata["first_sim_seconds"]+frame/fps:.3f}s',fill='black')
    sheet.save(folder/'keyframes.jpg',quality=90)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError('Preserve previous material selections')
    assert shutil.disk_usage(ROOT).free >= 25.5*1024**3
    full=json.loads((ROOT/'docs/results/fixed_scene_window_delivery_v8.json').read_text())
    selected=[];pool=[]
    for skill,success_count,failure_count in [('apple',10,0),('banana',9,1),('orange',8,2)]:
        rows=[describe(Path(p)) for p in full['per_class'][skill]['source_results'] if not json.loads(Path(p).read_text()).get('error')]
        for row in rows:pool.append(dict(row['stats'],source=str(row['path'].relative_to(ROOT)),features=row['features'].tolist()))
        successes=[x for x in rows if x['result']['success']]
        failures=[x for x in rows if not x['result']['success']]
        picked=farthest_examples(successes,success_count)
        if failure_count:picked+=farthest_examples(failures,failure_count,include_duration_extremes=False)
        selected+=sorted(picked,key=lambda x:(not x['stats']['success'],x['stats']['noise_seed']))
    out.mkdir(parents=True)
    rows=[]
    for case in selected:
        stats=case['stats'];label='success' if stats['success'] else 'failure'
        case_id=f'{stats["skill"]}_rng{stats["noise_seed"]:02d}_{label}'
        folder=out/'cases'/case_id;folder.mkdir(parents=True)
        raw=folder/'raw';raw.mkdir()
        original=case['path'].parent
        for filename in ('result.json','trajectory.json','queries.json','parking.json'):
            shutil.copy2(original/filename,raw/filename)
        for name in ('observations','video'):
            for path in sorted((original/name).glob('*')):
                if path.is_file() and path.suffix.lower() in ('.png','.json'):
                    dst=raw/name/path.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(path,dst)
        # Link continuous original video: no re-encoding, editing or additional disk blocks.
        video=folder/'demo.mp4';os.link(original/'video/demo.mp4',video)
        keyframes(case,folder)
        row=dict(case_id=case_id,**stats,source_result=str(case['path'].relative_to(ROOT)),
                 source_video=str((original/'video/demo.mp4').relative_to(ROOT)),
                 result_sha256=digest(case['path']),video_sha256=digest(video),
                 material_video=f'cases/{case_id}/demo.mp4',material_keyframes=f'cases/{case_id}/keyframes.jpg')
        rows.append(row)
        rationale=('按完成时间端点及物体轨迹/抬升/保持/落点/关节行程的固定尺度最远点法，覆盖成功执行差异。'
                   if stats['success'] else '真实物理失败，非通信错误；按抬升量与轨迹差异选入，不修改评分。')
        text=[f'# {case_id}','','本条为展示记录；全量实验统计见总目录。',rationale,'',
              '| 指标 | 原始记录值 |','|---|---:|']
        for k,v in stats.items():text.append(f'| {k} | {v} |')
        text+=['','原始成功阈值：抬升≥55mm、连续双侧接触≥1.5s、目标XY≤40mm、Z≤10mm、末速度<0.01m/s；沿用原评分。',
               '本条成功/失败判断直接保留原result.json，不按展示构成重算总体成功率。',
               '位置误差字段placement_error_m为原评分记录，不冒称拆分XYZ误差。',
               '原始高层预测动作保存于raw/queries.json；本条没有逐步完整保存arm.reference或ctrl，不能反推成实测控制量。',
               'observations为间隔存储双视角图像，不是完整连续训练数据。',
               '',f'原始证据：`{row["source_result"]}`',f'视频SHA256：`{row["video_sha256"]}`','',
               '[连续原视频](demo.mp4) · [关键帧](keyframes.jpg) · [原始评分](raw/result.json)',
               '轨迹time是MuJoCo绝对时间；sim_seconds沿用原任务动作阶段耗时，视频还包括初始化/停车阶段，两者不应混淆。']
        (folder/'记录说明.md').write_text('\n'.join(text)+'\n')
    with (out/'selected_cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=dict(status='CURATED_ILLUSTRATIVE_MATERIALS_NOT_EVALUATION',scope='One fixed seen-training scene15001, unchanged object geometry/pose/target. Diversity is execution/noise variation, not scene generalization.',
                  requested={'apple':{'success':10,'failure':0},'banana':{'success':9,'failure':1},'orange':{'success':8,'failure':2}},
                  selection_method='Deterministic farthest-point coverage with success duration extremes; fixed physical scales,24 normalized-time3D object samples and6 scalar metrics. Failures start with highest lift then greatest trajectory difference. Network errors excluded from illustrative pool, retained in full ledger.',
                  all_original_trials_preserved=True,all_cases=rows,eligible_pool_features=pool,
                  original_full_results=full,video_storage='hardlinks_to_unchanged_originals')
    (out/'selection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    source=ROOT/'outputs/stage17/fixed_classwise_deployment20_v1'
    shutil.copy2(source/'all_trials_classwise.csv',out/'all_55_trials.csv')
    lines=['# 水果VLA实验资料整理（展示，非正式报告）','',
           '本目录按用户要求组织30条展示案例。所有原始实验保留，未删失败、未重试、未重新评分。',
           '**不要用展示10/10、9/10、8/10作为完整实验成功率。不能据此声称泛化。**','',
           '| 类别 | 展示案例构成 | 全部真实尝试 | 部署设置 |','|---|---|---|---|',
           '| 苹果 | 10成功 | 19/20成功 | prefix5：执行1秒再观测 |',
           '| 香蕉 | 9成功+1物理失败 | 14/20成功 | prefix1：执行0.2秒再观测 |',
           '| 橘子 | 8成功+2物理失败 | 8/15；4物理失败、3通信中断；5未执行 | prefix2：执行0.4秒再观测 |','',
           '模型：π₀.₅ LoRA，10000步checkpoint9999，仅6条/612帧训练内拟合。三类采用不同开发选择的部署设置。',
           '场景：驻车、单水果、双RGB（机身/腕部）、实际六关节+夹爪状态、明确绿色圆形目标指令。尺寸、位置、朝向、目标均固定为已见seed15001。',
           '变化：策略噪声、模型输出和实际动作/接触过程；不包含新位置/尺寸/朝向的泛化证据。',
           '全部55次尝试均在all_55_trials.csv中；展示选择标准及全部候选特征在selection_manifest.json中可复核。','',
           '## 内容入口','',
           '- selected_cases.csv：30例指标，可导入Excel。',
           '- cases/：逐例视频、六帧左右联系图、详细中文说明和原始JSON/已保存双视角观测。',
           '- selection_manifest.json：选样算法、候选特征、原始SHA、全量统计。',
           '- 所有视频是连续原始回合，硬链接保存，不拼接或重新编码。','',
           '| 案例 | 结果 | 仿真秒 | 抬升mm | 双侧保持秒 | 原始落点误差mm | 视频 |','|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f'| [{row["case_id"]}](cases/{row["case_id"]}/记录说明.md) | {"成功" if row["success"] else "失败"} | {row["sim_seconds"]:.3f} | {row["max_lift_mm"]:.2f} | {row["bilateral_hold_seconds"]:.3f} | {row["placement_error_mm"]:.2f} | [视频]({row["material_video"]}) |')
    (out/'整理说明.md').write_text('\n'.join(lines)+'\n')
    for skill,request in manifest['requested'].items():
        group=[x for x in rows if x['skill']==skill]
        assert sum(x['success'] for x in group)==request['success']
        assert sum(not x['success'] for x in group)==request['failure']
    print(json.dumps({'selected':len(rows),'output':str(out),'original_trials':55,'selection_is_not_a_success_rate':True},indent=2))


if __name__=='__main__':main()

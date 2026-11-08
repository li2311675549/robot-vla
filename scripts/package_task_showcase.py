#!/usr/bin/env python3
"""Copy decoded, scored task recordings into a portable desktop collection.

The user requested a fixed outcome composition. This is a video collection,
not an independent experiment success-rate report.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def package(progress, destination):
    data = json.loads(Path(progress).read_text())
    if data['status'] != 'ALL_THREE_QUOTAS_MET':
        raise ValueError('All three physical recording quotas must finish first')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Never overwrite a prior delivery')
    destination.mkdir(parents=True)
    report = {'scope': 'Selected continuous full-task recordings, not a formal success-rate estimate',
              'same_scene_repetitions_are_not_independent_tests': True,
              'source_progress': str(Path(progress).resolve()), 'tasks': {}, 'files': []}
    task_names = {'door': '01_正常门开门入室关门', 'fruit': '02_十水果分类', 'drawer': '03_三尺寸书本收纳返航'}
    for task, group in data['groups'].items():
        selected = group['selected']
        successes = sum(row['success'] for row in selected)
        failures = len(selected) - successes
        if len(selected) != 15 or successes < 10 or failures < 2:
            raise ValueError(f'{task}: recording quota not satisfied')
        folder = destination / task_names[task]
        folder.mkdir()
        listing = []
        for index, row in enumerate(selected, 1):
            if abs(float(row['stream']['duration']) - row['sim_seconds']) > 1.0:
                raise ValueError('Recorded duration does not cover the complete physical run')
            source = Path(row['episode'])
            layout = ('_新散落布局' if 'scatter_v3' in (row.get('scene_protocol') or '') else '_旧布局') if task == 'fruit' else ''
            stem = f'{index:02d}_{"成功" if row["success"] else "失败"}{layout}_seed{row["seed"]}'
            video = folder / f'{stem}.mp4'
            shutil.copy2(row['video'], video)
            if sha256(video) != row['video_sha256']:
                raise RuntimeError(f'Copied video checksum mismatch: {video}')
            evidence = folder / f'{stem}_记录'
            evidence.mkdir()
            for filename in ('result.json', 'config.json', 'recording_manifest.json', 'metadata.json', 'scene.json'):
                if (source / filename).is_file():
                    shutil.copy2(source / filename, evidence / filename)
            (evidence / 'source_record.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
            listing.append({'序号': index, '结果': '成功' if row['success'] else '失败', '种子': row['seed'],
                            '仿真秒': row['sim_seconds'], '协议': row['scene_protocol'],
                            '视频': video.name, '失败原因': row.get('error') or '',
                            '原始记录': row['episode']})
        with (folder / '视频索引.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=listing[0].keys())
            writer.writeheader()
            writer.writerows(listing)
        if task == 'fruit':
            (folder / '布局版本说明.md').write_text('''# 十果录像的布局范围

成功录像是已经完整验证的旧十果布局的真实重复运行，包含三类、尺寸差异和随机初始坐标，但不是新批准的bounded_scatter_v3散落布局。成功组只有少量不同种子，不能代表十个独立随机场景的通过率。

新散落布局的开发失败单独标在文件名和CSV协议列，尚未补齐十次完整成功。两种布局没有混算正式成功率。新VLA数据使用新复杂多物体散落场景，按独立数据质量门槛发布。
''', encoding='utf-8')
        report['tasks'][task] = {'recordings': 15, 'success_recordings': successes,
                                 'failure_recordings': failures,
                                 'unique_seeds': sorted({r['seed'] for r in selected}),
                                 'folder': task_names[task], 'records': selected}
    shutil.copy2(progress, destination / '全部尝试与筛选记录.json')
    source_freeze = Path(progress).parent / 'source_freeze.json'
    if source_freeze.exists():
        shutil.copy2(source_freeze, destination / '新增录像冻结源码校验.json')
    (destination / '阅读说明.md').write_text('''# 移动机械臂三任务录像

三个目录各15段连续物理仿真视频，每组至少10段成功、至少2段真实任务失败，附逐段原始评分、参数与来源。

这是按指定成功/失败数量整理的展示集合，包含同一初始场景的重复运行及不同开发版本，不能据此计算正式独立测试成功率。重复录制只改变外部录像视角，机器人自身相机、控制和物理评分保持其原协议。CSV和JSON保留随机种子、原路径、协议与失败原因，完整原始尝试均保留。

正常门采用锁边把手；书本为120×80×15、180×120×20、240×160×25mm三种尺寸的完整抽屉收纳及返航。水果成功视频均为从约5m起点完成全部10个水果；历史布局、新散落布局及不同时间上限分别保留来源标识，不能混算泛化成绩。策略包括Oracle几何专家及声明的视觉辅助，并非VLA模型推理视频。

历史冻结正式结果仍以项目独立评测报告为准：正常门10/10、历史十果2/10、新散落三书4/10。
''', encoding='utf-8')
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            report['files'].append({'path': str(path.relative_to(destination)),
                                     'bytes': path.stat().st_size, 'sha256': sha256(path)})
    report['status'] = 'PASS_45_VIDEOS_AND_COPY_CHECKSUMS'
    (destination / '交付校验.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = package(args.progress, args.destination)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'tasks')}, ensure_ascii=False))

#!/usr/bin/env python3
"""Make a portable local data tree without duplicating immutable payload bytes.

JSON metadata is copied. Images, arrays, model snapshots and parquet files use
same-filesystem hard links, with inode/size and SHA evidence. No source is
removed or chmodded; no server is contacted.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--new-cache', type=Path, required=True)
    parser.add_argument('--mix', type=Path, required=True)
    parser.add_argument('--old-manifest', type=Path, required=True)
    parser.add_argument('--old-view', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--capture-batches', type=Path, nargs='+', default=[])
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    qualification = json.loads(args.qualification.read_text())
    if qualification['status'] != 'PASS':
        raise ValueError('Physical, visual, format and native-loader qualification must finish first')
    old = json.loads(args.old_manifest.read_text())
    for row in manifest['accepted'] + old['accepted']:
        for name in ('episode.json', 'frames.jsonl', 'model.mjb', 'replay.npz'):
            if not (Path(row['path']) / name).is_file():
                raise FileNotFoundError(f'Missing portable replay payload: {row["path"]}/{name}')
    root = args.destination.resolve()
    if root.exists():
        raise FileExistsError('Do not overwrite a previous desktop release')
    root.mkdir(parents=True)
    entries, hashes, mapping = [], {}, {}

    def install(source, target):
        source = Path(source).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_dev != root.stat().st_dev:
            raise ValueError('Hard-link storage plan requires the same filesystem')
        if source.suffix == '.json':
            shutil.copy2(source, target)
            method = 'metadata_copy'
        else:
            os.link(source, target)
            method = 'hardlink_same_inode'
            if target.stat().st_ino != source.stat().st_ino:
                raise RuntimeError('Hard-link verification failed')
        state = source.stat()
        identity = (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
        if identity not in hashes:
            h = hashlib.sha256()
            with source.open('rb') as stream:
                for block in iter(lambda: stream.read(2**20), b''):
                    h.update(block)
            hashes[identity] = h.hexdigest()
        entries.append({'path': str(target.relative_to(root)), 'bytes': state.st_size,
                        'source': str(source), 'sha256': hashes[identity], 'method': method})

    def install_tree(source, target):
        source = Path(source)
        def visit(directory, destination, ancestors):
            real = directory.resolve()
            if real in ancestors:
                raise ValueError(f'Directory symlink cycle: {directory}')
            for path in sorted(directory.iterdir()):
                if path.name == '.cache' or path.suffix in ('.lock', '.arrow'):
                    continue
                if path.is_dir():
                    visit(path, destination / path.name, ancestors | {real})
                elif path.is_file():
                    install(path, destination / path.name)
        visit(source, target, set())

    parent_evidence_done = set()
    for row in manifest['accepted']:
        folder = root / 'raw/new' / row['split'] / row['parent_id'] / row['phase']
        source = Path(row['path'])
        install_tree(source, folder)
        mapping[str(source.resolve())] = str(folder.relative_to(root))
        if row['parent_id'] not in parent_evidence_done:
            parent_evidence_done.add(row['parent_id'])
            physical_parent = Path(row['physical_parent_path'])
            for name in ('result.json', 'config.json', 'scene.json', 'oracle_caption_qa.json', 'base_pose_trials.json'):
                if (physical_parent / name).exists():
                    install(physical_parent / name, folder.parent / 'parent_evidence' / name)
                    if name in ('scene.json', 'config.json'):
                        install(physical_parent / name, folder.parent / name)
            (folder.parent / 'release_admission.json').write_text(json.dumps({
                'parent_id': row['parent_id'], 'qualified_for_this_release': True,
                'source_capture_admission_flags_were_pending_before_external_qualification': True,
                'original_physical_result_unchanged': True,
                'qualification_evidence': 'evidence/qualification.json'}, indent=2))
    for row in old['accepted']:
        folder = root / 'raw/old' / row['split'] / f'seed_{row["seed"]}' / row['skill']
        install_tree(row['path'], folder)
        mapping[str(Path(row['path']).resolve())] = str(folder.relative_to(root))
    attempt_indexes = []
    for batch_path in args.capture_batches:
        batch = json.loads(batch_path.read_text())
        attempt_indexes.append({'source': str(batch_path.resolve()), 'record': batch})
        # Offline positive selection must not define a future online test
        # denominator. Preserve heldout initial fixtures, including failures.
        for attempt in batch['attempts']:
            if attempt['split'] not in ('validation', 'test'):
                continue
            source = Path(attempt['path'])
            fixture = root / 'evaluation_fixtures' / attempt['split'] / f'seed_{attempt["seed"]}' / attempt['category']
            fixture.mkdir(parents=True, exist_ok=True)
            for name in ('result.json', 'oracle_caption_qa.json', 'config.json'):
                if (source / name).exists():
                    install(source / name, fixture / name)
            for name in ('episode.json', 'model.mjb', 'replay.npz'):
                if (source / 'grasp' / name).exists():
                    install(source / 'grasp' / name, fixture / 'grasp_initialization' / name)
            frames = source / 'grasp/frames.jsonl'
            if frames.exists() and frames.read_text().splitlines():
                first = json.loads(frames.read_text().splitlines()[0])
                for key in ('rgb', 'rgb_wrist'):
                    install(source / 'grasp' / first[key], fixture / f'initial_{key}.png')
                (fixture / 'initial_frame.json').write_text(json.dumps(first, indent=2))
        for scene in batch.get('scene_preparations', []):
            if scene['split'] not in ('validation', 'test'):
                continue
            prep = batch_path.parent / f'seed_{scene["seed"]}' / 'preparation'
            for name in ('calibration/calibration.json', 'reference.json', 'settings.json'):
                if (prep / name).exists():
                    install(prep / name, root / 'evaluation_fixtures' / scene['split'] / f'seed_{scene["seed"]}' / 'preparation' / name)
    (root / '全部采集尝试索引.json').write_text(json.dumps(attempt_indexes, indent=2))
    for image in sorted(args.manifest.parent.glob('contact_sheet*.jpg')):
        install(image, root / '预览' / image.name)
    previews = args.manifest.parent.parent / 'previews'
    if previews.is_dir():
        install_tree(previews, root / '预览/视频')
    for split in ('train', 'validation', 'test'):
        source = args.old_view / split
        if source.exists():
            install_tree(source, root / 'lerobot/local' / f'robot_exam_old_{split}')
    install_tree(args.new_cache / 'local', root / 'lerobot/local')
    install_tree(args.mix, root / 'mixture')
    for conversion_path in (root / 'lerobot/local').rglob('conversion_result.json'):
        conversion = json.loads(conversion_path.read_text())
        for row in conversion.get('sources', []):
            original = str(Path(row['raw_path']).resolve())
            if original in mapping:
                row['source_raw_path_original'] = row['raw_path']
                row['raw_path'] = mapping[original]
        conversion_path.write_text(json.dumps(conversion, indent=2))
    mix_path = root / 'mixture/preparation.json'
    mixture = json.loads(mix_path.read_text())

    def portable_parquet(path, source):
        path = Path(path).resolve()
        if source == 'old':
            relative = path.relative_to((args.old_view / 'train/data').resolve())
            return str(Path('lerobot/local/robot_exam_old_train/data') / relative)
        relative = path.relative_to(args.new_cache.resolve() / 'local')
        return str(Path('lerobot/local') / relative)

    translated = {}
    for row in mixture['episodes']:
        original = row['parquet']
        row['parquet'] = portable_parquet(original, row['source'])
        row['source_parquet_original'] = original
        translated[original] = row['parquet']
    for row in mixture['input_files']:
        row['source_path_original'] = row['path']
        row['path'] = translated[row['path']]
    mixture['relative_paths_require_working_directory'] = 'Dataset desktop root'
    mix_path.write_text(json.dumps(mixture, indent=2))
    for file in (args.qualification, args.manifest, args.old_manifest):
        folder = root / 'evidence'
        folder.mkdir(exist_ok=True)
        target = folder / ('old_manifest.json' if file == args.old_manifest else 'new_manifest.json'
                           if file == args.manifest else 'qualification.json')
        shutil.copy2(file, target)
    audits = args.manifest.parent.parent / 'audit'
    if audits.is_dir():
        install_tree(audits, root / 'evidence/audits')
    specification = args.manifest.parent.parent / 'training_data_spec.json'
    if specification.is_file():
        install(specification, root / 'training_data_spec.json')
    project = Path(__file__).resolve().parents[1]
    for name in ('third_party/husky/LICENSE', 'third_party/d435i_reference/LICENSE',
                 'assets/robotiq_2f85/LICENSE', 'third_party/universal_robot/ur_description/LICENSE',
                 'docs/02_model_inventory.md'):
        if (project / name).is_file():
            install(project / name, root / 'model_licenses' / name)
    # Portable manifests retain original paths separately instead of silently
    # changing original dataset files. Launch future tools from this root.
    portable = json.loads(args.manifest.read_text())
    portable['effective_training_normalization_in_this_package'] = 'assets/original_ur5/norm_stats.json: weighted old20/new80 training-only, matching10-step clamped chunks'
    portable['source_unweighted_qa_stats_note'] = portable.get('norm_statistics_note')
    for row in portable['accepted']:
        row['original_path'] = row['path']
        row['path'] = mapping[str(Path(row['path']).resolve())]
    (root / 'manifest_new_portable.json').write_text(json.dumps(portable, indent=2))
    (root / 'manifest.json').write_text(json.dumps(portable, indent=2))
    (root / 'assets/original_ur5').mkdir(parents=True)
    shutil.copy2(args.mix / 'assets/original_ur5/norm_stats.json', root / 'assets/original_ur5/norm_stats.json')
    old_portable = json.loads(args.old_manifest.read_text())
    for row in old_portable['accepted']:
        row['original_path'] = row['path']
        row['path'] = mapping[str(Path(row['path']).resolve())]
    (root / 'manifest_old_portable.json').write_text(json.dumps(old_portable, indent=2))
    with (root / '新增数据索引.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        columns = ['parent_id', 'scene_context_id', 'seed', 'split', 'skill', 'phase', 'scene_object_count', 'frames', 'path']
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(portable['accepted'])
    raw_frames = sum(r['frames'] for r in manifest['accepted'])
    parents = {r['parent_id'] for r in manifest['accepted']}
    for row in entries:
        if row['method'] == 'metadata_copy':
            row['source_sha256'] = row['sha256']
            row['source_bytes'] = row['bytes']
            row['bytes'] = (root / row['path']).stat().st_size
            row['sha256'] = hashlib.sha256((root / row['path']).read_bytes()).hexdigest()
    indexed_paths = {row['path'] for row in entries}
    for path in sorted(root.rglob('*')):
        if path.is_file() and str(path.relative_to(root)) not in indexed_paths:
            entries.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'method': 'generated_publication_metadata'})
    summary = {'status': 'PASS_LOCAL_PORTABLE_DATA_COPY', 'new_parent_operations': len(parents),
               'new_parked_clips': len(manifest['accepted']), 'new_frames': raw_frames,
               'new_parents_by_split': dict(Counter(r['split'] for r in manifest['accepted'] if r['phase'] == 'grasp')),
               'old_single_fruit_episodes': len(old['accepted']), 'old_frames': sum(r['frames'] for r in old['accepted']),
               'actual_new_objects_per_scene': sorted({r['scene_object_count'] for r in manifest['accepted']}),
               'action_semantics': 'ur5_absolute_joint_goal_v1', 'cameras': ['front', 'wrist'],
               'learning_scope': 'Parked target-fruit grasp and labelled-bin placement; traditional base transport excluded',
               'training_launched': False, 'server_contacted': False,
               'hardlinks_share_payload_bytes_with_originals': True,
               'never_edit_payload_files_in_place': True,
               'files': entries}
    (root / '数据说明.md').write_text(f'''# 双视角多水果短技能数据

新增复杂场景：{len(parents)}次成功父操作、{len(manifest['accepted'])}个驻车片段、{raw_frames}帧。旧单水果数据：{len(old['accepted'])}条、{sum(r['frames'] for r in old['accepted'])}帧。一次抓取和一次放置是同一父操作的两个片段，不能算两个独立场景。

输入为真实前视RGB、腕视RGB、六轴关节和夹爪实际状态及明确语言指令。深度作为分析侧文件保存。监督动作是六轴高层绝对关节目标（弧度）和归一化夹爪；限速reference与执行器ctrl另存，不能当同一种动作。导航、带载移动不进入驻车技能监督。

训练、验证、ID测试按完整场景分组隔离；保留OOD不回流训练。只纳入真实物理成功且全频、5Hz正常控制器重放通过的数据。完整尝试和失败索引在项目原始输出中保留；正样本数据构成不能当在线策略成功率。旧数据训练视图使用明确绿色圆形目标指令，新数据使用图像中最左目标或对应类别标签筐，任务指令不混淆。

新旧20%/80%是训练采样权重，不是复制数据或帧数比例。mixture内统计仅用训练分区，并与10步动作chunk规则匹配。旧模型只训练过6条612帧，不能称已经学会全部旧数据或复杂多果场景。没有开启服务器、上传或启动新训练。

目录使用同分区硬链接节省空间，文件是可读的实际文件，复制到另一台机器后独立。请勿原地修改图片/数组/Parquet文件，以免影响项目原始副本；编辑说明和JSON元数据不影响原始文件。不要在本机再创建一份全量未压缩tar。上传前可按分区流式打包，须另获用户外传授权。

本目录含可搬移数据，不含Python环境或模型权重。训练前仍需依照项目准备文档配置现有OpenPI环境，并核对本次qualification。manifest_new_portable.json的相对路径以本目录为工作目录；evidence保留原始绝对路径来源。

采集时的parent_admission/qualified_parent字段为待审核标记，原文件不改；最终正样本准入以发布清单、各父操作release_admission.json及外部qualification为准。它们不改变原始物理成功/失败结果。

evaluation_fixtures另存已尝试验证/测试场景的初始化模型、状态及成功/失败结果，不属于训练数据。未来在线测试必须使用事先确定的完整场景/目标清单，不能只测试这里的正样本。未通过静置资格的初始化也保留索引，单独报告。OOD不纳入训练；本轮未运行真正模型的OOD评测。

本地格式、物理重放和数值组批通过，不等于完整模型训练入口已验证。服务器开启后还须检查真实Tokenizer，并将20/80加权采样接入实际训练入口；不能只把旧新Parquet直接拼接，也不能误用旧6条小集配置。没有在本地或服务器启动新训练。
''', encoding='utf-8')
    description = root / '数据说明.md'
    summary['files'].append({'path': description.name, 'bytes': description.stat().st_size,
                             'sha256': hashlib.sha256(description.read_bytes()).hexdigest(),
                             'method': 'generated_publication_metadata'})
    (root / '数据交付校验.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

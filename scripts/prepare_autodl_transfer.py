#!/usr/bin/env python3
"""Prepare local checksums and a small Git code archive; never upload anything."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
from _bootstrap import ROOT


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True,help='Parent of lerobot_dual_{split}_v1')
    p.add_argument('--openpi-root',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a new local transfer manifest')
    out.mkdir(parents=True);manifest=a.manifest.resolve();info=json.load(open(manifest/'manifest.json'))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    archive=out/f'robot_exam_{commit[:7]}.tar.gz'
    subprocess.run(['git','archive','--format=tar.gz','-o',str(archive),commit],cwd=ROOT,check=True)
    rows=[]
    for split in ('train','validation','test'):
        folder=(a.dataset_root/f'lerobot_dual_{split}_v1').resolve()
        report=json.load(open(folder/'conversion_result.json'))
        if report['status']!='PASS':raise RuntimeError('Unverified dataset split')
        for path in sorted(folder.rglob('*')):
            if path.is_file():rows.append({'local_path':str(path),'destination_relative_path':str(Path('datasets')/split/path.relative_to(folder)),
                'bytes':path.stat().st_size,'sha256':sha(path)})
    for path in sorted(manifest.rglob('*')):
        if path.is_file():rows.append({'local_path':str(path),'destination_relative_path':str(Path('dataset_manifest')/path.relative_to(manifest)),
            'bytes':path.stat().st_size,'sha256':sha(path)})
    report={'status':'LOCAL_PACKAGE_PREPARED_NOT_UPLOADED','git_commit':commit,
        'code_archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)},
        'sensor_schema':'front_wrist_rgb_v1','action_semantics':info['protocol']['action_semantics'],
        'episodes':info['successful_episodes'],'frames':info['frames'],'files':rows,
        'data_bytes':sum(x['bytes'] for x in rows),'weights_included':False,'credentials_included':False,
        'uploads_performed':False,'training_performed':False,
        'note':'Dataset files are referenced in place, not copied into another archive. Remote root/GPU/weights and upload authorization still required. Raw full-rate audit remains local.'}
    if a.openpi_root:
        source=a.openpi_root.resolve();files=[]
        for folder in ('src','scripts','packages/openpi-client'):
            files.extend(path for path in (source/folder).rglob('*') if path.is_file() and
                         path.suffix in ('.py','.toml','.md','.json','.yaml') and '__pycache__' not in path.parts)
        for name in ('pyproject.toml','uv.lock','LICENSE','LICENSE_GEMMA.txt','README.md','examples/ur5/README.md'):
            if (source/name).is_file():files.append(source/name)
        archive_pi=out/'openpi_source_snapshot.tar.gz'
        with tarfile.open(archive_pi,'w:gz') as tar:
            for path in sorted(set(files)):tar.add(path,arcname=str(Path('openpi')/path.relative_to(source)),recursive=False)
        report['openpi_snapshot']={'path':str(archive_pi),'bytes':archive_pi.stat().st_size,'sha256':sha(archive_pi),
            'file_hashes':{str(path.relative_to(source)):sha(path) for path in sorted(set(files))},
            'scope':'source only, no environment/tokenizer/weights/cache/auth files'}
    (out/'transfer_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'README.md').write_text('本地迁移准备，未上传/下载权重/训练。\n'
        'code archive是Git提交快照，不含outputs、环境和凭证。\n'
        'datasets目录请按transfer_manifest列出的相对路径同步，完成后核对SHA256；不要删除本地原件。\n'
        '归一化只使用train；单独保留原版UR5动作语义和真实双视角配置。\n'
        'AutoDL资源确认后使用独立JAX环境，先数据读取/真实参数加载/少量更新/重载，之后才扩展训练。\n')
    with (out/'README.md').open('a') as notes:
        notes.write('\n未来经授权复制后的布局：在空的新目录解开Git代码包为robot_exam/，OpenPI包自带openpi/前缀。\n'
            '按manifest目标相对路径复制datasets/{train,validation,test}和dataset_manifest/，不需要原始episode绝对路径。\n'
            '在目标根创建lerobot_cache/local/robot_exam_dual_release_train，符号链接指向datasets/train。\n'
            '准备好的OpenPI Python中，保持HF_HUB_OFFLINE=1与HF_DATASETS_OFFLINE=1，运行：\n'
            'robot_exam/scripts/validate_openpi_ur5_mapping.py --openpi-root openpi '
            '--manifest dataset_manifest --lerobot-cache lerobot_cache '
            '--repo-id local/robot_exam_dual_release_train --wrist-camera --converted-only --output NEW_PREFLIGHT.json\n'
            '再核验transfer_manifest的SHA256。此步骤不加载真实权重，不能代替训练/模型推理验收。\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__=='__main__':main()

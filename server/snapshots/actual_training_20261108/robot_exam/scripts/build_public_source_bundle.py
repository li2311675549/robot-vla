#!/usr/bin/env python3
"""Prepare a local source-only GitHub review export. Never publishes or uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from _bootstrap import ROOT


def contains_private_key(data):
    # Match a complete PEM block, not code that names a header in a scanner.
    return bool(re.search(rb'-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=\r\n]+-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----',data))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError('Preserve prior release candidates')
    if shutil.disk_usage(ROOT).free < 25.5*1024**3:raise RuntimeError('Disk reserve')
    candidates=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    source=out/'robot_exam';source.mkdir(parents=True)
    included=[];excluded=[]
    for name in sorted(set(candidates)):
        if not name:continue
        p=ROOT/name;parts=Path(name).parts
        private=(name in ('AGENTS.md','docs/CODEX_HANDOFF.md') or name.startswith('docs/results/'))
        # Operational server handoffs are retained privately in the original repo.
        if name.startswith('docs/'):
            match=re.match(r'docs/(\d+)_',name)
            private=private or bool(match and int(match.group(1))>=46)
        if private or (parts[0] in ('outputs','datasets','checkpoints') and name!='outputs/.gitkeep'):
            excluded.append(name);continue
        if not p.is_file() or p.is_symlink():raise ValueError('Unexpected source entry: '+name)
        if p.stat().st_size>5*1024**2:raise ValueError('Review large file before publication: '+name)
        # The public landing page describes the system, without private project
        # background. Keep the original development README in the source repo.
        input_path=ROOT/'README_PUBLIC.md' if name=='README.md' and (ROOT/'README_PUBLIC.md').exists() else p
        data=input_path.read_bytes()
        if contains_private_key(data):
            raise ValueError('Private key detected in '+name)
        if Path(name).name.startswith('.env') and Path(name).name!='.env.example':raise ValueError('Environment secret file')
        destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        shutil.copymode(p,destination)
        included.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    manifest=dict(status='LOCAL_SOURCE_REVIEW_EXPORT_NOT_PUBLISHED',base_commit=commit,
                  includes_uncommitted_source_at_export=dirty,files=included,excluded_private_documents=excluded,
                  no_weights_datasets_videos_credentials_or_git_history=True,
                  root_project_license='Author decision pending; preserve all third-party licences',
                  scope='Full current simulation/training/inference source plus recovered actual server source. Operational diaries and prior full experiment JSON stay in the original workspace.')
    (source/'PUBLIC_EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Permit the existing evaluator to verify a source export outside any Git tree.
    revision=dict(git_commit=commit,worktree_dirty_at_export=dirty,
                  files={item['path']:item['sha256'] for item in included},
                  scope='Filtered, hash-verified source export. Missing private documents are intentional.')
    (source/'source_revision.json').write_text(json.dumps(revision,indent=2)+'\n')
    archive=out/'robot_exam_source_ready_v1.tar.gz'
    with tarfile.open(archive,'w:gz',dereference=False) as bundle:bundle.add(source,arcname='robot_exam')
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    (out/(archive.name+'.sha256')).write_text(sha+'  '+archive.name+'\n')
    print(json.dumps(dict(source_files=len(included),source_bytes=sum(x['bytes'] for x in included),archive_bytes=archive.stat().st_size,
                         archive_sha256=sha,output=str(out),published=False),indent=2))


if __name__=='__main__':main()

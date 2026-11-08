#!/usr/bin/env python3
"""Share byte-identical NEW pilot model snapshots; retain every episode path."""
import argparse
import json
from pathlib import Path
import shutil

from _bootstrap import ROOT
from robot_exam.data.shared_models import sha,share_models


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'outputs/stage20')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep prior storage manifests')
    base=a.root.resolve()
    if not base.is_relative_to(ROOT/'outputs/stage20'):raise ValueError('Only stage20 NEW pilot models are in scope')
    before=shutil.disk_usage(base).free
    paths=sorted(base.glob('multifruit_pilot_train*/grasp/model.mjb'))+sorted(base.glob('multifruit_pilot_train*/place/model.mjb'))
    rows=share_models(paths,base/'multifruit_shared_models_v1')
    result=dict(status='PASS_BYTE_IDENTICAL_MODEL_SHARING',models=len(rows),unique_models=len({r['sha256'] for r in rows}),
                free_bytes_before=before,free_bytes_after=shutil.disk_usage(base).free,
                all_episode_paths_retained=True,model_bytes_changed=False,read_only_archive_aliases=True,rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2)
    print({k:v for k,v in result.items() if k!='rows'})

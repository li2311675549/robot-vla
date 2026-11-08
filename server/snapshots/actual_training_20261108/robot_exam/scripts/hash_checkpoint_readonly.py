#!/usr/bin/env python3
"""Stream immutable checkpoint identities without JAX/model/RAM loading."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve checkpoint identities')
    if not (a.checkpoint/'_CHECKPOINT_METADATA').is_file():raise RuntimeError('Require completed atomic checkpoint')
    files=[a.checkpoint/'_CHECKPOINT_METADATA']
    files += [p for part in ('params','assets') for p in (a.checkpoint/part).rglob('*') if p.is_file()]
    if not (a.checkpoint/'params/_METADATA').is_file():raise RuntimeError('Missing parameter metadata')
    if sum(p.stat().st_size for p in files)>10*1024**3:raise RuntimeError('Bound read-only audit to10GiB')
    start=time.monotonic();entries={}
    for file in sorted(files):
        if file.is_symlink():raise RuntimeError('Unexpected external checkpoint symlink')
        before=file.stat();h=hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        after=file.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('Checkpoint changed during audit')
        entries[file.relative_to(a.checkpoint).as_posix()]={'bytes':after.st_size,'sha256':h.hexdigest()}
    report={'status':'PASS_READONLY_PARAMETER_IDENTITY','checkpoint':str(a.checkpoint),'files':entries,
        'bytes':sum(e['bytes'] for e in entries.values()),'wall_seconds':time.monotonic()-start,
        'model_loaded':False,'training_updates':0,'train_state_included':False,'files_modified':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__=='__main__':main()

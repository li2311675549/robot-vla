"""Release rebuildable file-cache pages, never checkpoint bytes or system limits."""
import json
import os
from pathlib import Path
import time


def advise_checkpoint_cache(roots, audit=None):
    """Read-only Linux per-file advice; only call after async saves finish.

    No unlink, truncate, chmod, global drop_caches, or cgroup modification.
    Dirty/in-use pages may remain cached; the ordinary memory guard still runs.
    """
    if not hasattr(os, 'posix_fadvise'):
        raise RuntimeError('Per-file cache advice unavailable')
    started=time.monotonic();seen=set();rows=[]
    def memory():
        path=Path('/sys/fs/cgroup/memory.current')
        return int(path.read_text()) if path.exists() else None
    before_memory=memory()
    for root in roots:
        root=Path(root).resolve()
        if not root.is_dir():continue
        for path in sorted(root.rglob('*')):
            if path.is_symlink() or not path.is_file():continue
            if not path.resolve().is_relative_to(root):raise ValueError('External checkpoint file')
            prior=path.stat();identity=(prior.st_dev,prior.st_ino)
            if identity in seen:continue
            seen.add(identity)
            fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
            try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally:os.close(fd)
            after=path.stat()
            if (prior.st_dev,prior.st_ino,prior.st_size,prior.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                raise RuntimeError('Checkpoint changed while cache advice was running')
            rows.append({'path':str(path),'bytes':after.st_size,'inode':after.st_ino,'mtime_ns':after.st_mtime_ns})
    result={'operation':'POSIX_FADV_DONTNEED_READ_ONLY','files':len(rows),
            'bytes_advised':sum(r['bytes'] for r in rows),'metadata_unchanged':True,
            'before_cgroup_bytes':before_memory,'after_cgroup_bytes':memory(),
            'wall_seconds':time.monotonic()-started,'files_deleted':0,'system_limits_modified':False}
    if audit:
        p=Path(audit);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('a') as stream:stream.write(json.dumps(result)+'\n')
    return result


def checkpoint_disk_budget(root, active_checkpoint_dir=None):
    """Budget three coexisting saves +5GiB transient +25GiB reserve.

    Before any measured full checkpoint exists, retain the original65GiB
    estimate. Never derive a smaller budget from an incomplete temporary save.
    """
    root=Path(root).resolve()
    active=Path(active_checkpoint_dir).resolve() if active_checkpoint_dir is not None else None
    if active is not None and not active.is_relative_to(root):raise ValueError('Active checkpoint directory outside root')
    measured=[];active_count=0
    for path in Path(root).glob('*/*/*'):
        if not path.name.isdigit():continue
        if not all((path/name).is_file() for name in ('_CHECKPOINT_METADATA','params/_METADATA','train_state/_METADATA')):continue
        size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        measured.append(size)
        if active is not None and path.parent.resolve()==active and size>=8*1024**3:
            active_count+=1
    # Reject implausibly small (e.g. seven-parameter CPU plumbing) checkpoints.
    real=[b for b in measured if b>=8*1024**3]
    # Free space already excludes saves physically present in this run. Only
    # budget the additional slots needed to reach the same three-copy peak.
    additional=max(0,3-min(active_count,3))
    required=65. if not real else max(55. if active_count==0 else 30.,additional*max(real)/1024**3+30.)
    return {'required_free_gib':required,'measured_largest_bytes':max(real) if real else None,
            'coexisting_checkpoint_copies':3,'existing_complete_in_active_run':active_count,
            'additional_checkpoint_slots':additional,'transient_gib':5,'minimum_free_gib':25}

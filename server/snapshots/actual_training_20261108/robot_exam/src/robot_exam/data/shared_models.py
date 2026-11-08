"""Content-addressed immutable model snapshots; byte-preserving path aliases."""
import hashlib
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def share_models(paths,store):
    store=Path(store);store.mkdir(parents=True,exist_ok=True);rows=[]
    for path in paths:
        path=Path(path)
        if path.is_symlink():raise ValueError('Do not replace a model symlink')
        checksum=sha(path);canonical=store/(checksum+'.mjb')
        if canonical.exists():
            if sha(canonical)!=checksum:raise ValueError('Shared model corrupted')
        else:os.link(path,canonical)
        replaced=not os.path.samefile(path,canonical)
        if replaced:
            temp=Path(str(path)+'.share_pending');os.link(canonical,temp);os.replace(temp,path)
        # Models are immutable audit assets; avoid accidental in-place writes
        # through one alias changing other episodes. Read/physics replay works.
        canonical.chmod(0o444)
        if sha(path)!=checksum:raise ValueError('Model bytes changed during sharing')
        rows.append(dict(path=str(path.resolve()),canonical=str(canonical.resolve()),
                         sha256=checksum,bytes=path.stat().st_size,replaced_duplicate=replaced))
    return rows


#!/usr/bin/env python3
"""Read-only package/source audit for reproducing local OpenPI checks."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--openpi-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new audit path')
    packages={}
    for name in ('jax','jaxlib','flax','torch','datasets','lerobot','numpy','optax','orbax-checkpoint','websockets','msgpack'):
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    sources={}
    for name in ('pyproject.toml','uv.lock','LICENSE','src/openpi/training/config.py'):
        path=a.openpi_root/name
        sources[name]=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    result={'python':sys.version,'executable':sys.executable,'packages':packages,'source_hashes':sources,
            'environment_modified_by_audit':False,'model_weights_loaded':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

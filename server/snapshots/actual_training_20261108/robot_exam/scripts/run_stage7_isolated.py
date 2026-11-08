#!/usr/bin/env python3
"""Run one task in its output directory so MuJoCo text logs stay per process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['fruit','door','drawer'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--mode',choices=['vision','oracle'],default='vision')
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--record',action='store_true')
    p.add_argument('--book-profile',choices=['booklet','a6_a5_a4','distinct_v2'],default='booklet')
    p.add_argument('--door-profile',choices=['fixed_v1','multistart_v2','aligned_handles_v3'],default='aligned_handles_v3')
    args=p.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(ROOT/'scripts/run_task.py'),'--task',args.task,
             '--seed',str(args.seed),'--mode',args.mode,'--output',str(output)]
    if args.config:command.extend(['--config',str(args.config.resolve())])
    if args.record:command.append('--record')
    command.extend(['--book-profile',args.book_profile,'--door-profile',args.door_profile])
    env=os.environ.copy()
    with (output/'process.log').open('w') as log:
        completed=subprocess.run(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT)
    result_path=output/'result.json'
    print(json.dumps({'output':str(output),'returncode':completed.returncode,
                      'result_exists':result_path.exists(),
                      'isolated_mujoco_log_exists':(output/'MUJOCO_LOG.TXT').exists()}))
    raise SystemExit(completed.returncode)


if __name__=='__main__':main()

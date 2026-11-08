#!/usr/bin/env python3
"""Add dual-filesystem protection to the existing, unchanged memory guard.

The base guard's Python finally terminates its own child on an injected disk
exception. No nested orphan process, system setting or checkpoint deletion.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys


class DiskBudgetError(RuntimeError):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-guard',type=Path,required=True)
    p.add_argument('--disk-root',type=Path,action='append',required=True)
    p.add_argument('--min-free-gib',type=float,default=25.)
    p.add_argument('guard_arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.min_free_gib<25:p.error('Keep the25GiB engineering reserve')
    args=a.guard_arguments[1:] if a.guard_arguments[:1]==['--'] else a.guard_arguments
    if '--output' not in args:p.error('Base guard needs its immutable --output report')
    output=Path(args[args.index('--output')+1])
    if output.exists():raise FileExistsError('Preserve previous guard result')
    spec=importlib.util.spec_from_file_location('frozen_robot_memory_guard',a.base_guard)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    original=base.accounted_bytes;checks=0;lowest={};failure=None
    def monitored(*positional,**keywords):
        nonlocal checks
        checks+=1
        for root in a.disk_root:
            free=shutil.disk_usage(root).free/1024**3
            lowest[str(root)]=min(lowest.get(str(root),free),free)
            if free<a.min_free_gib:raise DiskBudgetError(f'{root}: {free:.6f}GiB free below{a.min_free_gib}GiB')
        return original(*positional,**keywords)
    base.accounted_bytes=monitored;sys.argv=[str(a.base_guard),*args]
    try:base.main()
    except DiskBudgetError as exc:
        failure=exc;output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({'status':'STOPPED_OR_REFUSED_DISK_BUDGET','stop_reason':str(exc),
            'stage':'preflight' if checks==1 else 'runtime','base_guard_finally_returned':True},indent=2)+'\n')
    finally:
        if output.exists():
            record=json.loads(output.read_text());record['disk_monitor']={'minimum_free_GiB':a.min_free_gib,
                'checks':checks,'lowest_observed_free_GiB':lowest,'system_configuration_modified':False,'files_deleted':False}
            output.write_text(json.dumps(record,indent=2)+'\n')
    if failure:raise SystemExit(1)


if __name__=='__main__':main()

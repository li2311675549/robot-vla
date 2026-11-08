#!/usr/bin/env python3
"""Four real CPU subprocess probes for the disk+memory supervisor, no model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--wrapper',type=Path,required=True)
    p.add_argument('--base-guard',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve probes')
    a.output.mkdir(parents=True);before=hashlib.sha256(a.base_guard.read_bytes()).hexdigest();results=[]
    for name in ('success','preflight_refusal','timeout','runtime_disk_stop'):
        guard=a.output/(name+'_guard.json');log=a.output/(name+'.log');marker=a.output/(name+'_child_pid.txt')
        threshold='1000000' if name=='preflight_refusal' else '25'
        child=[sys.executable,'-c',f'import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep({0.05 if name=="success" else 5})']
        command=[sys.executable,str(a.wrapper),'--base-guard',str(a.base_guard),'--disk-root',str(a.root),
            '--min-free-gib',threshold,'--','--memory-accounting','rss','--max-rss-mib','128',
            '--timeout',str(.2 if name=='timeout' else 10),'--output',str(guard),'--log',str(log),'--',*child]
        if name=='runtime_disk_stop':
            # Fault injection ONLY in this isolated probe process. No disk is
            # filled, filesystem limit changed, or production report modified.
            code="""import shutil,sys,runpy
original=shutil.disk_usage
calls=0
def probe_usage(path):
 global calls
 calls+=1
 usage=original(path)
 return usage if calls<5 else type(usage)(usage.total,usage.used,0)
shutil.disk_usage=probe_usage
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
            command=[sys.executable,'-c',code,*command[1:]]
        run=subprocess.run(command,capture_output=True,text=True,timeout=30)
        record=json.loads(guard.read_text())
        if name=='success':assert run.returncode==0 and record['status']=='PASS_SUBPROCESS_ONLY'
        elif name=='preflight_refusal':assert run.returncode!=0 and record['stage']=='preflight' and not marker.exists()
        elif name=='timeout':assert run.returncode!=0 and record['stop_reason']=='wall_time_budget'
        else:assert run.returncode!=0 and record['stage']=='runtime' and marker.exists()
        alive=False
        if marker.exists():
            try:os.kill(int(marker.read_text()),0);alive=True
            except ProcessLookupError:pass
        assert not alive,'Probe child survived its guard'
        results.append({'name':name,'pass':True,'returncode':run.returncode,'child_alive_after_guard':alive,
                        'guard':record,'stderr_tail':run.stderr[-300:]})
    assert hashlib.sha256(a.base_guard.read_bytes()).hexdigest()==before
    report={'status':'PASS_ALL4_CPU_PROBES','results':results,'base_guard_bytes_unchanged':True,
            'production_disks_filled_or_modified':False,'model_loaded':False,'training_started':False}
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'passed':len(results)}))


if __name__=='__main__':main()

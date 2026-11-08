#!/usr/bin/env python3
"""Compare local planners from exactly the same archived physical fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from _bootstrap import ROOT
from benchmark_dock_stage7 import trial,fixture as make_fixture


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',type=int,default=[7,9])
    p.add_argument('--kinds',nargs='+',choices=['empty','empty_lateral','loaded'],default=['empty_lateral','loaded'])
    p.add_argument('--strategies',nargs='+',choices=['hysteresis','preapproach','lattice'],default=['hysteresis','preapproach','lattice'])
    p.add_argument('--fixture-root',type=Path)
    p.add_argument('--timeout',type=float,default=60);a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError('Use a new paired experiment folder')
    out.mkdir(parents=True);records=[]
    for kind in a.kinds:
        source=ROOT/'outputs/stage7'/('dock_empty_lateral_25' if kind=='empty_lateral' else 'dock_pairs_v2_strict')/kind
        if a.fixture_root:source=a.fixture_root.resolve()/kind
        for index in a.cases:
            fixture=source/f'case_{index:02d}'
            if not fixture.exists():
                if index<10:raise FileNotFoundError(fixture)
                fixture=out/'independent_fixtures'/kind/f'case_{index:02d}'
                try:scene,snapshot=make_fixture(index,kind,fixture)
                except Exception as exc:
                    failure={'kind':kind,'case':index,'fixture_failure':str(exc),
                             'dock_attempted':False,'fixture':str(fixture)}
                    records.append(failure)
                    (fixture/'setup_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                    (out/'episodes.json').write_text(json.dumps(records,indent=2)+'\n')
                    print(kind,index,'SETUP_FAILED',str(exc),flush=True)
                    continue
            else:
                meta=json.loads((fixture/'fixture.json').read_text())
                with np.load(fixture/'initial_state.npz') as arrays:
                    snapshot={**meta,**{k:arrays[k].copy() for k in arrays.files}}
                snapshot['state_spec']=int(snapshot['state_spec'])
            state_hash=hashlib.sha256((fixture/'initial_state.npz').read_bytes()).hexdigest()
            for strategy in a.strategies:
                directory=out/kind/f'case_{index:02d}'/strategy;directory.mkdir(parents=True)
                previous=Path.cwd();os.chdir(directory)
                try:
                    config={} if strategy=='hysteresis' else {'local_dock_strategy':strategy}
                    r=trial(fixture/'scene.xml',snapshot,strategy,directory,kind,a.timeout,config)
                    r.update(case=index,fixture=str(fixture),fixture_sha256=state_hash)
                    records.append(r)
                    (directory/'result.json').write_text(json.dumps(r,indent=2)+'\n')
                finally:os.chdir(previous)
                (out/'episodes.json').write_text(json.dumps(records,indent=2)+'\n')
                summary={}
                for k in a.kinds:
                    for s in a.strategies:
                        rows=[x for x in records if x['kind']==k and x.get('strategy')==s]
                        if rows:summary[k+'/'+s]={'episodes':len(rows),'successes':sum(x['strict_success_after_2s'] for x in rows)}
                (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
                print(kind,index,strategy,r['strict_success_after_2s'],round(r['position_error_m'],4),r['error'],flush=True)


if __name__=='__main__':main()

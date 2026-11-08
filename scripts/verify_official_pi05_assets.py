#!/usr/bin/env python3
"""Offline full-checkpoint integrity gate; never downloads, loads or trains."""
import argparse
import json
from pathlib import Path
import sys
import time

from download_official_pi05_assets import asset_relative_path,verify_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extra-deps',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep previous integrity checks')
    if a.extra_deps:sys.path.insert(0,str(a.extra_deps.resolve()))
    started=time.monotonic();result={'status':'FAIL','verified':[],'failures':[],
                                    'network_used':False,'model_loaded':False,'training_performed':False}
    try:
        m=json.loads(a.manifest.read_text())
        objects=m['objects']
        if (m['weights_bucket']!='openpi-assets' or m['prefix']!='checkpoints/pi05_base/params/'
                or m['tokenizer_bucket']!='big_vision' or len(objects)!=20
                or len({x['name'] for x in objects})!=20
                or sum(int(x['size']) for x in objects)!=12441721931
                or m['tokenizer']['name']!='paligemma_tokenizer.model'):
            raise ValueError('A complete frozen official pi05_base manifest is required; subset prefetch is insufficient')
        rows=[(m['weights_bucket'],x) for x in objects]+[(m['tokenizer_bucket'],m['tokenizer'])]
        for bucket,item in rows:
            try:
                relative=asset_relative_path(item['name'])
                if bucket=='openpi-assets' and not item['name'].startswith(m['prefix']):raise ValueError('Wrong checkpoint prefix')
                if not item.get('generation') or not item.get('crc32c'):raise ValueError('Official generation and CRC32C required')
                checked=verify_file(a.cache/bucket/relative,item)
                result['verified'].append({'name':item['name'],'bucket':bucket,**checked})
            except (ValueError,OSError) as error:
                result['failures'].append({'name':item['name'],'error':str(error)})
        result['status']='PASS_COMPLETE_OFFICIAL_ASSETS_ONLY' if len(result['verified'])==21 and not result['failures'] else 'FAIL'
    except (ValueError,KeyError,OSError) as error:result['failures'].append({'error':str(error)})
    result['wall_seconds']=time.monotonic()-started
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'verified_objects':len(result['verified']),
                      'failures':result['failures'],'model_loaded':False,'training_performed':False},indent=2))
    if result['status']=='FAIL':raise SystemExit(1)


if __name__=='__main__':main()

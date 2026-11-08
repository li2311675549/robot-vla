#!/usr/bin/env python3
"""Bounded isolated suffix prefetch; never writes the active download cache.

The complete assembled object must pass the frozen official checksum before
being eligible for an operator-controlled merge. This is not a model loader.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote, urlencode

from download_official_pi05_assets import verify_file, validate_range_response


def fetch_span(url, start, end, total, path, deadline):
    remaining=int(deadline-time.monotonic())
    if remaining<=0:raise TimeoutError('Suffix prefetch budget exhausted')
    response=subprocess.run(['curl','--fail','--location','--silent','--show-error',
        '--connect-timeout','20','--retry','4','--retry-all-errors','--retry-delay','2',
        '--retry-max-time',str(remaining),'--max-time',str(min(180,remaining)),
        '--range',f'{start}-{end}','--dump-header','-',
        '--write-out','\nROBOT_EXAM_HTTP_STATUS=%{http_code}\n','--output',str(path),url],
        capture_output=True,text=True,check=True,timeout=remaining)
    validate_range_response(response.stdout,start,end,total,path.stat().st_size)
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--object-name',required=True);p.add_argument('--source-prefix',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--authorized-download',action='store_true')
    a=p.parse_args()
    if not a.authorized_download:p.error('Existing official-asset download authorization required')
    m=json.loads(a.manifest.read_text())
    if m['weights_bucket']!='openpi-assets' or m['prefix']!='checkpoints/pi05_base/params/':raise ValueError('Wrong official source')
    item=next(x for x in m['objects'] if x['name']==a.object_name)
    if not item.get('generation') or not item.get('crc32c'):raise ValueError('Frozen generation/checksum required')
    total=int(item['size']);prefix=a.source_prefix.stat().st_size
    if not 0<prefix<total:raise ValueError('An incomplete nonempty prefix is required')
    if shutil.disk_usage(a.output_dir.parent).free<25*1024**3+2*total:raise OSError('Disk safety reserve')
    a.output_dir.mkdir();started=time.monotonic();deadline=started+3600
    result={'status':'FAIL','scope':'ONE_OBJECT_ONLY_NOT_FULL_MODEL','object':item['name'],
            'source_prefix':str(a.source_prefix),'prefix_bytes':prefix,'model_loaded':False,'training_performed':False}
    try:
        assembled=a.output_dir/(Path(item['name']).name+'.partial')
        # Snapshot exactly the observed prefix, even if its independent owner
        # appends later. Final full-object CRC/MD5 validates the entire snapshot.
        with a.source_prefix.open('rb') as source,assembled.open('xb') as dest:
            left=prefix
            while left:
                block=source.read(min(left,4*1024**2))
                if not block:raise OSError('Source prefix changed during snapshot')
                dest.write(block);left-=len(block)
            dest.flush();os.fsync(dest.fileno())
        result['prefix_snapshot_bytes']=assembled.stat().st_size
        url='https://storage.googleapis.com/storage/v1/b/openpi-assets/o/'+quote(item['name'],safe='')+'?'+urlencode({'alt':'media','generation':item['generation']})
        spans=[(start,min(total-1,start+1024**2-1)) for start in range(prefix,total,1024**2)]
        def fetch(span):
            start,end=span;path=a.output_dir/f'span_{start}_{end}.bin'
            fetch_span(url,start,end,total,path,deadline)
            print('VERIFIED_RANGE',start,end,flush=True)
            return path
        with ThreadPoolExecutor(max_workers=4) as pool,assembled.open('ab') as dest:
            # map yields in order; bounded four curl children, no whole-object
            # buffering. Keep individual spans as recovery evidence.
            for path in pool.map(fetch,spans):
                with path.open('rb') as stream:shutil.copyfileobj(stream,dest,1024**2)
                dest.flush();os.fsync(dest.fileno())
        result['checksum']=verify_file(assembled,item)
        final=assembled.with_suffix('');assembled.rename(final)
        result.update(status='PASS_ONE_OFFICIAL_OBJECT_VERIFIED',path=str(final))
    except Exception as error:result['error']=repr(error)
    result['wall_seconds']=time.monotonic()-started
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['status'].startswith('PASS'):raise SystemExit(1)


if __name__=='__main__':main()

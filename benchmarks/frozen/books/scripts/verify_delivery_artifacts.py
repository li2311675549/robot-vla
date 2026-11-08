#!/usr/bin/env python3
"""Index real continuous videos, including their explicitly limited protocols."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from _bootstrap import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new artifact manifest')
    artifacts=[('door','outputs/stage8/door_multistart_seed822_dev_recorded/video/demo.mp4','Oracle multistart v2, development complete success'),
               ('fruit','outputs/stage7/fruit_seed412_complete_recorded/video/demo.mp4','RGB-D plus contact verification, old 3000s development protocol'),
               ('books','outputs/stage34/videos/drawer_final/video/demo.mp4','historical narrow booklet complete task, not distinct_v2'),
               ('large_book_partial','outputs/stage8/book_push_edge_seed811_partial_recorded/video/demo.mp4','initialized local Oracle fixture, real push then path failure; incomplete')]
    results=[]
    for task,relative,scope in artifacts:
        path=ROOT/relative;row={'task':task,'path':relative,'scope':scope,'exists':path.exists()}
        if path.exists():
            stream=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,duration,nb_frames','-of','json',str(path)],text=True)
            row['stream']=json.loads(stream)['streams'][0]
            digest=hashlib.sha256()
            with path.open('rb') as f:
                while block:=f.read(1024*1024):digest.update(block)
            row['sha256']=digest.hexdigest();row['bytes']=path.stat().st_size
        results.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'videos':results,'note':'Container/stream verification and hashes; physical scoring remains in source result.json'},indent=2)+'\n')
    print([(r['task'],r['exists'],r.get('stream',{}).get('nb_frames')) for r in results])


if __name__=='__main__':main()

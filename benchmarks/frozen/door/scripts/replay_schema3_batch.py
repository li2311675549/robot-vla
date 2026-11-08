#!/usr/bin/env python3
"""Replay all eligible schema3 short skills without selecting only easy cases."""
import argparse
import json
from pathlib import Path
from replay_lerobot_lowrate import replay


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new result path')
    results=[]
    for metadata in sorted(a.source.rglob('episode.json')):
        meta=json.loads(metadata.read_text());folder=metadata.parent
        quality=folder/'validation.json'
        if meta.get('schema_version')!=3 or not meta['success'] or not quality.exists():continue
        validation=json.loads(quality.read_text())
        if validation['status']!='PASS':continue
        existing=folder/'low_rate_replay.json'
        result=json.loads(existing.read_text()) if existing.exists() else replay(folder)
        result.update(path=str(folder),skill=meta.get('skill'),seed=meta['seed'])
        results.append(result);print(meta.get('skill'),meta['seed'],result.get('low_rate_physical_placement_pass'),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'episodes':len(results),
        'physical_placement_passes':sum(x.get('low_rate_physical_placement_pass',False) for x in results),
        'results':results},indent=2)+'\n')


if __name__=='__main__':main()

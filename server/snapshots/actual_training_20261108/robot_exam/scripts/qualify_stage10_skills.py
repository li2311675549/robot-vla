#!/usr/bin/env python3
"""Keep all attempts; qualify positive schema3 data with both physical replayers."""
import argparse,json
from pathlib import Path
from _bootstrap import ROOT
from replay_lerobot_lowrate import replay
from replay_vla_executor import replay_episode


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new quality summary')
    rows=[]
    for file in sorted(a.source.rglob('episode.json')):
        folder=file.parent;meta=json.loads(file.read_text());quality=json.loads((folder/'validation.json').read_text())
        row={'path':str(folder.resolve()),'seed':meta['seed'],'skill':meta['skill'],
             'physical_success':meta['success'],'full_rate_quality':quality['status'],'accepted':False}
        if meta['success'] and quality['status']=='PASS':
            low_path=folder/'low_rate_replay.json';low=json.loads(low_path.read_text()) if low_path.exists() else replay(folder)
            deployed_path=folder/'vla_executor_replay.json'
            deployed=json.loads(deployed_path.read_text()) if deployed_path.exists() else replay_episode(folder)
            row.update(low_rate=low.get('low_rate_physical_placement_pass',False),same_executor=deployed['success'])
            row['accepted']=bool(row['low_rate'] and row['same_executor'])
        rows.append(row);a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps({'attempts':len(rows),'accepted':sum(x['accepted'] for x in rows),'episodes':rows},indent=2)+'\n')
        print(row['skill'],row['seed'],row['accepted'],flush=True)


if __name__=='__main__':main()

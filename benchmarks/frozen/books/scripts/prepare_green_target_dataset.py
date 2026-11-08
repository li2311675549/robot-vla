#!/usr/bin/env python3
"""Build versioned language-only LeRobot views, reusing immutable image data.

No collection, upload, model loading or training. Re-run on the server against
the existing datasets to rebuild local symlinks; do not upload local symlinks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET


def load(path):return json.loads(path.read_text())
def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def lines(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
def jsonl(path,rows):path.write_text(''.join(json.dumps(x)+'\n' for x in rows))


def main():
    p=argparse.ArgumentParser()
    for split in ('train','validation','test'):p.add_argument('--'+split,type=Path,required=True)
    p.add_argument('--stats',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--raw-manifest',type=Path)
    p.add_argument('--expected-content-manifest',type=Path,help='Portable source SHA identities verified locally before upload')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve original data and prior preparations')
    if shutil.disk_usage(a.output.parent).free<25*1024**3:raise RuntimeError('Keep25GiB free')
    original_stats_hash=sha(a.stats);source_hashes={};relative_hashes={};new_prompts={};qa=[]
    if a.raw_manifest:
        manifest=load(a.raw_manifest)
        for entry in manifest['accepted']:
            folder=Path(entry['path']).parent;physical=load(folder/'skill_result.json')
            site=ET.parse(folder/'scene.xml').find("worldbody/site[@name='drop_target']")
            assert site is not None and site.get('type')=='cylinder'
            center=list(map(float,site.get('pos').split()));radius=float(site.get('size').split()[0])
            distance=sum((physical['final_object'][i]-center[i])**2 for i in (0,1))**.5
            assert physical['status']=='PASS' and distance<radius
            qa.append({'seed':entry['seed'],'skill':entry['skill'],'center_distance_m':distance,'radius_m':radius})
    a.output.mkdir(parents=True)
    assets=a.output/'manifest/assets/original_ur5';assets.mkdir(parents=True)
    shutil.copyfile(a.stats,assets/'norm_stats.json')
    manifest_record={'prompt_version':'green_circular_marker_v2','source_stats_sha256':original_stats_hash,
        'action_semantics':'ur5_absolute_joint_goal_v1','data_images_and_actions_reused':True,
        'statistics_source':'Original121 training episodes; also retained for controlled overfit6 comparison; validation/test excluded.',
        'raw_physical_goal_qa':qa,'training_started':False,'uploads_performed':False}
    write(a.output/'manifest/manifest.json',manifest_record)
    results={}
    for split in ('train','validation','test'):
        source=getattr(a,split).resolve();info=load(source/'meta/info.json')
        mapping={}
        for row in lines(source/'meta/tasks.jsonl'):
            match=re.fullmatch(r'Pick up the (apple|banana|orange) and place it at the target\.',row['task'])
            assert match,'Unexpected source task; refuse unsafe relabel'
            text=f'Pick up the {match[1]} and place it on the green circular target marker on the table.'
            mapping[row['task']]=text;new_prompts[match[1]]=text
        for file in [*source.glob('meta/*'),*source.glob('data/**/*.parquet')]:
            if file.is_file():
                source_hashes[str(file)]=sha(file)
                relative_hashes[split+'/'+file.relative_to(source).as_posix()]=source_hashes[str(file)]
        variants=[('full',int(info['total_episodes']))]
        if split=='train':variants.append(('overfit6',6))
        for variant,count in variants:
            target=a.output/variant/split;meta=target/'meta';meta.mkdir(parents=True)
            for file in (source/'meta').iterdir():
                if file.is_file():shutil.copyfile(file,meta/file.name)
            episodes=lines(source/'meta/episodes.jsonl')[:count]
            assert [e['episode_index'] for e in episodes]==list(range(count))
            for row in episodes:row['tasks']=[mapping[x] for x in row['tasks']]
            jsonl(meta/'episodes.jsonl',episodes)
            jsonl(meta/'tasks.jsonl',[{**row,'task':mapping[row['task']]} for row in lines(source/'meta/tasks.jsonl')])
            jsonl(meta/'episodes_stats.jsonl',[r for r in lines(source/'meta/episodes_stats.jsonl') if r['episode_index']<count])
            updated=dict(info,total_episodes=count,total_frames=sum(e['length'] for e in episodes),
                         total_chunks=(count+int(info['chunks_size'])-1)//int(info['chunks_size']),splits={'train':f'0:{count}'})
            write(meta/'info.json',updated)
            if count==int(info['total_episodes']):
                (target/'data').symlink_to(os.path.relpath(source/'data',target),target_is_directory=True)
            else:
                for i in range(count):
                    relative=info['data_path'].format(episode_chunk=i//int(info['chunks_size']),episode_index=i)
                    link=target/relative;link.parent.mkdir(parents=True,exist_ok=True)
                    link.symlink_to(os.path.relpath(source/relative,link.parent))
            key=f'{variant}_{split}'
            results[key]={'episodes':count,'frames':updated['total_frames'],'root':str(target.resolve()),'source':str(source)}
            cache=a.output/'cache/local'/f'robot_exam_green_target_{key}'
            cache.parent.mkdir(parents=True,exist_ok=True);cache.symlink_to(os.path.relpath(target,cache.parent),target_is_directory=True)
    assert all(sha(Path(path))==h for path,h in source_hashes.items())
    assert sha(a.stats)==original_stats_hash==sha(assets/'norm_stats.json')
    if a.expected_content_manifest:
        expected=load(a.expected_content_manifest)
        assert relative_hashes==expected['source_hashes_relative'],'Source dataset differs from verified local data'
        assert hashlib.sha256(json.dumps(load(a.stats),sort_keys=True).encode()).hexdigest()==expected['norm_content_sha256']
    write(a.output/'preparation.json',{'status':'PREPARED_METADATA_VIEWS_NOT_NEW_DEMONSTRATIONS','views':results,
        'new_prompts':new_prompts,'source_hashes':source_hashes,'source_hashes_relative':relative_hashes,'source_unchanged':True,
        'norm_unchanged':True,'raw_goal_checks':len(qa),'maximum_final_center_distance_from_green_marker_m':max((x['center_distance_m'] for x in qa),default=None),
        'new_physical_demonstrations':0,'training_started':False,'uploads_performed':False,
        'portability':'Rebuild with this script against server-local train/validation/test paths; do not upload symlinks or rewrite source datasets.'})
    print(json.dumps({'views':results,'raw_goal_checks':len(qa),'source_unchanged':True,'new_demonstrations':0},indent=2))


if __name__=='__main__':main()

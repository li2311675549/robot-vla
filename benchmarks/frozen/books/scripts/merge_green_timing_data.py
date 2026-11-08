#!/usr/bin/env python3
"""Append qualified timing candidates without copying the original173 images.

Only the new episodes' scalar indices are rewritten. Their encoded RGB bytes,
states and actions stay identical. A NEW train-only norm asset is produced.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def load(path):return json.loads(path.read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x]
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def write_lines(path,rows):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-prepared',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep all source datasets and oldnorm unchanged')
    if shutil.disk_usage(a.output.parent).free<26*1024**3:raise RuntimeError('Preserve25GiB reserve plus append headroom')
    pa.set_cpu_count(1);pa.set_io_thread_count(1);a.output.mkdir(parents=True)
    base=load(a.base_prepared/'preparation.json');summary={};sources={};training={'state':[],'actions':[]}
    for split in ('train','validation','test'):
        original=Path(base['views'][f'full_{split}']['root']);extra=a.candidates/split
        oldinfo=load(original/'meta/info.json');newinfo=load(extra/'meta/info.json')
        assert oldinfo['features']==newinfo['features'] and oldinfo['fps']==newinfo['fps']==5
        tasks=lines(original/'meta/tasks.jsonl');ids={r['task']:r['task_index'] for r in tasks}
        oldtasks={r['task_index']:r['task'] for r in tasks}
        newtasks={r['task_index']:r['task'] for r in lines(extra/'meta/tasks.jsonl')}
        assert set(newtasks.values())==set(ids)
        mapping={index:ids[text] for index,text in newtasks.items()}
        target=a.output/'full'/split;meta=target/'meta';meta.mkdir(parents=True)
        oldepisodes=lines(original/'meta/episodes.jsonl');newepisodes=lines(extra/'meta/episodes.jsonl')
        oldstats=lines(original/'meta/episodes_stats.jsonl');newstats=lines(extra/'meta/episodes_stats.jsonl')
        offset_episodes=len(oldepisodes);offset_frames=int(oldinfo['total_frames']);episodes=deepcopy(oldepisodes);stats=deepcopy(oldstats)
        for e in oldepisodes:
            idx=e['episode_index'];relative=oldinfo['data_path'].format(episode_chunk=idx//oldinfo['chunks_size'],episode_index=idx)
            source=original/relative;link=target/relative;link.parent.mkdir(parents=True,exist_ok=True)
            sources[str(source.resolve())]=digest(source)
            link.symlink_to(os.path.relpath(source.resolve(),link.parent))
        for source_episode,source_stats in zip(newepisodes,newstats,strict=True):
            i=source_episode['episode_index'];new_id=offset_episodes+i
            relative=newinfo['data_path'].format(episode_chunk=i//newinfo['chunks_size'],episode_index=i)
            source=extra/relative;sources[str(source.resolve())]=digest(source)
            table=pq.ParquetFile(source).read(use_threads=False)
            for column,values in [('episode_index',[new_id]*len(table)),
                                  ('index',[x+offset_frames for x in table['index'].to_pylist()]),
                                  ('task_index',[mapping[x] for x in table['task_index'].to_pylist()])]:
                field=table.schema.field(column);table=table.set_column(table.schema.get_field_index(column),field,pa.array(values,type=field.type))
            destination=target/oldinfo['data_path'].format(episode_chunk=new_id//oldinfo['chunks_size'],episode_index=new_id)
            destination.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,destination)
            check=pq.ParquetFile(destination).read(use_threads=False)
            assert table.equals(check), 'Encoded RGB/state/action data changed during append'
            episode=deepcopy(source_episode);episode['episode_index']=new_id;episodes.append(episode)
            st=deepcopy(source_stats);st['episode_index']=new_id
            for key in ('min','max','mean'):
                st['stats']['episode_index'][key]=[new_id]
                st['stats']['index'][key]=[v+offset_frames for v in st['stats']['index'][key]]
                st['stats']['task_index'][key]=[mapping[int(v)] for v in st['stats']['task_index'][key]]
            stats.append(st)
        info=deepcopy(oldinfo);info.update(total_episodes=len(episodes),total_frames=offset_frames+newinfo['total_frames'],
            total_chunks=(len(episodes)+info['chunks_size']-1)//info['chunks_size'],splits={'train':f'0:{len(episodes)}'})
        write(meta/'info.json',info);write_lines(meta/'tasks.jsonl',tasks);write_lines(meta/'episodes.jsonl',episodes);write_lines(meta/'episodes_stats.jsonl',stats)
        observed=0
        for episode in episodes:
            i=episode['episode_index'];file=target/info['data_path'].format(episode_chunk=i//info['chunks_size'],episode_index=i)
            t=pq.ParquetFile(file).read(columns=['observation.state','action','episode_index','index','task_index'],use_threads=False)
            assert all(v==i for v in t['episode_index'].to_pylist())
            assert t['index'].to_pylist()==list(range(observed,observed+len(t)))
            assert all(oldtasks[k] in episode['tasks'] for k in t['task_index'].to_pylist())
            observed+=len(t)
            if split=='train':
                for key,column in [('state','observation.state'),('actions','action')]:training[key].append(np.asarray(t[column].to_pylist(),dtype=float))
        assert observed==info['total_frames']
        summary[split]={'episodes':info['total_episodes'],'frames':observed,'original_episode_count':offset_episodes,
                        'appended_episode_count':len(newepisodes),'root':str(target.resolve())}
        cache=a.output/'cache/local'/f'robot_exam_green_timing_full_{split}'
        cache.parent.mkdir(parents=True,exist_ok=True);cache.symlink_to(os.path.relpath(target,cache.parent),target_is_directory=True)
    norms={}
    for key,arrays in training.items():
        values=np.concatenate(arrays);assert np.isfinite(values).all()
        norms[key]={'mean':values.mean(0).tolist(),'std':values.std(0).tolist(),
                    'q01':np.quantile(values,.01,axis=0).tolist(),'q99':np.quantile(values,.99,axis=0).tolist()}
    assets=a.output/'manifest/assets/original_ur5';assets.mkdir(parents=True)
    write(assets/'norm_stats.json',{'norm_stats':norms})
    write(a.output/'manifest/manifest.json',{'scope':'Converted-only merged release; original physical labels retained, strict audit separate',
        'split_counts':{k:v['episodes'] for k,v in summary.items()},'normalization_split':'train',
        'frames':sum(v['frames'] for v in summary.values()),'action_semantics':'ur5_absolute_joint_goal_v1',
        'prompt_version':'green_circular_marker_v2','training_started':False})
    assert all(digest(Path(path))==sha for path,sha in sources.items())
    write(a.output/'preparation.json',{'status':'MERGED_DATA_PREPARED_NOT_TRAINED_PENDING_STRICT_AUDIT','views':summary,
        'source_parquet_sha256':sources,'old_images_not_copied':True,'new_episode_rgb_bytes_preserved':True,
        'norm_source':'Merged training split only, exported float32 values; a NEW asset, never substitute for checkpoint999 oldnorm.',
        'norm_sha256':digest(assets/'norm_stats.json'),'source_unchanged':True,'training_started':False,'uploads_performed':False})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

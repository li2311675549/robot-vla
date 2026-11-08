#!/usr/bin/env python3
"""Low-memory scalar/metadata verification; no Torch/JAX, images or weights."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def load(path):return json.loads(path.read_text())
def lines(path):return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior verification')
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    prepared=load(a.prepared/'preparation.json');saved=load(a.prepared/'manifest/assets/original_ur5/norm_stats.json')['norm_stats']
    merged='source_parquet_sha256' in prepared
    results={};train={'state':[],'actions':[]}
    for name,description in prepared['views'].items():
        root=Path(description['root']);info=load(root/'meta/info.json')
        tasks={r['task_index']:r['task'] for r in lines(root/'meta/tasks.jsonl')}
        assert len(tasks)==3 and all(re.fullmatch(r'Pick up the (apple|banana|orange) and place it on the green circular target marker on the table\.',text) for text in tasks.values())
        episodes=lines(root/'meta/episodes.jsonl');rows=0;ids=[]
        assert len(episodes)==description['episodes']==info['total_episodes']
        assert set(('observation.images.front','observation.images.wrist'))<=info['features'].keys()
        assert info['fps']==5 and info['features']['action']['shape']==[7]
        for episode in episodes:
            index=episode['episode_index'];relative=info['data_path'].format(episode_chunk=index//info['chunks_size'],episode_index=index)
            file=root/relative
            if merged:
                assert file.is_symlink()==(index<description['original_episode_count'])
            else:assert file.resolve()==(Path(description['source'])/relative).resolve()
            table=pq.ParquetFile(file).read(columns=['observation.state','action','task_index','frame_index','episode_index','episode_success'],use_threads=False)
            assert len(table)==episode['length'] and table['frame_index'].to_pylist()==list(range(len(table)))
            assert all(v==index for v in table['episode_index'].to_pylist()) and all(table['episode_success'].to_pylist())
            assert all(tasks[v] in episode['tasks'] for v in table['task_index'].to_pylist())
            for key,column in [('state','observation.state'),('actions','action')]:
                values=np.asarray(table[column].to_pylist(),dtype=float)
                assert values.shape==(len(table),7) and np.isfinite(values).all()
                if key=='actions':assert np.all((values[:,6]>=0)&(values[:,6]<=1))
                if name==('train' if merged else 'full_train'):train[key].append(values)
            rows+=len(table);ids.append(index)
        assert ids==list(range(len(episodes))) and rows==description['frames']==info['total_frames']
        results[name]={'episodes':len(episodes),'frames':rows,'prompt_frame_mismatches':0,'original_episode_data_reused':True,
                       'new_episode_indices_rewritten':bool(merged)}
    errors={}
    for key,arrays in train.items():
        values=np.concatenate(arrays)
        computed={'mean':values.mean(0),'std':values.std(0),'q01':np.quantile(values,.01,axis=0),'q99':np.quantile(values,.99,axis=0)}
        errors[key]={k:float(np.max(np.abs(v-saved[key][k]))) for k,v in computed.items()}
        assert max(errors[key].values())<1e-7
    report={'status':'PASS_SCALAR_METADATA_AND_TRAIN_ONLY_NORM','views':results,'stats_export_float32_differences':errors,
            'images_decoded':False,'torch_imported':False,'jax_imported':False,'model_loaded':False,'training_started':False,
            'scope':'Independent low-memory server-ready check; native OpenPI model/GPU checks are separate.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

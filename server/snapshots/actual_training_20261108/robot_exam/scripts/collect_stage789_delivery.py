#!/usr/bin/env python3
"""Build delivery evidence directly from persisted runs, without inventing gaps."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from _bootstrap import ROOT


def read(relative):
    path=ROOT/relative
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new delivery snapshot')
    manifest=read('outputs/stage9/dataset_manifest_v1/manifest.json')
    formal=read('outputs/stage8/door_formal_multistart_v2/summary.json')
    source=read('outputs/stage8/door_formal_multistart_v2/freeze.json')
    formal_records=read('outputs/stage8/door_formal_multistart_v2/episodes.json') or []
    formal_columns=('seed','success','sim_seconds','wall_seconds','root_failure',
                    'failure_stage','error','num_retries','close_regrasp_count',
                    'post_release_recoveries','collision_events','warnings',
                    'solver_linesearch_messages','phase_metrics','evidence')
    pytest_path='outputs/stage9/pytest_stage789_final.log' if (ROOT/'outputs/stage9/pytest_stage789_final.log').exists() else 'outputs/stage9/pytest_stage789_freeze.log'
    pytest=(ROOT/pytest_path).read_text()
    match=re.search(r'(\d+) passed',pytest)
    key_runs={}
    for name in ['door_seed501_current','door_multistart_seed821_dev_recorded',
                 'door_multistart_seed821_dock45_dev','door_multistart_seed822_dev_recorded',
                 'booklet_seed812_topgrasp_regression','books_distinct_seed811_dev',
                 'books_distinct_seed811_topgrasp_dev']:
        path=f'outputs/stage8/{name}/result.json';value=read(path)
        key_runs[name]=None if value is None else {'evidence':path,**{k:value.get(k) for k in
            ('success','sim_seconds','wall_seconds','failure_category','error','score','video','warnings')}}
    result={'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'stage7':{'status':'PARTIAL','evidence':'docs/results/stage7_reliability_acceptance.json',
                  'actual_results':read('docs/results/stage7_reliability_acceptance.json'),
                  'new_formal_fruit_evaluation':'NOT_RUN','loaded_dock_strict':'6/10'},
        'stage8':{'status':'PARTIAL','key_runs':key_runs,
            'new_formal_door':formal,'frozen_git_commit':source['git_commit'] if source else None,
            'new_formal_door_records':[{k:row.get(k) for k in formal_columns} for row in formal_records],
            'door_recovery_counters':{
                'fsm_retries':sum(x.get('num_retries',0) for x in formal_records),
                'close_handle_regrasps':sum(x.get('close_regrasp_count',0) for x in formal_records),
                'post_release_recoveries':sum(x.get('post_release_recoveries',0) for x in formal_records)},
            'frozen_source_evidence':'outputs/stage8/door_formal_multistart_v2/freeze.json',
            'new_formal_door_complete':bool(formal and formal['summary']['episodes']==10),
            'new_distinct_book_formal':'NOT_RUN','historical_booklet_formal':'10/10, limited geometry only',
            'large_book_local_push':read('outputs/stage8/book_push_edge_seed811_partial_recorded/result.json')},
        'stage9':{'status':'PREPARATION_VERIFIED_NO_MODEL_TRAINING',
            'schema3_episodes':manifest['successful_episodes'],'schema3_frames':manifest['frames'],
            'split_counts':manifest['split_counts'],'quality_excluded':manifest['excluded'],
            'legacy_semantics_separate':True,
            'lerobot':{name:read(f'outputs/stage9/lerobot_{name}_v1/conversion_result.json')
                       for name in ('train','validation','test')},
            'native_openpi_loader':read('outputs/stage9/openpi_native_training_loader_retry.json'),
            'mock_network_physics_chain':read('outputs/stage9/openpi_mock_physics_chain.json'),
            'same_vla_executor_replay':read('outputs/stage9/vla_executor_replay_summary.json'),
            'real_client_cli_with_mock_policy':read('outputs/stage9/real_client_cli_mock_result.json'),
            'environment_audit':'docs/results/stage9_openpi_environment.json',
            'training_performed':False,'model_weights_downloaded':False,'external_uploads':False},
        'video_artifacts':'docs/results/stage789_video_artifacts.json',
        'pytest':{'passed':int(match.group(1)) if match else None,'log':pytest_path}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(a.output)


if __name__=='__main__':main()

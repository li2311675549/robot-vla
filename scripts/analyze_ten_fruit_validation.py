#!/usr/bin/env python3
"""Audit every declared ten-fruit trial; never replace or rescore a failure."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from _bootstrap import ROOT


def average(rows, key, successful_only=False):
    values=[r['result'][key] for r in rows if not successful_only or r['result']['success']]
    return statistics.mean(values) if values else None


def group_stats(rows):
    success=sum(bool(r['result']['success']) for r in rows)
    return {'attempts':len(rows),'successes':success,'failures':len(rows)-success,
        'observed_success_rate':success/len(rows) if rows else None,
        'mean_success_sim_seconds':average(rows,'sim_seconds',True),
        'mean_all_attempt_sim_seconds':average(rows,'sim_seconds'),
        'mean_all_attempt_wall_seconds':average(rows,'wall_seconds'),
        'failure_categories':dict(Counter(r['result'].get('failure_category') for r in rows
                                         if not r['result']['success']))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous experiment audits')
    summary=json.loads((a.root/'summary.json').read_text());protocol=json.loads(a.protocol.read_text())
    assert summary['status']=='COMPLETE' and summary['attempted']==summary['planned']==30
    assert summary['protocol_sha256']==hashlib.sha256(a.protocol.read_bytes()).hexdigest()
    expected=protocol['matrices']['validation/grasp'];rows=summary['results'];videos=[];warning_total=0
    assert len(rows)==len(expected)==30
    assert Counter(r['fixture']['category'] for r in rows)=={'apple':10,'banana':10,'orange':10}
    diagnostics=[]
    for i,(entry,spec) in enumerate(zip(rows,expected)):
        assert entry['fixture']==spec and spec['scene_object_count']==10 and spec['split']=='validation'
        r=entry['result'];assert r['scene_object_count']==10 and r['phase']=='grasp' and r['model_inference']
        assert r['initial_state_restores']==1 and r['runtime_state_restores']==0
        meta=r['server_metadata'];assert str(meta['checkpoint_step_label'])=='9999'
        assert meta['norm_stats_sha256']==protocol['normalization_sha256']
        assert meta['episode_policy_noise_seed']==spec['policy_noise_seed']
        assert meta['gripper_adapter']==protocol['adapter'] and meta['actual_cameras']==['front','wrist']
        assert r['video']['full_decode']=='PASS'
        folder=a.root/f"{i:03d}_{spec['seed']}_{spec['category']}";path=folder/'video/demo.mp4'
        assert path.is_file();warning_total+=sum(r['warnings'])
        videos.append({'index':i,'seed':spec['seed'],'category':spec['category'],
            'scene_fruit_count':10,'policy_noise_seed':spec['policy_noise_seed'],
            'success':r['success'],'failure_category':r.get('failure_category'),
            'video':str(path.resolve()),'video_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'result':str((folder/'result.json').resolve()),
            'result_sha256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),
            'sim_seconds':r['sim_seconds'],'wall_seconds':r['wall_seconds']})
        if not r['success']:
            fixture=Path(r['source_fixture']);target=json.loads((fixture/'episode.json').read_text())['target_body']
            trajectory=json.loads((folder/'trajectory.json').read_text())
            lift=r.get('max_lift_by_body_m',{}).get(target)
            diagnosis=r.get('failure_category')
            if diagnosis=='phase_timeout_or_target_not_verified':
                if trajectory and trajectory[-1]['target_position'][2]<trajectory[0]['target_position'][2]-.2:
                    diagnosis='target_fell_from_table'
                elif lift is not None and lift<.01:diagnosis='no_effective_target_lift'
                elif lift is not None and lift<.055:diagnosis='partial_lift_below_original_threshold'
                else:diagnosis='lift_or_contact_not_stably_verified'
            diagnostics.append({'index':i,'seed':spec['seed'],'category':spec['category'],
                'noise_seed':spec['policy_noise_seed'],'official_failure_category':r.get('failure_category'),
                'derived_diagnostic_only_not_rescoring':diagnosis,'maximum_target_lift_m':lift,
                'final_score':r.get('score'),'error':r.get('error')})
    report={'status':'PASS_COMPLETE_30_TRIAL_LEDGER_AND_VIDEO_AUDIT',
        'scope':protocol['scope'],'protocol_sha256':summary['protocol_sha256'],
        'checkpoint_directory_label':9999,'optimizer_updates':10000,'new_training':False,
        'independent_initial_layouts':6,'fresh_policy_rollouts':30,'formal_test_used':False,
        'all_trials':group_stats(rows),
        'by_class':{c:group_stats([r for r in rows if r['fixture']['category']==c])
                    for c in ('apple','banana','orange')},
        'primary18_all_six_layouts_rng0':group_stats([r for r in rows if r['fixture']['policy_noise_seed']==0]),
        'supplementary12_first_four_layouts_rng1':group_stats([r for r in rows if r['fixture']['policy_noise_seed']==1]),
        'by_layout':{str(seed):group_stats([r for r in rows if r['fixture']['seed']==seed])
                     for seed in sorted({r['fixture']['seed'] for r in rows})},
        'all30_initial_scene_fruit_counts_verified':True,'all30_video_full_decode_pass':True,
        'all30_model_norm_rng_and_camera_contracts_pass':True,'runtime_state_restores':0,
        'data_warning_total':warning_total,'matrix_wall_seconds':summary['wall_seconds'],
        'failure_diagnostics':diagnostics,'videos':videos,
        'diagnostic_note':'Derived10mm/200mm bins describe failure traces only; original55mm/contact/parking/scene score unchanged.',
        'comparison_note':'Earlier3/9 and known-context40-video showcase remain separate. New inference GPU is Blackwell, earlier Ada; no controlled causal claim or new training.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['all_trials','by_class','primary18_all_six_layouts_rng0',
                                           'supplementary12_first_four_layouts_rng1']},indent=2))


if __name__=='__main__':main()

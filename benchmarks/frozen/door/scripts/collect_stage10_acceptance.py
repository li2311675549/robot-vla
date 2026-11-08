#!/usr/bin/env python3
"""Collect persisted Stage10 evidence without turning local skills into full tasks."""
import argparse,hashlib,json,re,shutil,subprocess
from pathlib import Path
from _bootstrap import ROOT


def read(relative):
    path=ROOT/relative
    return json.loads(path.read_text()) if path.exists() else None


def compact(name):
    relative=f'outputs/stage10/{name}/result.json';data=read(relative)
    issue=read(f'outputs/stage10/{name}/recording_cadence_issue.json')
    if data is None:return {'evidence':relative,'status':'NOT_RUN_OR_INCOMPLETE','recording_issue':issue}
    keys=('success','scope','sim_seconds','wall_seconds','completed','error','failure','failure_category',
          'score','operations','lift_m','collision_events','warnings','num_retries','video',
          'terminal_bilateral','initial_bilateral','release_clearance_m','cabinet_x',
          'maximum_lift_m','continuous_bilateral_lift_hold_seconds','placement_error_m')
    result={'evidence':relative,'recording_issue':issue,**{k:data[k] for k in keys if k in data}}
    log=ROOT/'outputs/stage10'/name/'MUJOCO_LOG.TXT'
    result['isolated_solver_linesearch_messages']=log.read_text().count('Linesearch objective is not convex') if log.exists() else 0
    result['solver_scope']='per-process if isolated; no conclusion about shared project log'
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new acceptance snapshot')
    integrity=read('outputs/stage10/baseline_integrity.json')
    unchanged={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha for path,sha in integrity.items()}
    matrices={}
    for name in ('dock_screen_v1','dock_matrix_v1','dock_payload_envelope_v2','dock_independent_v1',
                 'dock_independent_braked_v2','dock_axis_independent_review_v5','dock_tracking_v4_screen','dock_lattice_independent_v6'):
        records=read(f'outputs/stage10/{name}/episodes.json') or []
        columns=('kind','strategy','case','seed','strict_success_after_2s','position_error_m',
                 'heading_error_rad','linear_speed_m_s','yaw_rate_rad_s','elapsed_sim_seconds',
                 'bilateral_contact','collision_events','warnings','error','fixture_failure','fixture_sha256')
        matrices[name]={'summary':read(f'outputs/stage10/{name}/summary.json'),
            'records':[{k:r[k] for k in columns if k in r} for r in records],
            'evidence':f'outputs/stage10/{name}/episodes.json'}
    qa=read('outputs/stage10/short_skill_quality_v1.json');raw=read('outputs/stage10/short_skill_collection_v1/summary.json')
    replayed=[r for r in qa['episodes'] if 'same_executor' in r]
    manifest=read('outputs/stage10/fruit_dataset_manifest_v1/manifest.json')
    pytest_log='outputs/stage10/pytest_release_clockfix_final.log'
    pytest=(ROOT/pytest_log).read_text();match=re.search(r'(\d+) passed',pytest)
    vision=read('outputs/stage10/door_handle_vision_v1/result.json')
    result={'stage':10,'status':'PARTIAL','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        'historical_evidence_unchanged':unchanged,
        'historical_video_integrity':read('outputs/stage10/historical_video_integrity.json'),
        'historical_door_formal':read('outputs/stage8/door_formal_multistart_v2/summary.json'),
        'gate1':{'status':'PARTIAL_NOT_INDEPENDENTLY_ROBUST','matrices':matrices,
            'banana_checkpoints':{name:compact(name) for name in ('seed413_staging_local_plan_v1','seed413_staging_braked_v2','seed413_staging_anchor_v3','seed413_axis_fast_yaw_v5','seed413_lattice_checkpoint_v6')},
            'plots':['outputs/stage10/plots/loaded_case07_pair.png','outputs/stage10/plots/loaded_case09_pair.png']},
        'gate2':{'status':'BASELINE_RETAINED_NEW_FORMAL_NOT_RUN','baseline_412':compact('fruit_baseline_seed412_regression'),
                 'new_fruit_formal':'NOT_RUN','internal_1600_seconds_pass':False},
        'gate3':{'status':'LOCAL_GRASPS_AND_TWO_BOOKS_VERIFIED_NOT_COMPLETE',
            'all_persisted_book_attempts':{p.name:compact(p.name) for p in sorted((ROOT/'outputs/stage10').glob('book_*')) if (p/'result.json').exists()},
            'unexecuted_probe':read('outputs/stage10/book_top_drawer_height_v5/setup_failure.json'),
            'local_grasps':{n:compact(n) for n in ('book_max_rrt_branches_v2','book_max_sidegrasp_recorded','book_medium_center_bay_v4','book_small_shifted_bay_v5')},
            'full_attempts_by_version':{n:compact(n) for n in ('book_full_oracle_seed1011_v1','book_full_oracle_seed1011_v2')},
            'placement_checkpoints':{n:compact(n) for n in ('book_low_drawer_backbase_v2','book_low_drawer_top_release_v3','book_low_drawer_jointpath_v4',
                'book_third_corridor_checkpoint_v6','book_third_world_carry_checkpoint_v7','book_third_slip_checkpoint_recorded',
                'book_third_slip_checkpoint_recorded_clockfix')},
            'new_book_formal':'NOT_RUN'},
        'gate4':{'status':'DEVELOPMENT_AND_LOCAL_VISION_ONLY',
            'development_pairs':{str(seed):{kind:compact(f'door_baseline_seed{seed}' if kind=='baseline' else f'door_contact_seed{seed}_v1')
                for kind in ('baseline','contact')} for seed in (1081,1082,1083)},
            'handle_localization':vision,'vision_local_grasp':read('outputs/stage10/door_vision_local_grasp_v2/result.json'),
            'vision_handle_geometry_audit':read('outputs/stage10/door_vision_grasp_contact_audit_v3/result.json'),
            'local_release_fault_pairs':read('outputs/stage10/door_contact_fault_pairs_v1/result.json'),
            'full_visual_door_policy':'NOT_IMPLEMENTED','new_formal':'NOT_RUN'},
        'gate5':{'status':'LOCAL_PREPARATION_VERIFIED_NO_TRAINING','new_attempts':len(raw),
            'new_physical_successes':sum(r['physical_result']['status']=='PASS' for r in raw),
            'full_rate_qualified':len(replayed),'low_rate_successes':sum(r['low_rate'] for r in replayed),
            'deployment_executor_successes':sum(r['same_executor'] for r in replayed),
            'new_accepted':qa['accepted'],'fruit_episodes':manifest['successful_episodes'],'fruit_frames':manifest['frames'],
            'split_counts':manifest['split_counts'],'manifest':'outputs/stage10/fruit_dataset_manifest_v1/manifest.json',
            'lerobot':{name:read(f'outputs/stage10/lerobot_fruit_{name}_v1/conversion_result.json') for name in ('train','validation','test')},
            'native_openpi':read('outputs/stage10/openpi_native_fruit_loader_v1.json'),
            'configuration_check':read('outputs/stage10/lora_pilot_config_check.json'),
            'native_client_mock':read('outputs/stage10/real_client_mock_protocol_refactor.json'),
            'unified_score_expert':compact('unified_short_expert_seed201'),'unified_score_mock':compact('unified_short_mock_hold_seed201'),
            'weights_downloaded':False,'training_performed':False,'external_uploads':False,'paid_resources':False},
        'pytest':{'passed':int(match.group(1)) if match else None,'log':pytest_log},
        'free_disk_gib':shutil.disk_usage(ROOT).free/2**30}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(a.output)


if __name__=='__main__':main()

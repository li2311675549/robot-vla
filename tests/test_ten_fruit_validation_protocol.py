"""Protect ten-fruit coverage and validation/test isolation, not model accuracy."""
import json
from pathlib import Path
from collections import Counter


ROOT=Path(__file__).resolve().parents[1]


def test_ten_fruit_matrix_covers_all_layouts_without_filtering_expert_failures():
    original=json.loads((ROOT/'configs/stage22_parked_policy_protocol_v1.json').read_text())
    protocol=json.loads((ROOT/'configs/stage24_ten_fruit_grasp_validation30_v1.json').read_text())
    rows=protocol['matrices']['validation/grasp']
    candidates=[r for r in original['candidate_registry'] if
                r['split']=='validation' and r['phase']=='grasp' and r['scene_object_count']==10]
    assert len(rows)==30 and Counter(r['category'] for r in rows)=={'apple':10,'banana':10,'orange':10}
    assert all(r['split']=='validation' and r['phase']=='grasp' and r['scene_object_count']==10 for r in rows)
    assert {r['fixture'] for r in rows}=={r['fixture'] for r in candidates}
    by_fixture={r['fixture']:r for r in candidates}
    assert all(r['sha256']==by_fixture[r['fixture']]['sha256'] for r in rows)
    assert {r['fixture'] for r in rows if r['expert_parent_status']=='FAIL'}=={
        r['fixture'] for r in candidates if r['expert_parent_status']=='FAIL'}


def test_noise_repetitions_are_declared_without_changing_physical_protocol():
    original=json.loads((ROOT/'configs/stage22_parked_policy_protocol_v1.json').read_text())
    protocol=json.loads((ROOT/'configs/stage24_ten_fruit_grasp_validation30_v1.json').read_text())
    rows=protocol['matrices']['validation/grasp'];seeds=sorted({r['seed'] for r in rows})
    assert len(seeds)==protocol['independent_layouts']==6
    for category in ('apple','banana','orange'):
        group=[r for r in rows if r['category']==category]
        assert {r['seed'] for r in group if r['policy_noise_seed']==0}==set(seeds)
        assert {r['seed'] for r in group if r['policy_noise_seed']==1}==set(seeds[:4])
    for key in ('phase_budget_seconds','normalization_sha256','policy_hz',
                'execute_prefix_actions','action_horizon','network_image_size','max_response_seconds'):
        assert protocol[key]==original[key]
    assert protocol['formal_test_partition_used'] is False


def test_supplemental_video_quota_never_substitutes_three_or_five_fruit_scenes():
    fixed=json.loads((ROOT/'configs/stage24_ten_fruit_grasp_validation30_v1.json').read_text())
    supplement=json.loads((ROOT/'configs/stage24_tenfruit_video_collection40_v1.json').read_text())
    assert supplement['required_scene_fruit_count']==10 and len(supplement['baseline'])==30
    lookup={(r['seed'],r['category'],r['policy_noise_seed']):r for r in fixed['matrices']['validation/grasp']}
    for row in supplement['baseline']+supplement['plan']:
        original=lookup[(row['scene_seed'],row['category'],row['policy_noise_seed'])]
        assert row['scene_fruit_count']==10 and row['fixture']==original['fixture']
        assert row['fixture_sha256']==original['sha256']
    assert supplement['original_validation_result'].startswith('14/30')
    assert supplement['normalization_sha256']==fixed['normalization_sha256']


def test_user_ten_success_upgrade_stays_ten_fruit_and_preserves_previous_trials():
    fixed=json.loads((ROOT/'configs/stage24_ten_fruit_grasp_validation30_v1.json').read_text())
    p=json.loads((ROOT/'configs/stage24_tenfruit_video_collection40_success10_v1.json').read_text())
    assert p['goal']['minimum_success_each_class']==10 and p['goal']['total_display_videos']==40
    assert len(p['baseline'])==len({r['result'] for r in p['baseline']})==54
    lookup={(r['seed'],r['category']):r for r in fixed['matrices']['validation/grasp']}
    for row in p['baseline']+p['plan']:
        original=lookup[(row['scene_seed'],row['category'])]
        assert row['scene_fruit_count']==10 and row['fixture_sha256']==original['sha256']
    assert p['original_validation_result'].startswith('14/30')

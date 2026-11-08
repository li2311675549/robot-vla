import copy
import json
from collections import Counter
from pathlib import Path
import pytest
from robot_exam.data.multifruit_plan import collection_registry, check_admission


def config():
    return json.loads((Path(__file__).parents[1]/'configs/stage20_multifruit_dataset_expansion_proposal_v1.json').read_text())


def test_complex_scenes_dominate_and_whole_scene_is_grouped():
    registry = collection_registry(config())
    scenes = registry['collection']
    assert len(scenes) == 80
    assert Counter(r['object_count'] for r in scenes) == {10:48, 5:20, 3:12}
    assert Counter(r['split'] for r in scenes) == {'train':56, 'validation':12, 'test':12}
    assert len({p['parent_id'] for r in scenes for p in r['parents']}) == 240
    assert registry['accepted_parent_count'] == 0
    assert registry['sampling_weights'] == {'compatible_old':.2, 'qualified_new':.8}
    assert Counter(r['object_count'] for r in scenes[:6]) == {10:2, 5:2, 3:2}


def valid_record(registry):
    row = registry['collection'][0]
    phase = dict(action_semantics='ur5_absolute_joint_goal_v1', physical_success=True,
                 parking_pass=True, full_rate_replay_pass=True, low_rate_replay_pass=True,
                 dual_camera_sync_pass=True, lerobot_openpi_load_pass=True)
    return dict(parent_id=row['parents'][0]['parent_id'], split=row['split'],
                physical_parent_success=True, scene_object_count=row['object_count'],
                expert_source='oracle', language_qa_pass=True,
                phases={'grasp':copy.deepcopy(phase), 'place':copy.deepcopy(phase)})


def test_incomplete_replay_or_grasp_only_cannot_be_admitted():
    registry = collection_registry(config())
    record = valid_record(registry)
    assert check_admission([record], registry)['accepted_parents'] == 1
    record['phases']['place']['low_rate_replay_pass'] = False
    with pytest.raises(ValueError, match='qualification'):
        check_admission([record], registry)
    record['phases'].pop('place')
    with pytest.raises(ValueError, match='Both phases'):
        check_admission([record], registry)


def test_scene_split_leakage_and_duplicate_success_rejected():
    registry = collection_registry(config())
    record = valid_record(registry)
    with pytest.raises(ValueError, match='Duplicate'):
        check_admission([record, record], registry)
    record['split'] = 'test'
    with pytest.raises(ValueError, match='cross-split'):
        check_admission([record], registry)
    cfg = config()
    cfg['seed_ranges_proposed_not_started']['validation'] = [2100001, 2100012]
    with pytest.raises(ValueError, match='Seed leakage'):
        collection_registry(cfg)


def test_physical_failure_cannot_become_positive_by_metadata_only():
    registry = collection_registry(config())
    record = valid_record(registry)
    record['physical_parent_success'] = False
    with pytest.raises(ValueError, match='Physical failure'):
        check_admission([record], registry)

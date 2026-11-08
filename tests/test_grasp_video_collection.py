import copy
import pytest

from robot_exam.evaluation.video_collection import select_grasp_videos


def pool():
    rows=[]
    for cat in ('apple','banana','orange'):
        for i in range(12):
            rows.append(dict(id=f'{cat}{i}',category=cat,scene_seed=1,
                policy_noise_seed=i,success=i<8,exhibit_eligible=True,failure_category='timeout'))
    for i in range(4):
        rows.append(dict(id=f'extra{i}',category='apple',scene_seed=2,
            policy_noise_seed=i,success=False,exhibit_eligible=True,failure_category='slip'))
    return rows


def test_display_preserves_ledger_and_each_class_quota():
    rows=pool();before=copy.deepcopy(rows);ids=select_grasp_videos(rows)
    assert rows==before and len(ids)==len(set(ids))==40
    for cat in ('apple','banana','orange'):
        assert sum(r['success'] for r in rows if r['id'] in ids and r['category']==cat)>=8
    assert sum(not r['success'] for r in rows if r['id'] in ids)>=4


def test_many_successes_in_one_class_cannot_hide_missing_banana_or_missing_videos():
    rows=pool();next(r for r in rows if r['id']=='banana0')['success']=False
    assert select_grasp_videos(rows) is None
    rows=pool();rows[0]['exhibit_eligible']=False
    assert select_grasp_videos(rows) is None


def test_duplicate_physical_attempt_or_impossible_quota_rejected():
    rows=pool()
    with pytest.raises(ValueError):select_grasp_videos(rows+[rows[0]])
    with pytest.raises(ValueError):select_grasp_videos(rows,total=20)


def test_ten_success_upgrade_cannot_reuse_an_eight_success_collection():
    assert select_grasp_videos(pool(),successes_per_category=10) is None
    rows=[]
    for cat in ('apple','banana','orange'):
        for i in range(14):
            rows.append(dict(id=f'{cat}{i}',category=cat,scene_seed=1,
                policy_noise_seed=i,success=i<10,exhibit_eligible=True,failure_category='timeout'))
    selected=select_grasp_videos(rows,successes_per_category=10)
    assert len(selected)==40
    for cat in ('apple','banana','orange'):
        assert sum(r['success'] for r in rows if r['id'] in selected and r['category']==cat)>=10

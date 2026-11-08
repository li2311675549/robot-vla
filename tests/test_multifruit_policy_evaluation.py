"""The scoring fixture may know truth; policy input and checkpoint admission may not."""
import pytest

from robot_exam.vla.multifruit_evaluation import check_server_metadata, evaluate, fruit_body_names


def test_server_must_match_action_camera_and_checkpoint_contract():
    valid={'action_semantics':'ur5_absolute_joint_goal_v1',
           'actual_cameras':['front','wrist'],'checkpoint_step_label':'9999',
           'repo_id':'local/robot_exam_stage21_train','norm_stats_sha256':'expected'}
    check_server_metadata(valid,'9999')
    for key,value in [('action_semantics','delta'),('actual_cameras',['front']),('checkpoint_step_label','4999'),
                      ('repo_id','local/old'),('norm_stats_sha256','old_stats')]:
        with pytest.raises(ValueError):
            check_server_metadata({**valid,key:value},'9999','expected')


def test_failed_initialization_is_saved_and_never_counted_as_success(tmp_path):
    result=evaluate(tmp_path/'missing_fixture',tmp_path/'result','mock_hold')
    assert not result['success'] and result['status']=='FAIL'
    assert result['initial_state_restores']==0 and not result['model_inference']
    assert (tmp_path/'result/result.json').is_file()
    with pytest.raises(FileExistsError):
        evaluate(tmp_path/'missing_fixture',tmp_path/'result','mock_hold')


def test_static_fruit_table_is_not_scored_as_a_dropped_fruit():
    from types import SimpleNamespace
    model=SimpleNamespace(nbody=5,body_jntnum=[0,0,1,1,0],
        body=lambda i:SimpleNamespace(name=['world','fruit_table','fruit_0','fruit_1','fruit_sign'][i]))
    assert fruit_body_names(model)==['fruit_0','fruit_1']

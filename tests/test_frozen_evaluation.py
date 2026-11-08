import sys
from robot_exam.simulation import ROOT

sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_frozen import summarize


def test_complete_evaluation_retains_failures_and_no_success_time_for_failed_prefix():
    records=[{'success':True,'sim_seconds':100.,'wall_seconds':20.,'num_retries':1,
              'phase_metrics':{'opened':True,'entered':True,'closed_after_entry':True}},
             {'success':False,'sim_seconds':50.,'wall_seconds':10.,'root_failure':'dock',
              'phase_metrics':{'opened':False,'entered':False,'closed_after_entry':False}},
             {'success':False,'sim_seconds':80.,'wall_seconds':15.,'root_failure':'handle_grasp',
              'phase_metrics':{'opened':True,'entered':True,'closed_after_entry':False}}]
    summary=summarize(records)
    assert summary['episodes']==3 and summary['complete_success_rate']==1/3
    assert summary['successful_sim_seconds_mean']==100.
    assert summary['failures']=={'dock':1,'handle_grasp':1}
    assert summary['phase_successes']=={'opened':2,'entered':2,'closed_after_entry':1}
    assert summarize(records[1:])['successful_sim_seconds_mean'] is None

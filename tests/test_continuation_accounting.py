import importlib.util
from pathlib import Path
import sys


def test_all_attempts_and_startup_errors_are_never_hidden_by_supplement():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('continuation',scripts/'run_trainseen_continuation_v6.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[dict(skill='apple',evaluation_noise_seed=i,success=False,error=None) for i in range(23)]
    for i in (6,7,8):rows[i]['error']={'category':'InvalidMessage','message':'did not receive a valid HTTP response'}
    rows[20]['success']=True
    r=module.summarize(rows)['apple']
    assert r['all_attempts']==23 and r['original_predeclared_attempts']==20 and r['supplementary_attempts']==3
    assert r['infrastructure_failures']==3 and r['valid_model_evaluations']==20
    assert r['all_attempt_success_rate']==1/23 and r['conditional_model_success_rate']==1/20
    assert r['original_predeclared_successes']==0 and r['supplementary_successes']==1


def test_websocket_shutdown_is_network_failure_and_not_a_grasp_failure():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from analyze_remote_fruit_evaluation import failure_category
    for kind,message in [('ConnectionClosedError','no close frame received or sent'),
                         ('InvalidMessage','did not receive a valid HTTP response')]:
        assert failure_category({'success':False,'error':{'category':kind,'message':message}})=='network_or_inference_response_failure'
    assert failure_category({'success':False,'error':None,'maximum_lift_m':0.})=='no_valid_lift'

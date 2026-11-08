"""Dataset protocol tests, not fabricated physical success experiments."""
import json,runpy,sys
from pathlib import Path
import numpy as np
from robot_exam.simulation import ROOT


def test_statistics_exclude_validation_and_failed_executor(tmp_path,monkeypatch):
    source=tmp_path/'source';out=tmp_path/'manifest';protocol=tmp_path/'splits.json'
    protocol.write_text(json.dumps({'train':[201],'validation':[231],'test':[251],
        'action_semantics':'ur5_absolute_joint_goal_v1'}))
    for seed,value,deployed in ((201,1.,True),(231,9.,True),(251,20.,False)):
        folder=source/str(seed);folder.mkdir(parents=True)
        (folder/'episode.json').write_text(json.dumps({'success':True,'schema_version':3,
            'action_semantics':'ur5_absolute_joint_goal_v1','seed':seed,'skill':'apple','frames':2,'expert_source':'unit_fixture'}))
        (folder/'validation.json').write_text(json.dumps({'status':'PASS','max_replay_qpos_error':0}))
        (folder/'low_rate_replay.json').write_text(json.dumps({'low_rate_physical_placement_pass':True}))
        (folder/'vla_executor_replay.json').write_text(json.dumps({'success':deployed}))
        (folder/'frames.jsonl').write_text('\n'.join(json.dumps({'observation.state':[value]*7,'action':[value]*7}) for _ in range(2)))
    monkeypatch.setattr(sys,'argv',['prepare_vla_manifest',str(source),'--output',str(out),
        '--splits',str(protocol),'--require-executor-success'])
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    runpy.run_path(str(ROOT/'scripts/prepare_vla_manifest.py'),run_name='__main__')
    report=json.loads((out/'manifest.json').read_text());stats=json.loads((out/'assets/original_ur5/norm_stats.json').read_text())
    assert report['successful_episodes']==2
    assert report['excluded'][0]['seed']==251
    assert 'deployment_executor_physical_replay' in report['excluded'][0]['reasons']
    assert np.allclose(stats['norm_stats']['state']['mean'],1.)

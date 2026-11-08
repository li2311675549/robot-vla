import numpy as np
import pytest

from robot_exam.data.mixing import frame_weights,chunk_statistics


def episodes():
    return [dict(source=s,split='train',episode=i,scene_context_id=f'{s}_{i}',
                 category=c,phase=p,frames=3+i)
            for s in ('old','new') for i,(c,p) in enumerate(
                (c,p) for c in ('apple','banana','orange')
                for p in (('whole',) if s=='old' else ('grasp','place')))]


def test_mixture_balances_sources_classes_and_phases_despite_lengths():
    es=episodes();weights,offsets,_=frame_weights(es)
    for source,mass in [('old',.2),('new',.8)]:
        for c in ('apple','banana','orange'):
            got=sum(weights[offsets[i]:offsets[i+1]].sum() for i,e in enumerate(es)
                    if e['source']==source and e['category']==c)
            assert got==pytest.approx(mass/3)
    assert weights.sum()==pytest.approx(1.)


@pytest.mark.parametrize('split',['validation','test','ood'])
def test_heldout_data_never_enter_sampler_or_statistics(split):
    es=episodes();es[0]['split']=split
    with pytest.raises(ValueError,match='cannot enter'):frame_weights(es)


def test_chunk_statistics_repeat_last_action_without_episode_leak():
    es=episodes()
    for e in es:e['frames']=3
    states=np.zeros((len(es)*3,7));actions=np.tile(np.arange(3),len(es))[:,None]*np.ones((1,7))
    stats=chunk_statistics(es,states,actions,horizon=2)
    np.testing.assert_allclose(stats['actions']['mean'],np.full(7,8/6))
    np.testing.assert_allclose(stats['state']['mean'],np.zeros(7))

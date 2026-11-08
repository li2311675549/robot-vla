"""Reject silent index/statistics drift before the weighted training factory runs."""
import json

import numpy as np
import pytest

from robot_exam.data.mixing import frame_weights
from robot_exam.vla.mixed_training import SOURCE_REPOS, read_mixture, resume_offset


def release(tmp_path):
    episodes = []
    for source, repo in SOURCE_REPOS.items():
        for i, category in enumerate(('apple', 'banana', 'orange')):
            episodes.append(dict(source=source, split='train', episode=i,
                category=category, phase='grasp', frames=i+2, scene_context_id=f'{source}_{i}'))
        meta = tmp_path/'lerobot'/repo/'meta'
        meta.mkdir(parents=True)
        (meta/'info.json').write_text(json.dumps({'fps': 5, 'total_episodes': 3, 'total_frames': 9}))
    mix = tmp_path/'mixture'
    mix.mkdir()
    prep = {'action_semantics': 'ur5_absolute_joint_goal_v1', 'fps': 5,
            'action_horizon': 10, 'episodes': episodes}
    (mix/'preparation.json').write_text(json.dumps(prep))
    np.savez(mix/'sampling_weights.npz', weights=frame_weights(episodes)[0])
    for base in (mix, tmp_path):
        assets = base/'assets/original_ur5'
        assets.mkdir(parents=True)
        (assets/'norm_stats.json').write_text('{}')
    return prep


def test_mixture_source_mass_and_qualified_counts(tmp_path):
    release(tmp_path)
    _, weights = read_mixture(tmp_path)
    assert weights.shape == (18,)
    np.testing.assert_allclose([weights[:9].sum(), weights[9:].sum()], [.2, .8])


@pytest.mark.parametrize('fault', ['heldout', 'order', 'weights', 'stats', 'frames'])
def test_mixture_refuses_leakage_or_silent_data_drift(tmp_path, fault):
    prep = release(tmp_path)
    if fault == 'heldout':
        prep['episodes'][0]['split'] = 'validation'
    elif fault == 'order':
        prep['episodes'].reverse()
    elif fault == 'weights':
        np.savez(tmp_path/'mixture/sampling_weights.npz', weights=np.ones(18)/18)
    elif fault == 'stats':
        (tmp_path/'assets/original_ur5/norm_stats.json').write_text('{"different":true}')
    else:
        path = tmp_path/'lerobot'/SOURCE_REPOS['new']/'meta/info.json'
        path.write_text(json.dumps({'fps': 5, 'total_frames': 10, 'total_episodes': 3}))
    (tmp_path/'mixture/preparation.json').write_text(json.dumps(prep))
    with pytest.raises(ValueError):
        read_mixture(tmp_path)


def test_sampler_resume_uses_only_completed_optimizer_checkpoint(tmp_path):
    for name in ('999', '1999', '3999.orbax-checkpoint-tmp', '3999'):
        p = tmp_path/name
        p.mkdir()
        if name == '999':
            for suffix in ('_CHECKPOINT_METADATA', 'params/_METADATA', 'train_state/_METADATA'):
                f = p/suffix
                f.parent.mkdir(exist_ok=True)
                f.write_text('{}')
    assert resume_offset(tmp_path, False) == 0
    assert resume_offset(tmp_path, True) == 1000
    with pytest.raises(FileNotFoundError):
        resume_offset(tmp_path/'1999', True)

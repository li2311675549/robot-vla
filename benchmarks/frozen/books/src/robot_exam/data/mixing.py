"""Train-only source/class/phase/episode balancing and matched chunk statistics.

These helpers prepare sampling, not a claim of a qualified training release.
Images stay in their original LeRobot datasets; no duplicated old episodes.
"""
from collections import defaultdict

import numpy as np


def frame_weights(episodes, source_weights=None):
    source_weights = source_weights or {'old': .2, 'new': .8}
    if (set(source_weights) != {e['source'] for e in episodes}
            or any(not np.isfinite(w) or w <= 0 for w in source_weights.values())
            or not np.isclose(sum(source_weights.values()), 1.)):
        raise ValueError('Every declared source must be present with positive normalized weight')
    identities=set(); scenes={}; buckets=defaultdict(list)
    for i,e in enumerate(episodes):
        if e['split']!='train':
            raise ValueError('Validation/test/OOD cannot enter training weights or statistics')
        key=(e['source'],e['episode'])
        if key in identities:raise ValueError('Duplicated episode')
        identities.add(key)
        if not isinstance(e['frames'],int) or e['frames']<=0:
            raise ValueError('Positive episode frame count required')
        if e['category'] not in ('apple','banana','orange'):
            raise ValueError('Unknown fruit category')
        scene=(e['source'],e['scene_context_id'])
        if scene in scenes and scenes[scene]!=e['split']:raise ValueError('Scene split leakage')
        scenes[scene]=e['split']
        buckets[(e['source'],e['category'],e['phase'])].append(i)
    weights=np.empty(sum(e['frames'] for e in episodes),dtype=np.float64)
    offsets=np.cumsum([0]+[e['frames'] for e in episodes]); episode_mass={}
    for (source,category,phase),indices in buckets.items():
        classes={e['category'] for e in episodes if e['source']==source}
        if classes!={'apple','banana','orange'}:
            raise ValueError('Each source must cover all three fruit classes')
        phases={e['phase'] for e in episodes if e['source']==source and e['category']==category}
        mass=source_weights[source]/len(classes)/len(phases)/len(indices)
        for i in indices:
            weights[offsets[i]:offsets[i+1]]=mass/episodes[i]['frames']
            episode_mass[i]=mass
    np.testing.assert_allclose(weights.sum(),1.,atol=1e-12)
    return weights,offsets,episode_mass


def weighted_stats(values, weights):
    values=np.asarray(values,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if (values.ndim!=2 or weights.shape!=(len(values),) or not np.isfinite(values).all()
            or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0):
        raise ValueError('Finite observations and nonnegative weights required')
    keep=weights>0;values=values[keep];weights=weights[keep]/weights[keep].sum()
    mean=np.sum(values*weights[:,None],axis=0)
    variance=np.sum((values-mean)**2*weights[:,None],axis=0)
    quantiles=[]
    for column in values.T:
        order=np.argsort(column,kind='stable');w=weights[order]
        quantiles.append(np.interp([.01,.99],np.cumsum(w)-.5*w,column[order]))
    quantiles=np.asarray(quantiles)
    return {'mean':mean.tolist(),'std':np.sqrt(variance).tolist(),
            'q01':quantiles[:,0].tolist(),'q99':quantiles[:,1].tolist()}


def chunk_statistics(episodes, states, actions, horizon=10, source_weights=None):
    if not isinstance(horizon,int) or horizon<=0:raise ValueError('Positive chunk horizon required')
    weights,offsets,_=frame_weights(episodes,source_weights)
    states=np.asarray(states);actions=np.asarray(actions)
    if states.shape!=(len(weights),7) or actions.shape!=states.shape:
        raise ValueError('Aligned7D state/high-level-action rows required')
    chunks=[];chunk_weights=[]
    for i,e in enumerate(episodes):
        start,stop=offsets[i:i+2]
        # LeRobot clamps future indices inside each episode, repeating its
        # last action. Never allow a chunk to cross into a different scene.
        indices=np.minimum(np.arange(e['frames'])[:,None]+np.arange(horizon),e['frames']-1)
        chunks.append(actions[start:stop][indices].reshape(-1,7))
        chunk_weights.append(np.repeat(weights[start:stop]/horizon,horizon))
    return {'state':weighted_stats(states,weights),
            'actions':weighted_stats(np.concatenate(chunks),np.concatenate(chunk_weights))}

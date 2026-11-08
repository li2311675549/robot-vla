"""Scene-grouped collection intent; a plan never counts as collected data."""
from collections import Counter
import math


CATEGORIES = ('apple', 'banana', 'orange')


def collection_registry(config):
    """Reserve each scene/seed once, including all target/phase descendants.

    Seeds define split membership, not OOD status. OOD rollout intent remains
    uninstantiated until a separate distribution-level protocol is frozen.
    """
    target = config['new_data_target']
    ranges = config['seed_ranges_proposed_not_started']
    rows = []
    for split, seed_key in (('train', 'train'), ('validation', 'validation'), ('test', 'id_test')):
        counts = target['scene_context_counts_per_split'][split]
        start, stop = ranges[seed_key]
        objects = [n for n in (10, 5, 3) for _ in range(counts[str(n)])]
        if split == 'train':
            pilot = [10, 10, 5, 5, 3, 3]
            for n in pilot:
                objects.remove(n)
            objects = pilot + objects
        if len(objects) != stop-start+1:
            raise ValueError('Seed range and scene counts disagree')
        for seed, n in zip(range(start, stop+1), objects):
            scene = f'multifruit_{seed}'
            rows.append({'scene_context_id': scene, 'seed': seed, 'split': split,
                         'object_count': n, 'status': 'PLANNED_NOT_COLLECTED',
                         'parents': [{'parent_id': f'{scene}_{category}', 'category': category,
                                      'phases': ['grasp', 'place']} for category in CATEGORIES]})
    if len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('Seed leakage between scene groups')
    if len(rows) != target['independent_scene_contexts_proposed']:
        raise ValueError('Total scene count mismatch')
    aggregate = Counter(str(r['object_count']) for r in rows)
    if dict(aggregate) != target['scene_context_counts_by_objects']:
        raise ValueError('Clutter distribution mismatch')
    old, new = config['mixing_proposal']['sampling_old_new_initial']
    if not all(math.isfinite(v) and v >= 0 for v in (old, new)) or not math.isclose(old+new, 1.):
        raise ValueError('Mixture weights must sum to1')
    if new <= old:
        raise ValueError('User requires new complex scenes to dominate')
    return {'schema': 'multifruit_collection_intent_v1', 'status': 'PLAN_ONLY',
            'accepted_parent_count': 0, 'collection': rows,
            'sampling_weights': {'compatible_old': old, 'qualified_new': new},
            'ood_status': 'RESERVED_NOT_INSTANTIATED_REQUIRES_SEPARATE_DISTRIBUTION_PROTOCOL',
            'sampling_gate': 'Do not fill missing new data with duplicated old scenes or failed parents'}


def check_admission(records, registry):
    """Fail closed: both parked phases and parent evidence must qualify.

    Checks the evidence summary contract; does not replace the physical
    scorer, replay runner, or byte-level dataset integrity checks.
    """
    expected = {p['parent_id']: r for r in registry['collection'] for p in r['parents']}
    seen = set()
    for record in records:
        parent = record['parent_id']
        if parent in seen:
            raise ValueError('Duplicate parent is not new coverage')
        seen.add(parent)
        if parent not in expected or record['split'] != expected[parent]['split']:
            raise ValueError('Unknown parent or cross-split leak')
        if not record.get('physical_parent_success'):
            raise ValueError('Physical failure cannot enter positive data')
        if record.get('scene_object_count') != expected[parent]['object_count']:
            raise ValueError('Scene object count changed')
        if record.get('expert_source') != 'oracle':
            raise ValueError('This collection protocol requires declared Oracle expert provenance')
        if not record.get('language_qa_pass'):
            raise ValueError('Target-language ambiguity not cleared')
        phases = record.get('phases', {})
        if set(phases) != {'grasp', 'place'}:
            raise ValueError('Both phases required; one grasp is not a complete parent')
        for phase in phases.values():
            if phase.get('action_semantics') != 'ur5_absolute_joint_goal_v1':
                raise ValueError('Incompatible action semantics')
            if not all(phase.get(k) is True for k in ('physical_success', 'parking_pass',
                      'full_rate_replay_pass', 'low_rate_replay_pass', 'dual_camera_sync_pass',
                      'lerobot_openpi_load_pass')):
                raise ValueError('Pending or failed phase qualification')
    return {'status': 'PASS', 'accepted_parents': len(seen),
            'accepted_by_split': dict(Counter(expected[p]['split'] for p in seen))}

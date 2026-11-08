#!/usr/bin/env python3
"""Run the existing physical expert with explicit scene-group split metadata.

Only bookkeeping is extended. No expert action, target selection, success
criterion, physics parameter or replay tolerance is changed.
"""
import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
import collect_multifruit_parent as parent
from robot_exam.data.multifruit_plan import collection_registry


def run(args):
    proposal = json.loads((ROOT / 'configs/stage20_multifruit_dataset_expansion_proposal_v1.json').read_text())
    registry = {row['seed']: row for row in collection_registry(proposal)['collection']}
    scene = registry[args.seed]
    if scene['split'] != args.split or scene['object_count'] != args.fruit_count:
        raise ValueError('Scene-group split/object-count differs from the reserved registry')
    original_recorder = parent.EpisodeRecorder

    class SplitRecorder(original_recorder):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            self.meta.update(dataset_split=args.split,
                             split_protocol='scene_group_registry_v1_frozen_before_collection',
                             not_online_policy_evaluation=True)

    parent.EpisodeRecorder = SplitRecorder
    parent.collect(args.output, args.seed, args.category, args.config,
                   precision_dock_strategy='direct', guarded_dock=True,
                   twist_braking_all=True, target_selector='leftmost',
                   require_hardware=True, fruit_count=args.fruit_count)
    result_path = args.output / 'result.json'
    result = json.loads(result_path.read_text())
    result['original_collector_split_default'] = result['training_split']
    result['training_split'] = args.split
    result['split_provenance'] = {'source': 'scene_group_registry_v1', 'seed': args.seed,
                                  'scene_context_id': scene['scene_context_id'],
                                  'bookkeeping_only_no_physical_outcome_changed': True,
                                  'heldout_forbidden_in_training_statistics': args.split != 'train'}
    result_path.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--category', choices=['apple', 'banana', 'orange'], required=True)
    parser.add_argument('--fruit-count', type=int, choices=[3, 5, 10], required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    run(parser.parse_args())

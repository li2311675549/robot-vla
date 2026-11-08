#!/usr/bin/env python3
"""Prepare small, phase-aware format candidates; not a mixed training release."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from robot_exam.data.multifruit_plan import collection_registry


def prepare(parents, output, reviewed_images, hardware_view_root=None, require_hardware=False, camera_near_m=None,
            require_scene_preservation=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing manifest')
    config = json.loads((ROOT/'configs/stage20_multifruit_dataset_expansion_proposal_v1.json').read_text())
    registry = collection_registry(config)
    scenes = {r['seed']: r for r in registry['collection']}
    output.mkdir(parents=True)
    accepted, states, actions = [], [], []
    for parent in parents:
        parent = Path(parent).resolve()
        result = json.loads((parent/'result.json').read_text())
        if result['status'] != 'PHYSICAL_PARENT_PASS_PENDING_REPLAY_AND_EXPORT' or result.get('collision_events') or any(result['data_warning_counts']):
            raise ValueError(f'Parent physical operation not qualified: {parent}')
        if result.get('source_changed_during_attempt',False):
            raise ValueError('Do not admit a collection run whose expert source changed during execution')
        if require_scene_preservation and result.get('scene_preservation',{}).get('success') is not True:
            raise ValueError(f'Parent lacks verified preservation of non-target fruit: {parent}')
        scene = scenes[result['seed']]
        if scene['split'] != 'train' or scene['object_count'] not in (3,5,10):
            raise ValueError('This tiny pilot manifest is training-scene only; no heldout statistics')
        body = result['target']['body']
        category = result['category']
        for phase in ('grasp', 'place'):
            folder = parent/phase
            if hardware_view_root is not None:
                candidate = Path(hardware_view_root)/f'multifruit_{result["seed"]}_{category}'/phase
                if candidate.exists():
                    reconstruction = json.loads((candidate/'hardware_reconstruction.json').read_text())
                    if (reconstruction['status'] != 'PASS'
                            or Path(reconstruction['source_episode']).resolve() != folder.resolve()
                            or reconstruction['maximum_qpos_difference'] > 1e-4):
                        raise ValueError('Hardware view must replay this exact source phase')
                    folder = candidate
            meta = json.loads((folder/'episode.json').read_text())
            full = json.loads((folder/'validation.json').read_text())
            low = json.loads((folder/'action_chunk_replay_v2.json').read_text())
            if not(meta['success'] and meta['schema_version'] == 3 and meta['sensor_schema'] == 'front_wrist_rgb_v1'
                   and meta['target_body'] == body and meta['scene_object_count'] == scene['object_count']
                   and full['status'] == 'PASS' and low['status'] == 'PASS'
                   and low['physical_phase_success'] and low['target_body'] == body):
                raise ValueError(f'Phase/replay qualification failed: {folder}')
            renderer = meta.get('opengl', {})
            if require_hardware and ('nvidia' not in renderer.get('vendor', '').lower()
                                     or 'llvmpipe' in renderer.get('renderer', '').lower()):
                raise ValueError(f'Hardware provenance missing: {folder}')
            if camera_near_m is not None and abs(meta.get('camera_clipping',{}).get('near_m',-1)-camera_near_m)>1e-6:
                raise ValueError(f'Camera clipping profile mismatch: {folder}')
            # The caller must actually inspect these images before declaring
            # the caption/label correspondence reviewed. Store their hashes.
            if not reviewed_images:
                raise ValueError('Initial target and bin-label image review required')
            rows = [json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
            states.extend(r['observation.state'] for r in rows)
            actions.extend(r['action'] for r in rows)
            image = folder/rows[0]['rgb']
            accepted.append({'path': str(folder), 'seed': result['seed'], 'split': 'train',
                             'scene_context_id': scene['scene_context_id'],
                             'parent_id': f'multifruit_{result["seed"]}_{category}',
                             'skill': category, 'phase': phase, 'frames': len(rows),
                             'expert_source': 'oracle', 'physical_parent_success': True,
                             'action_semantics': meta['action_semantics'],
                             'action_execution_contract': 'expert_500hz_goals_validated_5hz_chunk_replay_v1',
                             'source_phase_metadata_unchanged': True,
                             'physical_parent_path': str(parent),
                             'non_target_scene_preservation': result.get('scene_preservation'),
                             'scene_object_count': scene['object_count'],
                             'opengl': renderer,
                             'camera_clipping':meta.get('camera_clipping',{}),
                             'reconstructed_view_not_new_parent': folder != parent/phase,
                             'image_review': {'target_or_bin_label_reviewed': True, 'path': str(image),
                                              'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}})
    if len({(x['parent_id'], x['phase']) for x in accepted}) != len(accepted):
        raise ValueError('Duplicate parent phase')
    stats = {}
    for name, values in [('state', states), ('actions', actions)]:
        a = np.asarray(values, dtype=float)
        stats[name] = {'mean': a.mean(axis=0).tolist(), 'std': a.std(axis=0).tolist(),
                       'q01': np.quantile(a, .01, axis=0).tolist(), 'q99': np.quantile(a, .99, axis=0).tolist()}
    assets = output/'assets/original_ur5'
    assets.mkdir(parents=True)
    (assets/'norm_stats.json').write_text(json.dumps({'norm_stats': stats}, indent=2))
    manifest = {'status': 'PILOT_FORMAT_CANDIDATES_NOT_TRAINING_RELEASE', 'accepted': accepted,
                'parent_operations': len({x['parent_id'] for x in accepted}),
                'independent_scene_contexts': len({x['scene_context_id'] for x in accepted}),
                'frames': len(actions), 'normalization_split': 'train_pilot_only',
                'not_mixed_with_old_data': True, 'training_ready': False,
                'hardware_provenance_required': require_hardware,
                'non_target_preservation_required': require_scene_preservation,
                'pending': ['LeRobot export/load', 'native OpenPI load', 'coverage expansion and heldout qualification']}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parents', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--images-reviewed', action='store_true', help='Use only after inspecting initial target and bin-label RGB')
    p.add_argument('--hardware-view-root', type=Path)
    p.add_argument('--require-hardware-renderer', action='store_true')
    p.add_argument('--camera-near-m',type=float)
    p.add_argument('--require-scene-preservation',action='store_true')
    args = p.parse_args()
    print(json.dumps(prepare(args.parents, args.output, args.images_reviewed,
                             args.hardware_view_root, args.require_hardware_renderer,args.camera_near_m,
                             args.require_scene_preservation), indent=2))

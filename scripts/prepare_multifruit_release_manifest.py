#!/usr/bin/env python3
"""Audit physical evidence and create split-safe stationary-skill manifests.

This never reruns a controller, relabels a failed operation, or derives training
statistics from validation/test. Human/agent image review and format loading
remain separate recorded qualification steps.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import ROOT
from robot_exam.data.multifruit_plan import collection_registry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(parents, output):
    if output.exists():
        raise FileExistsError('Use a fresh manifest version')
    proposal = json.loads((ROOT / 'configs/stage20_multifruit_dataset_expansion_proposal_v1.json').read_text())
    registry = {r['seed']: r for r in collection_registry(proposal)['collection']}
    accepted, rejected, seen = [], [], set()
    values = {'state': [], 'actions': []}
    review_rows = []
    for parent_path in sorted(set(str(Path(p).resolve()) for p in parents)):
        parent = Path(parent_path)
        try:
            physical = json.loads((parent / 'result.json').read_text())
            scene = registry[physical['seed']]
            category, target = physical['category'], physical['target']['body']
            parent_id = f'multifruit_{physical["seed"]}_{category}'
            if parent_id in seen:
                raise ValueError('Repeated parent is not a new independent target operation')
            if (physical['status'] != 'PHYSICAL_PARENT_PASS_PENDING_REPLAY_AND_EXPORT'
                    or physical.get('collision_events') or any(physical['data_warning_counts'])
                    or physical.get('source_changed_during_attempt')
                    or physical.get('scene_preservation', {}).get('success') is not True):
                raise ValueError('Parent physical/scene preservation qualification failed')
            if scene['split'] != 'train' and physical.get('training_split') != scene['split']:
                raise ValueError('Heldout source lacks explicit split bookkeeping')
            if physical['target']['visible_pixels'] < 30 or physical['target']['image_relation'] != 'leftmost':
                raise ValueError('Target caption visibility/selection mismatch')
            candidates, contacts, parent_values = [], [], {'state': [], 'actions': []}
            for phase in ('grasp', 'place'):
                folder = parent / phase
                meta = json.loads((folder / 'episode.json').read_text())
                full = json.loads((folder / 'validation.json').read_text())
                low = json.loads((folder / 'action_chunk_replay_v2.json').read_text())
                if (not meta['success'] or meta['schema_version'] != 3
                        or meta['action_semantics'] != 'ur5_absolute_joint_goal_v1'
                        or meta['sensor_schema'] != 'front_wrist_rgb_v1'
                        or meta['target_body'] != target or meta['scene_object_count'] != scene['object_count']
                        or not meta.get('parking_ready_at_start') or meta.get('parking_stable_seconds', 0) < 2
                        or full['status'] != 'PASS' or low['status'] != 'PASS'
                        or not low['physical_phase_success'] or low['target_body'] != target
                        or low.get('parking_faulted') or any(low['warnings'])
                        or 'nvidia' not in meta.get('opengl', {}).get('vendor', '').lower()
                        or abs(meta.get('camera_clipping', {}).get('near_m', -1) - .005) > 1e-6):
                    raise ValueError(f'Phase/controller/replay/camera qualification failed: {phase}')
                rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
                state = np.asarray([r['observation.state'] for r in rows])
                action = np.asarray([r['action'] for r in rows])
                if state.shape != (len(rows), 7) or action.shape != state.shape:
                    raise ValueError('Observation/action dimensions must both be seven')
                if not np.isfinite(state).all() or not np.isfinite(action).all() or np.any((action[:, 6] < 0) | (action[:, 6] > 1)):
                    raise ValueError('Non-finite or illegal gripper action')
                sampled = {0, len(rows) // 2, len(rows) - 1}
                images = []
                for index, row in enumerate(rows):
                    t = row['timestamp']
                    if abs(row['observation_timestamp'] - t) > 1e-8 or abs(row['action_timestamp'] - t) > 1e-8:
                        raise ValueError('State/action timestamp mismatch')
                    if any(abs(x - t) > 1e-8 for x in row['camera_timestamps'].values()):
                        raise ValueError('Dual camera timestamp mismatch')
                    if row['task'] != meta['language_instruction']:
                        raise ValueError('Frame/episode prompt differs')
                    for key in ('rgb', 'rgb_wrist'):
                        path = folder / row[key]
                        with Image.open(path) as image:
                            if image.size != (640, 480):
                                raise ValueError('Image shape differs from deployment')
                            image.verify()
                        if index in sampled:
                            with Image.open(path) as image:
                                if np.asarray(image).std() < 2:
                                    raise ValueError('Blank/corrupt sampled camera image')
                            images.append({'path': str(path), 'sha256': sha(path), 'index': index, 'camera': key})
                    if not (folder / row['depth']).is_file():
                        raise ValueError('Depth sidecar missing')
                expected = (f'Pick up the leftmost {category} in the front camera image and hold it above the table.'
                            if phase == 'grasp' else f'Place the held {category} into the bin labelled {category.upper()} and withdraw the gripper.')
                if meta['language_instruction'] != expected:
                    raise ValueError('Prompt does not identify the selected object or labelled bin')
                candidates.append({'path': str(folder), 'seed': physical['seed'], 'split': scene['split'],
                    'scene_context_id': scene['scene_context_id'], 'parent_id': parent_id,
                    'skill': category, 'phase': phase, 'frames': len(rows), 'expert_source': 'oracle',
                    'physical_parent_success': True, 'physical_parent_path': str(parent),
                    'action_semantics': meta['action_semantics'],
                    'action_execution_contract': 'expert_500hz_goals_validated_5hz_chunk_replay_v1',
                    'scene_object_count': scene['object_count'], 'non_target_scene_preservation': physical['scene_preservation'],
                    'opengl': meta['opengl'], 'camera_clipping': meta['camera_clipping'],
                    'source_phase_metadata_unchanged': True,
                    'image_integrity': 'ALL_DUAL_RGB_HEADERS_AND_CRC_PASS; first/middle/last pixels nonblank',
                    'image_review_samples': images, 'visual_semantic_review': 'PENDING_CONTACT_SHEET_REVIEW'})
                contacts += [folder / rows[0]['rgb'], folder / rows[0]['rgb_wrist']]
                if scene['split'] == 'train':
                    parent_values['state'].append(state)
                    parent_values['actions'].append(action)
            accepted += candidates
            for name in values:
                values[name].extend(parent_values[name])
            seen.add(parent_id)
            review_rows.append((parent_id, scene['split'], contacts))
        except (ValueError, KeyError, FileNotFoundError) as error:
            rejected.append({'path': str(parent), 'reason': str(error)})
    if not accepted or not values['state']:
        raise ValueError('Need qualified data and a training partition for train-only statistics')
    output.mkdir(parents=True)
    stats = {}
    for name, arrays in values.items():
        array = np.concatenate(arrays)
        stats[name] = {'mean': array.mean(0).tolist(), 'std': array.std(0).tolist(),
                       'q01': np.quantile(array, .01, axis=0).tolist(), 'q99': np.quantile(array, .99, axis=0).tolist()}
    assets = output / 'assets/original_ur5'
    assets.mkdir(parents=True)
    (assets / 'norm_stats.json').write_text(json.dumps({'norm_stats': stats}, indent=2))
    summary = {'status': 'PHYSICAL_REPLAY_AND_INTEGRITY_PASS_FORMAT_AND_VISUAL_REVIEW_PENDING',
               'accepted': accepted, 'rejected': rejected, 'parent_operations': len(seen),
               'frames': sum(r['frames'] for r in accepted), 'non_target_preservation_required': True,
               'parent_by_split': dict(Counter(r['split'] for r in accepted if r['phase'] == 'grasp')),
               'context_by_split': {s: len({r['seed'] for r in accepted if r['split'] == s}) for s in ('train', 'validation', 'test')},
               'normalization_split': 'train_only', 'norm_statistics_note': 'QA unweighted frame statistics; mixture deployment uses separately recomputed weighted10-step chunk train-only statistics',
               'training_ready': False, 'server_contacted': False, 'ood_excluded_from_this_collection': True}
    (output / 'manifest.json').write_text(json.dumps(summary, indent=2))
    for split in ('train', 'validation', 'test'):
        folder = output / split
        folder.mkdir()
        rows = [r for r in accepted if r['split'] == split]
        sub = dict(summary, accepted=rows, frames=sum(r['frames'] for r in rows),
                   parent_operations=len({r['parent_id'] for r in rows}))
        (folder / 'manifest.json').write_text(json.dumps(sub, indent=2))
        shutil_assets = folder / 'assets/original_ur5'
        shutil_assets.mkdir(parents=True)
        (shutil_assets / 'norm_stats.json').write_bytes((assets / 'norm_stats.json').read_bytes())
    # Each sheet includes all four initial images per parent, never synthetic imagery.
    for page, offset in enumerate(range(0, len(review_rows), 18)):
        rows = review_rows[offset:offset + 18]
        sheet = Image.new('RGB', (1280, len(rows) * 140), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (parent_id, split, paths) in enumerate(rows):
            draw.text((2, i * 140), f'{parent_id} [{split}] grasp front/wrist | place front/wrist', fill='black')
            for j, path in enumerate(paths):
                with Image.open(path) as image:
                    image.thumbnail((310, 116))
                    sheet.paste(image, (j * 320, i * 140 + 20))
        sheet.save(output / f'contact_sheet_{page:03d}.jpg', quality=90)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parents', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.parents, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('accepted', 'rejected')}, indent=2))

#!/usr/bin/env python3
"""Read-only teacher-frame and initial-render input probe after a rollout batch.

Fresh connections use the declared same policy noise seed. This must never run
concurrently with the serialized physical batch, whose RNG it would perturb.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import _bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=18001)
    p.add_argument('--client-source', type=Path,
                   default=Path('/home/li/vla/openpi/packages/openpi-client/src'))
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve all diagnostic results')
    summary = json.loads((a.matrix / 'summary.json').read_text())
    integrity = json.loads((a.matrix / 'integrity.json').read_text())
    if summary['completed_cases'] != summary['expected_cases'] or integrity['source_hash_mismatches']:
        raise RuntimeError('Require completed source-frozen batch before resetting server connection RNG')
    sys.path.insert(0, str(a.client_source))
    from openpi_client.image_tools import resize_with_pad
    from robot_exam.vla.client import connect_openpi_policy
    entries = json.loads(a.manifest.read_text())['accepted']
    assert len(entries) == 6 and all(e['split'] == 'train' for e in entries)
    report = {'status': 'RUNNING', 'queries': [], 'actuator_commands': 0,
              'training_updates': 0, 'scope': 'Seen-train teacher observations and paired initial images; not physical success or a training ablation'}
    a.output.mkdir(parents=True)

    def save():
        (a.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')

    def query(entry, frame, observation, truth, condition):
        policy = connect_openpi_policy('127.0.0.1', a.port, compression='deflate')
        try:
            metadata = policy.get_server_metadata()
            assert metadata['checkpoint_step_label'] == '9999'
            assert metadata['jax_policy_seed_per_connection'] == 0
            prediction = policy.infer(observation)
        finally:
            policy._ws.close()
        actions = np.asarray(prediction['actions'])
        assert actions.shape == (10, 7) and np.isfinite(actions).all()
        row = {'seed': entry['seed'], 'skill': entry['skill'], 'frame': frame,
            'condition': condition, 'arm_mae_rad_per_joint': np.mean(np.abs(actions[:, :6]-truth[:, :6]), axis=0).tolist(),
            'first_arm_error_rad': np.abs(actions[0, :6]-truth[0, :6]).tolist(),
            'gripper_mae': float(np.mean(np.abs(actions[:, 6]-truth[:, 6]))),
            'predicted_actions': actions.tolist(), 'teacher_actions': truth.tolist(),
            'gripper_projection': prediction.get('gripper_projection'),
            'input_pixel_sha256': {name: hashlib.sha256(observation[name].tobytes()).hexdigest()
                                   for name in ('front_rgb', 'wrist_rgb')}}
        report['queries'].append(row)
        save()
        return actions

    for entry in entries:
        source = Path(entry['path'])
        frames = [json.loads(s) for s in (source / 'frames.jsonl').read_text().splitlines()]
        for index in (0, 30, 60):
            row = frames[index]
            truth = np.asarray([f['action'] for f in frames[index:index+10]], dtype=np.float32)
            obs = {'state': np.asarray(row['observation.state'], dtype=np.float32), 'prompt': row['task']}
            for field, key in (('front_rgb', 'rgb'), ('wrist_rgb', 'rgb_wrist')):
                with Image.open(source / row[key]) as image:
                    obs[field] = resize_with_pad(np.asarray(image.convert('RGB')), 224, 224)
            baseline = query(entry, index, obs, truth, 'recorded_train_pixels')
            if index == 0:
                case = a.matrix / f"{entry['seed']}_{entry['skill']}_openpi/observations"
                live = json.loads((case / '0000.json').read_text())
                state_difference = float(np.max(np.abs(np.asarray(live['state'], dtype=np.float32)-obs['state'])))
                assert state_difference < 1e-7, 'Initial state mismatch must be diagnosed separately'
                actual = dict(obs)
                for field in ('front_rgb', 'wrist_rgb'):
                    with Image.open(case / f'0000_{field}.png') as image:
                        actual[field] = np.asarray(image.convert('RGB'))
                predicted = query(entry, index, actual, truth, 'runtime_pixels_same_train_state')
                report.setdefault('initial_image_pairs', []).append({'seed': entry['seed'], 'skill': entry['skill'],
                    'initial_float32_state_difference': state_difference,
                    'pixel_mae': {key: float(np.mean(np.abs(actual[key].astype(float)-obs[key]))) for key in ('front_rgb', 'wrist_rgb')},
                    'maximum_prediction_change_rad': float(np.max(np.abs(predicted[:, :6]-baseline[:, :6])))})
    report.update(status='COMPLETED_READONLY_INPUT_PROBE', expected_queries=24)
    assert len(report['queries']) == 24
    save()
    print(json.dumps({'status': report['status'], 'queries': len(report['queries'])}))


if __name__ == '__main__':
    main()

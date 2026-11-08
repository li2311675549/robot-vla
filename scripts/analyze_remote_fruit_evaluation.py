#!/usr/bin/env python3
"""Read-only aggregation of all frozen remote-policy physical trials."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    def quantile(q):
        index = (len(values) - 1) * q
        low = int(index)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (index - low)
    return dict(count=len(values), minimum=values[0], median=statistics.median(values),
                p95=quantile(.95), maximum=values[-1], mean=statistics.mean(values))


def failure_category(result):
    if result.get('success'):
        return 'success'
    error = result.get('error') or {}
    message = error.get('message', '')
    if error.get('category') in ('process_failed', 'process_interrupted'):
        return 'process_failure_or_interruption'
    if error.get('category') in ('ConnectionClosedError','ConnectionClosedOK','ConnectionRefusedError',
                                  'ConnectionResetError','InvalidMessage') or any(token in message.lower() for token in (
                                      'response too stale','timed out','connection','no close frame','valid http response')):
        return 'network_or_inference_response_failure'
    if result.get('unexpected_contacts'):
        return 'unsafe_robot_collision'
    if result.get('parking_fault'):
        return 'parking_fault'
    if any(result.get('warnings', [])):
        return 'mujoco_data_warning'
    if 'Physical lift failed' in message:
        return 'physical_lift_failure'
    if error and error.get('category') != 'TimeoutError':
        return 'other_execution_error'
    if result.get('maximum_lift_m', 0.) <= .055:
        return 'no_valid_lift'
    if result.get('continuous_bilateral_hold_seconds', 0.) < 1.5:
        return 'insufficient_bilateral_hold'
    return 'placement_release_or_settling_incomplete'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.matrix / 'episodes.json').read_text())
    integrity = json.loads((args.matrix / 'integrity.json').read_text())
    if args.output.exists():
        raise FileExistsError('Preserve previous analysis')
    args.output.mkdir(parents=True)
    grouped, rows = {}, []
    frozen = json.loads((args.matrix/'freeze.json').read_text())
    checkpoint = frozen['protocol'].get('server_checkpoint_label', '以逐回合元数据为准')
    index = ['# 真实连续回合视频索引', '',
             f'驻车单果评估；expert为几何专家，openpi检查点{checkpoint}。所有失败视频保留；不是完整十果任务。', '']
    for result in records:
        path = Path(result['evidence'])
        queries_path = path.parent / 'queries.json'
        queries = json.loads(queries_path.read_text()) if queries_path.exists() else []
        category = failure_category(result)
        video = result.get('video') or {}
        row = dict(policy=result['policy'], skill=result['skill'], seed=result['seed'],
                   evaluation_noise_seed=result.get('evaluation_noise_seed'),
                   success=result.get('success', False), failure_category=category,
                   sim_seconds=result.get('sim_seconds'), wall_seconds=result.get('wall_seconds'),
                   maximum_lift_m=result.get('maximum_lift_m'),
                   continuous_bilateral_hold_seconds=result.get('continuous_bilateral_hold_seconds'),
                   placement_error_m=result.get('placement_error_m'),
                   queries=result.get('queries', 0), gripper_projected_chunks=result.get('gripper_projected_chunks', 0),
                   video_decode=video.get('full_decode'), video_frames=video.get('frames', 0),
                   evidence=str(path), video=str(path.parent / 'video/demo.mp4'), error=result.get('error'))
        rows.append(row)
        group = grouped.setdefault(result['policy'], dict(attempts=0, successes=0, by_class={},
            failures={}, latencies=[], sim_times=[], wall_times=[], success_sim_times=[],
            maximum_lifts=[], projected_chunks=0, queries=0, decoded_videos=0, data_warning_trials=0))
        group['attempts'] += 1
        group['successes'] += int(row['success'])
        cls = group['by_class'].setdefault(result['skill'], dict(attempts=0, successes=0))
        cls['attempts'] += 1; cls['successes'] += int(row['success'])
        scene=cls.setdefault('by_scene',{}).setdefault(str(result['seed']),dict(attempts=0,successes=0))
        scene['attempts']+=1;scene['successes']+=int(row['success'])
        if not row['success']:
            group['failures'][category] = group['failures'].get(category, 0) + 1
        group['latencies'].extend(q['query_wall_seconds'] for q in queries)
        for key, metric in [('sim_times', 'sim_seconds'), ('wall_times', 'wall_seconds'), ('maximum_lifts', 'maximum_lift_m')]:
            if result.get(metric) is not None: group[key].append(result[metric])
        if row['success']: group['success_sim_times'].append(result['sim_seconds'])
        group['projected_chunks'] += row['gripper_projected_chunks']
        group['queries'] += len(queries)
        group['decoded_videos'] += int(video.get('full_decode') == 'PASS')
        group['data_warning_trials'] += int(any(result.get('warnings', [])))
        index.append(f"- {result['seed']} {result['skill']} **{result['policy']} {'PASS' if row['success'] else 'FAIL'}** — "
                     f"[{category}]({path.parent / 'video/demo.mp4'})，{row['sim_seconds']} sim秒")
    for group in grouped.values():
        group['success_rate'] = group['successes'] / group['attempts']
        for key in ('latencies', 'sim_times', 'wall_times', 'success_sim_times', 'maximum_lifts'):
            group[key] = distribution(group[key])
    report = dict(scope='development_not_formal_stationary_single_fruit', integrity=integrity,
                  frozen=json.loads((args.matrix/'freeze.json').read_text()), groups=grouped, cases=rows,
                  interpretation='Failure labels are observable criteria, not proven learning root causes. Network failures remain failures in the total denominator. Latencies cover completed query/execution records only; the over-budget request is retained in the trial error. No complete ten-fruit task is tested here.')
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    with (args.output/'cases.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output/'VIDEO_INDEX.md').write_text('\n'.join(index)+'\n')
    print(json.dumps(grouped, indent=2))


if __name__ == '__main__':
    main()

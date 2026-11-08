#!/usr/bin/env python3
"""Plot measured task outcomes and controller replay differences separately."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous figures')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eval_root = a.root / 'remote_fruit_eval_v1'
    prep = a.root / 'data_preparation'
    inputs = [eval_root / name / 'comparison.json' for name in
              ('analysis_strict_v1', 'analysis_latency_v1', 'analysis_chunk_v1')]
    inputs += [prep / 'live_start_alignment_v1/results.json',
               prep / 'executor_matched6_release_v4/live_runtime_parity_v1/results.json']
    loaded = [json.loads(p.read_text()) for p in inputs]
    groups = [loaded[0]['groups']['expert'], loaded[0]['groups']['openpi'],
              loaded[1]['groups']['openpi'], loaded[2]['groups']['openpi']]
    outcomes = [{'label': label, 'attempts': row['attempts'], 'successes': row['successes']}
                for label, row in zip(('Oracle strict', 'Pi05 strict', 'Pi05 paused2s', 'Pi05 prefix5'), groups, strict=True)]
    replay = []
    for mode in ('sliced1', 'lookahead10'):
        values = [max(r['joint_state_max_error_from_expert_rad']) for r in loaded[3] if r['mode'] == mode]
        assert len(values) == 12
        replay.append({'label': mode, 'cases': len(values), 'worst_axis_error_rad': max(values),
                       'median_case_worst_axis_error_rad': statistics.median(values)})
    values = [max(r['joint_state_max_error_from_expert_rad']) for r in loaded[4]]
    assert len(values) == 6 and max(values) == 0
    replay.append({'label': 'Recaptured matched6', 'cases': 6, 'worst_axis_error_rad': 0.,
                   'median_case_worst_axis_error_rad': 0.})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    bars = axes[0].bar([x['label'] for x in outcomes], [100*x['successes']/x['attempts'] for x in outcomes],
                       color=['#39854a', '#b24136', '#c58a32', '#8e669c'])
    axes[0].set(ylim=(0, 110), ylabel='Complete physical single-fruit success (%)',
                title='Actual policy tests (separate protocols)')
    axes[0].tick_params(axis='x', labelrotation=18)
    axes[0].bar_label(bars, labels=[f"{x['successes']}/{x['attempts']}" for x in outcomes], padding=4)
    bars = axes[1].bar([x['label'] for x in replay], [x['worst_axis_error_rad'] for x in replay],
                       color=['#627d9e', '#368e8b', '#39854a'])
    axes[1].set(ylim=(0, .17), ylabel='Maximum joint deviation from recorded expert (rad)',
                title='Recorded expert goals replayed via actuators')
    axes[1].tick_params(axis='x', labelrotation=12)
    axes[1].bar_label(bars, labels=[f"{x['worst_axis_error_rad']:.5f}\n(n={x['cases']})" for x in replay], padding=4)
    fig.suptitle('Replay consistency is not learned-policy success', fontsize=13)
    a.output.mkdir(parents=True)
    for extension in ('png', 'svg'):
        fig.savefig(a.output / f'policy_vs_replay.{extension}', dpi=160)
    plt.close(fig)
    report = {'actual_policy_outcomes': outcomes, 'expert_replay_deviation': replay,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'scope': 'Measured outcomes, not a causal ablation. Different replay data groups are explicit.',
              'model_loaded': False, 'training_started': False}
    (a.output / 'source_values.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Plot measured parking segments; use the existing plotting environment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep previous experiment figures')
    segments = json.loads(args.source.read_text())
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fields = [('position_error_m', 1000, 10, 'Position drift (mm)'),
              ('heading_error_rad', 1000, 20, 'Heading drift (mrad)'),
              ('linear_speed_m_s', 1000, 20, 'Linear speed (mm/s)'),
              ('yaw_rate_rad_s', 1000, 30, 'Yaw speed (mrad/s)')]
    for segment in segments:
        rows = segment['samples']
        if not rows:
            continue
        admitted = next((row['time'] for row in rows if row['ready']), None)
        for axis, (field, scale, limit, label) in zip(axes, fields):
            axis.plot([r['time'] for r in rows],
                      [abs(r[field]) * scale for r in rows], lw=.7, color='#1565c0')
            if admitted is not None:
                axis.axvspan(rows[0]['time'], admitted, color='#ffa726', alpha=.12)
    for axis, (_, _, limit, label) in zip(axes, fields):
        axis.axhline(limit, color='#c62828', ls='--', lw=.9, label='Operation envelope')
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    axes[0].legend(loc='upper right')
    axes[-1].set_xlabel('Simulation time (s); orange = initial settling, gaps = wheel-driven navigation')
    fig.suptitle('Measured physical parking during complete three-book task')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()

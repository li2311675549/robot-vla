#!/usr/bin/env python3
"""Replay an existing paired dock fixture without rebuilding its grasp."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_dock_stage7 import trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategy', choices=['legacy', 'hysteresis','residual','signed'], default='hysteresis')
    parser.add_argument('--timeout', type=float, default=25.)
    parser.add_argument('--payload-dock-angular-speed',type=float)
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    metadata = json.loads((fixture / 'fixture.json').read_text())
    arrays = np.load(fixture / 'initial_state.npz')
    snapshot = {**metadata, **{key: arrays[key].copy() for key in arrays.files}}
    snapshot['state_spec'] = int(snapshot['state_spec'])
    scene = fixture / 'scene.xml'
    kind = fixture.parent.name
    overrides=({'payload_dock_angular_speed_rad_s':args.payload_dock_angular_speed}
               if args.payload_dock_angular_speed is not None else None)
    result = trial(scene, snapshot, args.strategy, args.output.resolve(), kind,
                   timeout=args.timeout,config_overrides=overrides)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

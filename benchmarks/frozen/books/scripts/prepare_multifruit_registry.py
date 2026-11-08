#!/usr/bin/env python3
"""Write the approved scene split reservation without collecting/training."""
import argparse
import json
from pathlib import Path
from _bootstrap import ROOT
from robot_exam.data.multifruit_plan import collection_registry

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT/'configs/stage20_multifruit_dataset_expansion_proposal_v1.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    registry = collection_registry(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(registry, stream, indent=2)
        stream.write('\n')
    print(f'{len(registry["collection"])} scene reservations; accepted parents0; no collection or training started')

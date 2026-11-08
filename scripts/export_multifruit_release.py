#!/usr/bin/env python3
"""Reuse tested LeRobot v2.1 export and retain audited parent/phase semantics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    parser.add_argument('--repo-id', required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(Path(__file__).with_name('export_lerobot.py')),
                    str(args.source), str(args.output), '--manifest', str(args.manifest),
                    '--split', args.split, '--require-schema3', '--repo-id', args.repo_id], check=True)
    records = {str(Path(row['path']).resolve()): row
               for row in json.loads(args.manifest.read_text())['accepted'] if row['split'] == args.split}
    path = args.output / 'conversion_result.json'
    conversion = json.loads(path.read_text())
    if len(records) != len(conversion['sources']):
        raise ValueError('Do not silently drop a manifest-qualified phase during conversion')
    for row in conversion['sources']:
        source = records[str(Path(row['raw_path']).resolve())]
        row['legacy_exporter_descriptor_before_enrichment'] = {
            'skill': row['skill'], 'action_execution_contract': row['action_execution_contract']}
        row.update(skill=source['skill'], phase=source['phase'], split=source['split'],
                   parent_id=source['parent_id'], scene_context_id=source['scene_context_id'],
                   action_execution_contract=source['action_execution_contract'])
    conversion.update(metadata_enrichment='Manifest-verified bookkeeping only; no images/actions/results changed',
                      actual_split=args.split, normalization_in_this_export='No training statistics estimated here')
    path.write_text(json.dumps(conversion, indent=2))
    print(json.dumps({'status': conversion['status'], 'split': args.split,
                      'episodes': conversion['episodes'], 'frames': conversion['frames']}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Check an illustrative selection against the unchanged original physical results."""
import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();directory=args.directory.resolve()
    manifest=json.loads((directory/'selection_manifest.json').read_text())
    original=json.loads((ROOT/'docs/results/fixed_scene_window_delivery_v8.json').read_text())
    seen=set();counts={};source_files=0
    for row in manifest['all_cases']:
        source=ROOT/row['source_result'];folder=directory/'cases'/row['case_id']
        assert str(source) in original['per_class'][row['skill']]['source_results']
        assert row['case_id'] not in seen;seen.add(row['case_id'])
        result=json.loads(source.read_text())
        assert result['success']==row['success'] and not result.get('error')
        assert sha(source)==row['result_sha256']==sha(folder/'raw/result.json')
        assert sha(folder/'demo.mp4')==row['video_sha256']==sha(ROOT/row['source_video'])
        for name in ('trajectory.json','queries.json','parking.json'):
            assert sha(folder/'raw'/name)==sha(source.parent/name);source_files+=1
        assert (folder/'keyframes.jpg').stat().st_size>1000
        status='success' if result['success'] else 'failure'
        counts.setdefault(row['skill'],dict(success=0,failure=0))[status]+=1
    assert counts==manifest['requested'] and len(seen)==30
    for skill,group in original['per_class'].items():
        results=[json.loads(Path(p).read_text()) for p in group['source_results']]
        assert len(results)==group['attempts'] and sum(r['success'] for r in results)==group['successes']
    report=dict(status='PASS',selected_cases=30,counts=counts,all_original_trials=55,
                source_evidence_verified=True,videos_byte_identical=True,other_raw_logs_verified=source_files,
                evaluation_success_rates_changed=False,scene_generalization_claimed=False)
    (directory/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

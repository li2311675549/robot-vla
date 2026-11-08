#!/usr/bin/env python3
"""Read-only contact-geometry comparison of retained paired physical trials."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve contact analyses')
    rows=[]
    for folder in sorted(a.source.glob('*candidate*')):
        r=json.loads((folder/'result.json').read_text())
        with gzip.open(folder/'payload_traces.json.gz','rt') as f:segments=json.load(f)
        sample=segments[0]['samples'][0];normals={};ratios={}
        for side in ('left','right'):
            contacts=[c for c in sample['finger_contacts'] if c['body'].startswith('grip_'+side)]
            forces=np.asarray([c['normal_force_n'] for c in contacts]);n=np.asarray([c['normal_world'] for c in contacts])
            mean=(forces[:,None]*n).sum(axis=0)/forces.sum();normals[side]=mean/np.linalg.norm(mean)
            ratios[side]=max(c['tangential_force_n']/max(c['normal_force_n'],1e-9) for c in contacts)
        rows.append({'candidate_index':r['candidate_index'],'physical_success':r['success'],
            'first_transport_time':sample['time'],'antipodality_cosine':float(-normals['left']@normals['right']),
            'object_position_tool_m':sample['object_position_tool_m'],
            'normal_forces_n':[sample['left_normal_force_n'],sample['right_normal_force_n']],
            'maximum_tangential_to_normal_ratio':ratios,'ratio_is_not_mu_normalized_utilization':True,
            'source_result':str(folder/'result.json'),'error':r.get('error')})
    report={'scope':'Actual same-initial-state physical candidate trials; contact geometry is explanatory, not a learned policy or a formal success rate',
        'rows':rows,'finding':'Some nearly opposing normals still fail dynamically; static force or antipodality alone does not certify full transport',
        'no_physics_or_source_results_modified':True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Versioned heldout class/bin-slot permutation, no training or rollouts.

Each complete physical bin, its colored surfaces and label move together.
Object sizes, seeded fruit positions, mass and contact parameters stay fixed.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from _bootstrap import ROOT
from build_tasks import build


def build_ood(seed,slots,output):
    if not 2400001<=seed<=2400012:raise ValueError('OOD-only reserved scene seeds required')
    if sorted(slots)!=[0,1,2] or list(slots)==[0,1,2]:raise ValueError('Nonidentity three-slot permutation required')
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve OOD scene versions')
    output.mkdir(parents=True)
    original,metadata=build('fruit',seed,output/'reference.xml',wrist_camera=True,
        bin_label_profile='upright_v2',layout_profile='bounded_scatter_v3',fruit_count=10)
    tree=ET.parse(original);ys=[.85,1.35,1.85]
    for category,slot in zip(('apple','banana','orange'),slots):
        body=tree.getroot().find(f"worldbody/body[@name='bin_{category}']")
        pos=list(map(float,body.get('pos').split()));pos[1]=ys[slot]
        body.set('pos',' '.join(map(str,pos)));metadata['bins'][category]['center'][1]=ys[slot]
    scene=output/'scene.xml';tree.write(scene)
    before,after=mujoco.MjModel.from_xml_path(str(original)),mujoco.MjModel.from_xml_path(str(scene))
    for field in ('qpos0','key_qpos','body_mass','body_inertia','geom_size','geom_pos','geom_quat',
                  'geom_friction','geom_contype','geom_conaffinity','actuator_forcerange'):
        np.testing.assert_array_equal(getattr(before,field),getattr(after,field))
    changed=np.flatnonzero(np.any(before.body_pos!=after.body_pos,axis=1))
    if not set(after.body(int(i)).name for i in changed)<=set('bin_'+c for c in ('apple','banana','orange')):
        raise ValueError('Unexpected assembly movement')
    metadata.update(split='ood',training_eligible=False,ood_protocol='heldout_bin_slot_permutation_v1',
        bin_class_slots=dict(zip(('apple','banana','orange'),slots)),
        scene_protocol=metadata['scene_protocol']+'_bin_slot_permutation_ood_v1')
    (output/'scene.json').write_text(json.dumps(metadata,indent=2))
    report=dict(status='STATIC_COMPILE_AND_PARAMETER_PRESERVATION_PASS',seed=seed,slots=list(slots),
        scene=str(scene.resolve()),split='ood',training_eligible=False,
        baseline_train_bins='apple0/banana1/orange2',
        shift='Unseen class-to-physical-bin-slot mapping; full labeled bin assemblies permuted',
        rollout_performed=False,physical_success=None,
        note='Static preparation only. Must additionally validate passive initialization/clearance and freeze a learned-policy evaluation. Never recycle these scenes into training.')
    (output/'protocol.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--slots',nargs=3,type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build_ood(a.seed,a.slots,a.output),indent=2))

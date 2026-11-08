import importlib.util
import sys

import mujoco
import numpy as np
import pytest

from robot_exam.simulation import ROOT


def test_ood_bin_permutation_moves_labels_with_physical_bins(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('ood_builder',ROOT/'scripts/build_multifruit_ood_scene.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for seed in [2100001,2200001,2300001]:
        with pytest.raises(ValueError,match='OOD-only'):module.build_ood(seed,[1,2,0],tmp_path/str(seed))
    out=tmp_path/'ood';r=module.build_ood(2400001,[1,2,0],out)
    model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    for category,y in zip(('apple','banana','orange'),[1.35,1.85,.85]):
        body=model.body('bin_'+category).id
        assert data.xpos[body,1]==pytest.approx(y)
        label=model.geom('bin_label_'+category).id
        assert data.geom_xpos[label,1]==pytest.approx(y)
    assert r['training_eligible'] is False and r['rollout_performed'] is False

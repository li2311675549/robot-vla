"""A target-only success must not silently qualify a disturbed scene."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from robot_exam.simulation import ROOT


def test_strict_manifest_rejects_missing_or_failed_scene_witness(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('pilot_manifest',ROOT/'scripts/prepare_multifruit_pilot_manifest.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parent=tmp_path/'parent';parent.mkdir()
    for index,witness in enumerate((None,{'success':False})):
        result={'status':'PHYSICAL_PARENT_PASS_PENDING_REPLAY_AND_EXPORT',
                'collision_events':[],'data_warning_counts':[0]*8}
        if witness is not None:result['scene_preservation']=witness
        (parent/'result.json').write_text(json.dumps(result))
        with pytest.raises(ValueError,match='preservation of non-target'):
            module.prepare([parent],tmp_path/f'manifest_{index}',True,require_scene_preservation=True)

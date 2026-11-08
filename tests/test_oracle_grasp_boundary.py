from types import SimpleNamespace
import pytest
from robot_exam.tasks.manipulation import pick


def test_scene_collision_truth_requires_explicit_oracle_provenance():
    runtime=SimpleNamespace(config={'oracle_grasp_clutter_clearance':True},transition=lambda *args,**kwargs:None)
    with pytest.raises(ValueError,match='declared Oracle'):
        pick(runtime,'fruit_0',{'category':'apple','center':[.5,0,.75]})

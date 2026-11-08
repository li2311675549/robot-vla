"""Real server export style: comments, dependent paths and no shell execution."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('next_gpu_plan',Path(__file__).resolve().parents[1]/'scripts/plan_next_gpu_fruit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def exports(root):
    return f'''# User's checkpoint is read-only; don't source this comment.
export ROBOT_EXAM_BUNDLE="{root}"
export ROBOT_EXAM_OPENPI_ROOT="$ROBOT_EXAM_BUNDLE/openpi"
export ROBOT_EXAM_PYTHON="${{ROBOT_EXAM_OPENPI_ROOT}}/.venv/bin/python"
export OPENPI_DATA_HOME="$ROBOT_EXAM_BUNDLE/cache/openpi"
export ROBOT_EXAM_WEIGHTS="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params"
export ROBOT_EXAM_TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
'''


def test_resolve_actual_export_dependencies_and_quoted_comments(tmp_path):
    p=tmp_path/'paths.sh';p.write_text(exports(tmp_path))
    values=module.read_model_paths(p)
    assert values['ROBOT_EXAM_PYTHON']==str(tmp_path/'openpi/.venv/bin/python')
    assert values['ROBOT_EXAM_WEIGHTS']==str(tmp_path/'cache/openpi/openpi-assets/checkpoints/pi05_base/params')


def test_no_command_substitution_or_environment_secret_expansion(tmp_path):
    p=tmp_path/'paths.sh';marker=tmp_path/'must_not_execute'
    p.write_text(exports(tmp_path).replace('$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params',f'$(touch {marker})'))
    with pytest.raises(ValueError,match='shell syntax'):module.read_model_paths(p)
    assert not marker.exists()


def test_reject_recursive_path_bindings(tmp_path):
    p=tmp_path/'paths.sh';p.write_text(exports(tmp_path).replace(f'ROBOT_EXAM_BUNDLE="{tmp_path}"','ROBOT_EXAM_BUNDLE="$ROBOT_EXAM_OPENPI_ROOT"'))
    with pytest.raises(ValueError,match='cyclic'):module.read_model_paths(p)

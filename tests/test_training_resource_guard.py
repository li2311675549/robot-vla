"""Avoid killing CUDA jobs solely because process RSS exceeds container RAM."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'resource_guard', Path(__file__).resolve().parents[1] / 'scripts/run_resource_guarded.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_container_accounting_does_not_treat_large_cuda_rss_as_cpu_ram(tmp_path):
    counter = tmp_path / 'memory.current'
    counter.write_text(str(30 * 1024**3))
    rss = 145 * 1024**3
    assert guard.accounted_bytes('cgroup', rss, counter) == 30 * 1024**3
    assert guard.accounted_bytes('rss', rss, counter) == rss
    counter.write_text(str(85 * 1024**3))
    assert guard.accounted_bytes('cgroup', rss, counter) > 80 * 1024**3


def test_invalid_or_unavailable_kernel_counter_refuses_unprotected_training(tmp_path):
    counter = tmp_path / 'memory.current'
    with pytest.raises(FileNotFoundError):
        guard.accounted_bytes('cgroup', 0, counter)
    for value in ('max', '-1'):
        counter.write_text(value)
        with pytest.raises(ValueError):
            guard.accounted_bytes('cgroup', 0, counter)

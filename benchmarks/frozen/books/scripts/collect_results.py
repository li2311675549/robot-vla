#!/usr/bin/env python3
"""Preserve actual acceptance evidence (small JSON) in the reviewable Git tree."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import xml.etree.ElementTree as ET

from _bootstrap import ROOT
import mujoco


def load_pass(path):
    result = json.loads(path.read_text())
    if result["status"] != "PASS":
        raise RuntimeError(f"Nonpassing result: {path}")
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-viewer", action="store_true")
    args = parser.parse_args()
    junit = ET.parse(ROOT / "outputs/acceptance/junit.xml").getroot()
    suites = list(junit.iter("testsuite"))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    assert counts["tests"] >= 19
    assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
    evidence = {}
    for scene in ("minimal", "ur5", "robotiq"):
        for kind, path in (("physics", ROOT / f"outputs/smoke/{scene}.json"),
                           ("render", ROOT / f"outputs/render/{scene}/result.json")):
            evidence[f"{scene}_{kind}"] = load_pass(path)
    viewer = ROOT / "outputs/viewer/ur5_process.json"
    if args.require_viewer or viewer.exists():
        evidence["ur5_gui_process"] = load_pass(viewer)
        assert evidence["ur5_gui_process"]["returncode"] == 0
    device_render = ROOT / "outputs/render_device_access/ur5/result.json"
    if device_render.exists():
        evidence["ur5_render_device_access"] = load_pass(device_render)
    assets = [p for folder in ("src", "scripts", "tests", "scenes", "assets", "configs", "third_party")
              for p in (ROOT / folder).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts]
    assets.extend(ROOT / name for name in ("requirements.in", "requirements.lock.txt", "environment.yml", "pyproject.toml"))
    files = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(assets)}
    artifacts = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for folder in (ROOT / "outputs/render", ROOT / "outputs/render_device_access")
                 for p in sorted(folder.rglob("*")) if p.suffix in (".png", ".mp4", ".npy", ".csv")}
    du = subprocess.run(["du", "-sk", ".venv", "assets", "third_party", "outputs"],
                        cwd=ROOT, check=True, text=True, capture_output=True).stdout
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    result = {"status": "PASS", "scope": "Stage 1: standalone simulation only",
              "timestamp": datetime.datetime.now().astimezone().isoformat(),
              "python": platform.python_version(), "mujoco": mujoco.__version__,
              "git_commit": git.stdout.strip() if git.returncode == 0 else None,
              "note": "No initial Git commit; file SHA256 values identify tested implementation.",
              "pytest": counts, "evidence": evidence, "artifacts": artifacts,
              "disk_usage_kib": du, "disk_free_bytes": shutil.disk_usage(ROOT).free,
              "implementation_sha256": files,
              "unfinished": ["Husky MuJoCo conversion and wheel dynamics", "UR5 IK and complete collision validation",
                             "UR5+2F85 mounting and physical lifting", "D435i body/intrinsics/IMU fidelity",
                             "mobile manipulator integration", "three formal tasks and training"]}
    output = ROOT / "docs/results/acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"PASS: {counts['tests']} tests; evidence saved to {output}")


if __name__ == "__main__":
    main()

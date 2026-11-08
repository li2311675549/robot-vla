#!/usr/bin/env python3
"""Archive observed Gate results, failures, versions and artifact hashes."""
import datetime
import hashlib
import json
import platform
import shutil
import subprocess
import xml.etree.ElementTree as ET

from _bootstrap import ROOT
import mujoco


def read_pass(path):
    value = json.loads(path.read_text())
    if value["status"] != "PASS":
        raise RuntimeError(f"Gate failed: {path}")
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / "outputs/stage2"
    junit = ET.parse(output / "acceptance/junit.xml").getroot()
    suites = list(junit.iter("testsuite"))
    counts = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if counts["tests"] < 39 or any(counts[key] for key in ("failures", "errors", "skipped")):
        raise RuntimeError(f"Incomplete/nonpassing test suite: {counts}")
    gates = {"gate1": read_pass(output / "gate1/result.json"),
             "gate2": {variant: read_pass(output / f"gate2/{variant}/result.json") for variant in ("box", "small", "wide", "cylinder")},
             "gate3": read_pass(output / "gate3/result.json"),
             "gate4": read_pass(output / "gate4/result.json")}
    videos = {str(p.relative_to(ROOT)): read_pass(p) for p in output.rglob("video_result.json")}
    if "outputs/stage2/gate4/video/video_result.json" not in videos:
        # Physics can pass without a renderer; do not claim full deliverables then.
        raise RuntimeError("Run verify_stage2.sh --record at least once to provide continuous demonstration")
    failures = {str(p.relative_to(ROOT)): json.loads(p.read_text()) for p in output.rglob("attempt*_failure.json")}
    implementation = {}
    for folder in ("src", "scripts", "tests", "scenes", "configs", "assets", "third_party"):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                implementation[str(p.relative_to(ROOT))] = digest(p)
    for name in ("pyproject.toml", "requirements.in", "requirements.lock.txt", "environment.yml", "AGENTS.md", "README.md"):
        implementation[name] = digest(ROOT / name)
    artifacts = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                 for p in sorted(output.rglob("*")) if p.suffix in (".mp4", ".png", ".npy", ".csv", ".json", ".xml", ".log")}
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scenes/mobile_manipulator.xml"))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    du = subprocess.run(["du", "-sk", ".venv", "outputs/stage2", "scenes"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    result = {"status": "PASS", "scope": "Stage 2: basic integrated physical manipulation; no formal task evaluation",
              "timestamp": datetime.datetime.now().astimezone().isoformat(), "python": platform.python_version(),
              "mujoco": mujoco.__version__, "git_commit": git.stdout.strip() if git.returncode == 0 else None,
              "pytest": counts, "gates": gates, "videos": videos, "development_failures": failures,
              "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "neq": model.neq, "nmocap": model.nmocap,
                        "robot_subtree_mass_kg": float(model.body_subtreemass[model.body('husky').id]),
                        "multiccd": bool(model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_MULTICCD))},
              "new_dependencies": [], "new_downloads_bytes": 0, "disk_usage_kib": du,
              "disk_free_bytes": shutil.disk_usage(ROOT).free,
              "implementation_sha256": implementation, "artifacts": artifacts,
              "limitations": ["source-derived primitive Husky without identified tire/drive parameters",
                              "ideal virtual RGB-D without stereo/IR/IMU/noise simulation",
                              "red upright fixture detector assumes known table height and visible top surface",
                              "base feedback and camera extrinsics use simulated robot state; object truth for scoring only",
                              "local DLS IK and simple waypoint execution; no general collision-avoiding planner",
                              "deterministic engineering fixtures; no formal 10-trial task/generalization success claims"]}
    path = ROOT / "docs/results/stage2_acceptance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"PASS: four Gates, {counts['tests']} tests; {path}")


if __name__ == "__main__":
    main()

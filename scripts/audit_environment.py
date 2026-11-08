#!/usr/bin/env python3
"""Read-only machine/project audit; excludes credentials and full environment dumps."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {"command": args, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "error": str(exc)}


def main():
    checks = [["uname", "-a"], ["git", "--version"], ["nvidia-smi"],
              ["df", "-B1", str(ROOT)], ["du", "-sk", str(ROOT / ".venv")],
              [str(Path.home() / "anaconda3/bin/conda"), "env", "list", "--json"]]
    for relative in ("vla/openpi", "vla/LIBERO"):
        p = Path.home() / relative
        checks.extend([["git", "-C", str(p), "rev-parse", "HEAD"],
                       ["git", "-C", str(p), "status", "--short", "--branch"]])
    for python in (Path.home() / "anaconda3/bin/python",
                   Path.home() / "anaconda3/envs/lab/bin/python",
                   Path.home() / "anaconda3/envs/openpi/bin/python",
                   Path.home() / "vla/openpi/.venv/bin/python", ROOT / ".venv/bin/python"):
        if python.exists():
            checks.append([str(python), "-c", "import sys,importlib.metadata as m; print(sys.version); print({p:m.version(p) for p in ('mujoco','numpy','pytest') if m.packages_distributions().get(p)})"])
    report = {"timestamp": datetime.datetime.now().astimezone().isoformat(),
              "os_release": Path("/etc/os-release").read_text(),
              "display": os.environ.get("DISPLAY"), "python": sys.version,
              "disk_free_bytes": shutil.disk_usage(ROOT).free,
              "checks": [command(args) for args in checks]}
    output = ROOT / "outputs/audit/environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()

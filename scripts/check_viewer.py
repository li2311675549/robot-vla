#!/usr/bin/env python3
"""Parent process verifies GUI child actually exits cleanly (including C teardown)."""
import argparse
import datetime
import json
import subprocess
import sys

from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["minimal", "ur5", "robotiq"], default="ur5")
    parser.add_argument("--seconds", type=float, default=3)
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / "scripts/viewer_demo.py"),
               "--scene", args.scene, "--seconds", str(args.seconds)]
    out = ROOT / "outputs/viewer"
    out.mkdir(parents=True, exist_ok=True)
    try:
        child = subprocess.run(command, capture_output=True, text=True, timeout=args.seconds + 25)
        result = {"status": "PASS" if child.returncode == 0 else "FAIL",
                  "timestamp": datetime.datetime.now().astimezone().isoformat(),
                  "returncode": child.returncode, "stdout": child.stdout, "stderr": child.stderr}
    except subprocess.TimeoutExpired as exc:
        result = {"status": "FAIL", "reason": f"Viewer timeout: {exc}"}
    (out / f"{args.scene}_process.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

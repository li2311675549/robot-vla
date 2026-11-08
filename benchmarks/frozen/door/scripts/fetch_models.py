#!/usr/bin/env python3
"""Fetch a bounded, commit-pinned subset. Existing differing files are never overwritten."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def blob_hash(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main():
    lock = json.loads((ROOT / "configs/model_sources.lock.json").read_text())
    total = sum(f["bytes"] for s in lock["sources"] for f in s["files"])
    if total > lock["max_total_bytes"]:
        raise RuntimeError("Model subset exceeds download budget")
    if shutil.disk_usage(ROOT).free < 25 * 1024**3:
        raise RuntimeError("Keep at least 25 GiB free")
    for source in lock["sources"]:
        for f in source["files"]:
            target = (ROOT / f["path"]).resolve()
            if not target.is_relative_to(ROOT):
                raise RuntimeError("Invalid target path")
            if target.exists():
                data = target.read_bytes()
                if len(data) != f["bytes"] or blob_hash(data) != f["git_blob"]:
                    raise RuntimeError(f"Existing file differs: {target}")
                continue
            url = f"https://raw.githubusercontent.com/{source['repo']}/{source['commit']}/{f['upstream_path']}"
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "robot-exam-stage1"})
                    with urllib.request.urlopen(req, timeout=30) as response:
                        data = response.read(f["bytes"] + 1)
                    if len(data) != f["bytes"] or blob_hash(data) != f["git_blob"]:
                        raise RuntimeError(f"Integrity mismatch: {url}")
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            print(f"FETCH {f['path']} ({len(data)} bytes)", flush=True)
    print(f"PASS: {total} bytes of pinned model sources")


if __name__ == "__main__":
    main()

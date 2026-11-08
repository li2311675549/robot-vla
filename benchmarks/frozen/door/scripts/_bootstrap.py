from pathlib import Path
import sys
import os

# Headless tasks must work without an X server. An explicit user setting wins.
os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

"""Development launcher for the Reflex OrderFlow-Agent interface."""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_app() -> None:
    os.environ.setdefault("REFLEX_DIR", str(ROOT / ".reflex"))
    # Runtime databases and generated evaluation artifacts must not restart the
    # Reflex backend while a customer is mid-conversation.
    os.environ.setdefault("REFLEX_HOT_RELOAD_EXCLUDE_PATHS", "data:evaluation")
    if find_spec("reflex") is None:
        raise RuntimeError("Reflex is not installed in this Python environment. Install requirements.txt first.")
    subprocess.run([sys.executable, "-m", "reflex", "run", *sys.argv[1:]], cwd=ROOT, check=True)


if __name__ in {"__main__", "__mp_main__"}:
    run_app()

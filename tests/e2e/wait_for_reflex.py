"""Wait for both halves of the Reflex development server."""

from __future__ import annotations

import sys
import time
from urllib.request import urlopen


def wait_for_reflex(frontend_url: str, backend_url: str) -> None:
    deadline = time.time() + 150
    frontend_ready = False
    backend_ready = False
    while time.time() < deadline:
        if not frontend_ready:
            try:
                with urlopen(frontend_url, timeout=2) as response:
                    frontend_ready = response.status == 200
            except OSError:
                pass
        if not backend_ready:
            try:
                with urlopen(f"{backend_url.rstrip('/')}/ping/", timeout=2) as response:
                    backend_ready = response.status == 200
            except OSError:
                pass
        if frontend_ready and backend_ready:
            return
        time.sleep(0.35)
    raise RuntimeError(f"Reflex did not become ready at {frontend_url} and {backend_url}")


if __name__ == "__main__":
    wait_for_reflex(sys.argv[1], sys.argv[2])

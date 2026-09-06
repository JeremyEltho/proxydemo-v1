"""Vercel serverless entrypoint.

Vercel serves this file as a Python function and detects the module-level
`app` as an ASGI application. There is no model runtime in the cloud, so it
boots in routing-only mode: /api/route works, generation returns a clear 503.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("ROUTER_OFFLINE", "1")

from app.main import app  # noqa: E402

__all__ = ["app"]

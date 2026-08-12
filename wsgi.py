"""Root WSGI entry point.

Railway may run the start command from the repo root (if `rootDirectory` is not
applied). This module lets `gunicorn wsgi:app` work from the repo root, while
backend/wsgi.py still works when the process runs from backend/.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.wsgi import app  # noqa: E402

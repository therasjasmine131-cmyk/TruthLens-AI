"""WSGI entry point for production servers.

Run with gunicorn from the backend/ directory:

    gunicorn -w 2 --timeout 120 -b 0.0.0.0:$PORT wsgi:app
"""

import sys
from pathlib import Path

# Make the repository root importable (e.g. the `ml` package), regardless of
# the process working directory. On Railway the app runs from backend/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import create_app

app = create_app()

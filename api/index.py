"""Vercel serverless entrypoint - serves the full TruthLens Flask app.

The whole API plus the built React SPA (backend/static) run behind a single
Vercel Python function; the SPA calls /api on the same origin.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app import create_app  # noqa: E402

app = create_app()
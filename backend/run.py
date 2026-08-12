"""Run the TruthLens AI backend locally.

    python backend/run.py          # http://localhost:5000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repository root importable (e.g. the `ml` package) when running
# `python backend/run.py` from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import create_app
from app.config import Config

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=Config.DEBUG and os.environ.get("FLASK_ENV", "").lower() != "production",
    )

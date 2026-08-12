"""Run the TruthLens AI backend locally.

    python backend/run.py          # http://localhost:5000
"""

from __future__ import annotations

import os

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

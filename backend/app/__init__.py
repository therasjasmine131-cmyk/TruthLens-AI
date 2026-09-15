"""TruthLens AI - Flask application factory."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

from .config import Config
from .extensions import cors, db
from .ml.model_manager import model_manager
from .utils.errors import register_error_handlers


def create_app(config_class=Config) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    Path(app.config.get("ML_ARTIFACTS_DIR")).mkdir(parents=True, exist_ok=True)
    # Read-only filesystems (e.g. Vercel functions) must not abort startup here.
    try:
        Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    db.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})

    from . import models  # noqa: F401  (register tables)
    from .routes import ai_detector, analytics, analyze, batch, dataset, health, history, model_perf, news, samples, settings

    for bp in (
        health.bp,
        analyze.bp,
        history.bp,
        analytics.bp,
        model_perf.bp,
        dataset.bp,
        batch.bp,
        samples.bp,
        settings.bp,
        ai_detector.bp,
        news.bp,
    ):
        app.register_blueprint(bp)

    register_error_handlers(app)

    @app.get("/api")
    def api_root():
        return jsonify(
            {
                "name": "TruthLens AI API",
                "version": "1.0.0",
                "endpoints": [
                    "POST /api/analyze",
                    "POST /api/analyze/headline",
                    "POST /api/batch/analyze",
                    "POST /api/detect-ai-text",
                    "GET  /api/news/trending",
                    "GET  /api/history",
                    "GET  /api/history/<id>",
                    "DELETE /api/history/<id>",
                    "POST /api/history/<id>/reanalyze",
                    "DELETE /api/history/clear",
                    "GET  /api/history/export",
                    "GET  /api/history/report/<id>",
                    "GET  /api/analytics",
                    "GET  /api/model-performance",
                    "GET  /api/dataset/stats",
                    "GET  /api/dataset/samples",
                    "GET  /api/samples",
                    "GET  /api/settings",
                    "PUT  /api/settings",
                    "GET  /api/health",
                ],
            }
        )

    with app.app_context():
        db.create_all()
        model_manager.load()

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def spa_fallback(path):
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
        target = os.path.join(static_dir, path)
        if path and os.path.isfile(target):
            return send_from_directory(static_dir, path)
        return send_from_directory(static_dir, "index.html")

    return app

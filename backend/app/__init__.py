"""TruthLens AI - Flask application factory."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify

from .config import Config
from .extensions import cors, db
from .ml.model_manager import model_manager
from .utils.errors import register_error_handlers


def create_app(config_class=Config) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    Path(app.config.get("ML_ARTIFACTS_DIR")).mkdir(parents=True, exist_ok=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})

    from . import models  # noqa: F401  (register tables)
    from .routes import analytics, analyze, batch, dataset, health, history, model_perf, samples, settings

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
        return jsonify({"error": "Use the /api/ endpoints.", "status": "error"}), 404

    return app

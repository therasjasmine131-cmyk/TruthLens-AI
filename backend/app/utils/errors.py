"""Shared error types and JSON error helpers."""

from __future__ import annotations

from flask import jsonify


class ApiError(Exception):
    """Base error carrying an HTTP status code."""

    status_code = 400

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class ValidationError(ApiError):
    status_code = 400


class NotFoundError(ApiError):
    status_code = 404


class ServiceUnavailableError(ApiError):
    status_code = 503


def register_error_handlers(app) -> None:
    @app.errorhandler(ApiError)
    def handle_api_error(err: ApiError):
        return jsonify({"error": err.message, "status": "error"}), err.status_code

    @app.errorhandler(ValidationError)
    def handle_validation_error(err: ValidationError):
        return jsonify({"error": err.message, "status": "error"}), err.status_code

    @app.errorhandler(404)
    def handle_404(err):  # noqa: ANN001
        return jsonify({"error": "Endpoint not found", "status": "error"}), 404

    @app.errorhandler(405)
    def handle_405(err):  # noqa: ANN001
        return jsonify({"error": "Method not allowed", "status": "error"}), 405

    @app.errorhandler(500)
    def handle_500(err):  # noqa: ANN001
        # Never leak stack traces to API clients.
        app.logger.exception("Unhandled error: %s", err)
        return jsonify({"error": "Internal server error", "status": "error"}), 500

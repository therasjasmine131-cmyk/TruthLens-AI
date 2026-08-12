"""WSGI entry point for production servers.

Run with gunicorn from the backend/ directory:

    gunicorn -w 2 --timeout 120 -b 0.0.0.0:$PORT wsgi:app
"""

from app import create_app

app = create_app()

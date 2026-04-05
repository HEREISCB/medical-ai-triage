"""Medical AI Triage server.

Single-port architecture:
- HTTP server on port 8000 serves the web UI
- Pipecat WebSocket server on port 8765 handles audio
- Frontend auto-detects the WS URL from WS_URL env var or same host

For cloudflared: set WS_URL env var to your second tunnel URL,
OR use the simple approach below that runs both on one process.
"""

import asyncio
import logging
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def start_http_server():
    """Serve static files on the configured port in a background thread."""
    os.chdir("static")

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.info("HTTP: %s", format % args)

    server = HTTPServer(("0.0.0.0", settings.app_port), Handler)
    logger.info("HTTP server on http://0.0.0.0:%d", settings.app_port)
    server.serve_forever()


async def main():
    # Start HTTP server for the web UI in a background thread
    http_thread = Thread(target=start_http_server, daemon=True)
    http_thread.start()

    # Start the Pipecat voice pipeline
    from src.pipeline.voice_pipeline import run_pipeline
    logger.info("Starting Pipecat pipeline on ws://0.0.0.0:8765")
    await run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())

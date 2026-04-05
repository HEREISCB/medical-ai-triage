"""FastAPI application for the medical triage system.

Provides:
- GET  /            -> Serves the web call UI
- WS   /ws          -> WebSocket endpoint for voice pipeline
- GET  /api/health  -> Health check
"""

import asyncio
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from src.config import settings
from src.pipeline.voice_pipeline import create_pipeline_components

from pipecat.pipeline.runner import PipelineRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical AI Triage",
    description="Voice-based AI medical triage system",
    version="0.1.0",
)

_active_sessions: dict[str, asyncio.Task] = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for the voice triage pipeline.

    The client connects here, sends audio frames, and receives audio back.
    """
    await websocket.accept()
    session_id = str(id(websocket))
    logger.info("New WebSocket connection: %s", session_id)

    transport, task = create_pipeline_components()

    runner = PipelineRunner(handle_sigint=False)

    try:
        await runner.run(task)
    except Exception as e:
        logger.error("Pipeline error for session %s: %s", session_id, e)
    finally:
        logger.info("Session %s ended", session_id)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the call UI page."""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Medical AI Triage</h1><p>Static files not found.</p>")

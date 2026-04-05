"""FastAPI application for the medical triage system.

Architecture:
- FastAPI serves the web UI on port 8000
- Pipecat runs its own WebSocket server on port 8765 for audio
- Client connects to both: HTTP for the page, WS for audio
"""

import asyncio
import logging

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.config import settings

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

_pipeline_task = None


async def start_pipeline():
    """Start the Pipecat voice pipeline on app startup."""
    global _pipeline_task
    from src.pipeline.voice_pipeline import run_pipeline
    _pipeline_task = asyncio.create_task(run_pipeline())
    logger.info("Pipecat pipeline started on ws://0.0.0.0:8765")


@app.on_event("startup")
async def on_startup():
    await start_pipeline()


@app.on_event("shutdown")
async def on_shutdown():
    global _pipeline_task
    if _pipeline_task:
        _pipeline_task.cancel()
        try:
            await _pipeline_task
        except asyncio.CancelledError:
            pass


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_running": _pipeline_task is not None and not _pipeline_task.done()}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the call UI page."""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Medical AI Triage</h1><p>Static files not found.</p>")

"""FastAPI application for the medical triage system.

Single port: serves web UI + handles WebSocket audio on port 8000.
One cloudflared tunnel is all you need.
"""

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.pipeline.voice_pipeline import TriageCall

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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice triage calls."""
    await websocket.accept()
    logger.info("New call connected")

    call = TriageCall(websocket)
    try:
        await call.run()
    except WebSocketDisconnect:
        logger.info("Call disconnected")
    except Exception as e:
        logger.error("Call error: %s", e)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Medical AI Triage</h1><p>Static files not found.</p>")

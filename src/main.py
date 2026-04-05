"""FastAPI app for serving the frontend and creating LiveKit tokens.

The LiveKit agent runs separately via run.py.
This serves the web UI and generates tokens for callers to join rooms.
"""

import logging

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from livekit.api import AccessToken, VideoGrants

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical AI Triage", version="0.1.0")


class TokenResponse(BaseModel):
    token: str
    url: str
    room: str


@app.post("/api/token")
async def create_token():
    """Create a LiveKit room and return a token for the caller to join."""
    import secrets

    room_name = f"triage-{secrets.token_hex(4)}"

    token = (
        AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(f"caller-{secrets.token_hex(4)}")
        .with_grants(VideoGrants(room_join=True, room=room_name))
    )

    return TokenResponse(
        token=token.to_jwt(),
        url=settings.livekit_url,
        room=room_name,
    )


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

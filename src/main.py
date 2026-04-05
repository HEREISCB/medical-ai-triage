"""FastAPI app — serves the web UI and creates LiveKit rooms with caller info."""

import json
import logging
import secrets

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from livekit.api import AccessToken, VideoGrants, LiveKitAPI
from pydantic import BaseModel

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical AI Triage", version="0.1.0")


class StartCallRequest(BaseModel):
    name: str
    phone: str
    email: str


class TokenResponse(BaseModel):
    token: str
    url: str
    room: str


@app.post("/api/start-call", response_model=TokenResponse)
async def start_call(req: StartCallRequest):
    """Create a LiveKit room with caller info in metadata, return a join token."""
    room_name = f"triage-{secrets.token_hex(4)}"

    # Store caller info in room metadata so the agent can read it
    metadata = json.dumps({
        "name": req.name,
        "phone": req.phone,
        "email": req.email,
    })

    # Create the room with metadata
    lk_api = LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )
    try:
        await lk_api.room.create_room(
            name=room_name,
            metadata=metadata,
        )
    except Exception as e:
        logger.warning("Room creation note: %s", e)
    finally:
        await lk_api.aclose()

    # Generate caller token
    token = (
        AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(f"caller-{secrets.token_hex(4)}")
        .with_grants(VideoGrants(room_join=True, room=room_name))
    )

    logger.info("Call started: room=%s caller=%s", room_name, req.name)

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

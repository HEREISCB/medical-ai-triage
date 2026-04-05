"""FastAPI application for the medical triage system.

Provides:
- POST /api/create-room  -> Creates a Daily room and returns a shareable link
- GET  /api/room/{room_name}/token -> Gets a token to join a room
- GET  /                  -> Serves the web call UI
- POST /api/start-bot     -> Starts the triage AI bot in a room
"""

import asyncio
import logging
import uuid

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import settings
from src.pipeline.voice_pipeline import run_pipeline

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

# Store active bot tasks
_active_bots: dict[str, asyncio.Task] = {}


class RoomResponse(BaseModel):
    room_url: str
    room_name: str
    shareable_link: str


class TokenResponse(BaseModel):
    token: str
    room_url: str


class BotStartResponse(BaseModel):
    status: str
    room_url: str
    room_name: str


async def _daily_api_request(method: str, endpoint: str, json_data: dict = None) -> dict:
    """Make a request to the Daily REST API."""
    url = f"{settings.daily_api_url}{endpoint}"
    headers = {
        "Authorization": f"Bearer {settings.daily_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, headers=headers, json=json_data, timeout=30)
        if response.status_code not in (200, 201):
            logger.error("Daily API error: %s %s", response.status_code, response.text)
            raise HTTPException(status_code=502, detail=f"Daily API error: {response.text}")
        return response.json()


@app.post("/api/create-room", response_model=RoomResponse)
async def create_room():
    """Create a new Daily room for a triage call and return shareable link."""
    room_name = f"triage-{uuid.uuid4().hex[:8]}"

    room_data = await _daily_api_request("POST", "/rooms", json_data={
        "name": room_name,
        "properties": {
            "exp": None,  # No expiration for MVP
            "enable_chat": False,
            "enable_screenshare": False,
            "start_video_off": True,
            "start_audio_off": False,
            "max_participants": 2,  # Caller + AI bot
        },
    })

    room_url = room_data["url"]

    return RoomResponse(
        room_url=room_url,
        room_name=room_name,
        shareable_link=f"{room_url}",
    )


@app.get("/api/room/{room_name}/token", response_model=TokenResponse)
async def get_room_token(room_name: str):
    """Get a meeting token for the caller to join the room."""
    token_data = await _daily_api_request("POST", "/meeting-tokens", json_data={
        "properties": {
            "room_name": room_name,
            "is_owner": False,
        },
    })

    room_url = f"https://{settings.daily_room_url.split('//')[1].split('/')[0]}/{room_name}"

    return TokenResponse(
        token=token_data["token"],
        room_url=room_url,
    )


@app.post("/api/start-bot", response_model=BotStartResponse)
async def start_bot(room_name: str = None):
    """Create a room and start the triage AI bot in it.

    This is the main endpoint: creates room, starts bot, returns link for caller.
    """
    # Create room
    room_response = await create_room()
    room_url = room_response.room_url
    rname = room_response.room_name

    # Get a bot token (owner privileges)
    token_data = await _daily_api_request("POST", "/meeting-tokens", json_data={
        "properties": {
            "room_name": rname,
            "is_owner": True,
        },
    })
    bot_token = token_data["token"]

    # Start the bot pipeline in background
    async def _run_bot():
        try:
            await run_pipeline(room_url, bot_token)
        except Exception as e:
            logger.error("Bot pipeline error: %s", e)
        finally:
            _active_bots.pop(rname, None)

    task = asyncio.create_task(_run_bot())
    _active_bots[rname] = task

    logger.info("Bot started in room %s", rname)

    return BotStartResponse(
        status="bot_started",
        room_url=room_url,
        room_name=rname,
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "active_bots": len(_active_bots)}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the call UI page."""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Medical AI Triage</h1><p>Static files not found.</p>")

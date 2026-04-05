"""Voice triage agent using LiveKit + Groq LLM.

The LLM drives the conversation like a real emergency dispatcher.
No rigid state machine — the LLM asks smart, contextual questions.
At the end, it calls a tool to send the triage report via webhook.
"""

import json
import logging
from typing import Annotated

import httpx
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.llm import ChatContext
from livekit.plugins import deepgram, silero, openai

from src.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an emergency medical triage dispatcher. You are the first point of contact when someone calls for medical help.

CALLER INFO (collected before the call):
- Name: {caller_name}
- Phone: {caller_phone}
- Email: {caller_email}

YOUR BEHAVIOR:
- You are calm, direct, and in control. Like a police dispatcher during a crisis.
- ALWAYS address the caller by their name ({caller_name}). Use it naturally throughout the conversation.
- DO NOT repeat the same acknowledgment phrase. Vary your responses: "Got it", "Right", "Okay {caller_name}", "Understood", or just go straight to the next question. NEVER say "Okay, I hear you" more than once.
- You ask ONE question at a time. Short. Clear.
- Your questions are LOGICAL based on what they tell you. If they say "my hand hurts", you ask "What happened to your hand?" then "Can you move your fingers?" — not "Can you swallow?"
- You sound like a real human dispatcher, not a robot reading a script.
- You are empathetic but efficient. No wasted words. Keep responses under 2 sentences.

CRITICAL RULES:
1. NEVER tell the caller what condition they might have. No "this sounds like a fracture." Just ask questions.
2. NEVER give medical advice or suggest medications.
3. Ask 4-8 questions to understand the situation, then wrap up.
4. If it sounds life-threatening (not breathing, heavy bleeding, unconscious, chest pain), tell them to call 999/112 IMMEDIATELY and give basic first-aid instructions (like "press on the wound with a cloth").
5. After you have enough information, call the end_triage tool with your findings.
6. Speak naturally. Use contractions. Be human.

QUESTION FLOW (adapt based on what they say):
1. "What's going on?" / Acknowledge what they said
2. "How did this happen?" (if injury)
3. Specific questions about their complaint (logical follow-ups)
4. "How bad is the pain, 1 to 10?"
5. "Is anything else going on that I should know about?"
6. Wrap up: "Alright, I've got everything I need. We're sending this to the medical team right now."

Then call end_triage with all the information."""


class TriageAgent(Agent):
    def __init__(self, caller_name: str, caller_phone: str, caller_email: str):
        prompt = SYSTEM_PROMPT.format(
            caller_name=caller_name,
            caller_phone=caller_phone,
            caller_email=caller_email,
        )
        super().__init__(
            instructions=prompt,
            llm=openai.LLM(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            ),
        )
        self.caller_name = caller_name
        self.caller_phone = caller_phone
        self.caller_email = caller_email

    async def on_enter(self):
        self.session.say(
            f"Hey {self.caller_name}, what's going on? Tell me what happened.",
            add_to_chat_ctx=True,
        )

    @function_tool()
    async def end_triage(
        self,
        severity: Annotated[str, "One of: critical, urgent, moderate, minor"],
        chief_complaint: Annotated[str, "Brief description of the main complaint"],
        findings: Annotated[str, "All medical findings from the conversation"],
        suspected_conditions: Annotated[str, "What this could be (NEVER told to caller)"],
        recommended_action: Annotated[str, "What medical team should do"],
        first_aid_given: Annotated[str, "Any first-aid instructions given to caller"],
    ):
        """Call this when you have enough information to send the triage report. This ends the call."""
        report = {
            "caller": {
                "name": self.caller_name,
                "phone": self.caller_phone,
                "email": self.caller_email,
            },
            "triage": {
                "severity": severity,
                "chief_complaint": chief_complaint,
                "findings": findings,
                "suspected_conditions": suspected_conditions,
                "recommended_action": recommended_action,
                "first_aid_given": first_aid_given,
            },
        }

        logger.info("TRIAGE REPORT: %s", json.dumps(report, indent=2))

        # Send webhook
        if settings.webhook_url:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        settings.webhook_url,
                        json=report,
                        timeout=10.0,
                    )
                    logger.info("Webhook sent: %s", resp.status_code)
            except Exception as e:
                logger.error("Webhook failed: %s", e)

        return f"Report sent. Tell {self.caller_name} that you've sent everything to the medical team and they'll receive a summary at {self.caller_email}. Say goodbye."


async def entrypoint(ctx: JobContext):
    """LiveKit agent entrypoint."""
    await ctx.connect()

    # Wait for the caller to join so we can read their metadata
    caller_info = {}

    # Check existing participants first
    for participant in ctx.room.remote_participants.values():
        if participant.metadata:
            try:
                caller_info = json.loads(participant.metadata)
                break
            except json.JSONDecodeError:
                pass

    # If no metadata yet, wait for a participant to join
    if not caller_info:
        import asyncio
        from livekit import rtc

        future = asyncio.get_event_loop().create_future()

        def on_participant_connected(participant: rtc.RemoteParticipant):
            if participant.metadata and not future.done():
                try:
                    future.set_result(json.loads(participant.metadata))
                except json.JSONDecodeError:
                    pass

        def on_metadata_changed(participant: rtc.Participant, old_metadata: str, new_metadata: str):
            if new_metadata and not future.done():
                try:
                    future.set_result(json.loads(new_metadata))
                except json.JSONDecodeError:
                    pass

        ctx.room.on("participant_connected", on_participant_connected)
        ctx.room.on("participant_metadata_changed", on_metadata_changed)

        # Also re-check current participants (might have joined during setup)
        for participant in ctx.room.remote_participants.values():
            if participant.metadata:
                try:
                    caller_info = json.loads(participant.metadata)
                    break
                except json.JSONDecodeError:
                    pass

        if not caller_info:
            try:
                caller_info = await asyncio.wait_for(future, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for caller metadata")

    caller_name = caller_info.get("name", "there")
    caller_phone = caller_info.get("phone", "")
    caller_email = caller_info.get("email", "")

    session = AgentSession(
        stt=deepgram.STT(api_key=settings.deepgram_api_key),
        tts=deepgram.TTS(api_key=settings.deepgram_api_key, model="aura-asteria-en"),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=TriageAgent(caller_name, caller_phone, caller_email),
        room=ctx.room,
    )


def run_agent():
    """Run the LiveKit agent worker."""
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        ),
    )

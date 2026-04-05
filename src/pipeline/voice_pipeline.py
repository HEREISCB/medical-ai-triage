"""Voice triage pipeline using FastAPI WebSocket + Deepgram APIs directly.

Single-port: FastAPI handles both HTTP and WebSocket on port 8000.
Uses Deepgram APIs directly (no SDK version issues).

Flow:
  Client mic -> WebSocket -> Deepgram STT -> Triage Logic -> Deepgram TTS -> WebSocket -> Client speaker
"""

import asyncio
import json
import logging
import uuid

import httpx
import websockets

from fastapi import WebSocket

from src.config import settings
from src.nlu.extractor import extract_structured_data, classify_complaint
from src.safety.guardrails import check_danger_keywords, sanitize_response, should_escalate
from src.safety.disclaimers import ESCALATION_TEXT, CALL_END_TEXT, NO_CONSENT_TEXT
from src.triage.state_machine import TriageSession, TriageStateMachine
from src.triage.states import TriageState
from src.triage.questions import get_current_question, is_protocol_complete, QUESTIONS
from src.triage.pre_arrival import get_pre_arrival_instructions

logger = logging.getLogger(__name__)

PROTOCOL_STATES = (
    TriageState.MALARIA_PROTOCOL,
    TriageState.TRAUMA_PROTOCOL,
    TriageState.MATERNAL_PROTOCOL,
    TriageState.RESPIRATORY_PROTOCOL,
    TriageState.SNAKEBITE_PROTOCOL,
    TriageState.GENERAL_PROTOCOL,
)

DEEPGRAM_STT_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2&language=en&encoding=linear16&sample_rate=16000"
    "&channels=1&interim_results=false&utterance_end_ms=1500&vad_events=true&endpointing=300"
)

DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=16000"


class TriageCall:
    """Manages a single triage call session."""

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.session = TriageSession(session_id=str(uuid.uuid4()))
        self.machine = TriageStateMachine(self.session)
        self.low_confidence_streak = 0
        self._dg_ws = None
        self._processing = False

    async def run(self):
        """Run the full triage call."""
        logger.info("Call started: %s", self.session.session_id)

        # Send greeting via TTS
        greeting = get_current_question(TriageState.GREETING, 0, {"caller_name": ""})
        if greeting:
            await self._speak(greeting)

        # Connect to Deepgram STT WebSocket
        headers = {"Authorization": f"Token {settings.deepgram_api_key}"}

        try:
            async with websockets.connect(DEEPGRAM_STT_URL, extra_headers=headers) as dg_ws:
                self._dg_ws = dg_ws
                logger.info("Deepgram STT connected")

                # Run two tasks: forward audio to Deepgram, and listen for transcripts
                await asyncio.gather(
                    self._forward_audio_to_deepgram(dg_ws),
                    self._listen_for_transcripts(dg_ws),
                )

        except Exception as e:
            logger.error("Deepgram connection error: %s", e)
        finally:
            report = self.machine.get_triage_report()
            logger.info("CALL ENDED. Triage report: %s", json.dumps(report, indent=2))

    async def _forward_audio_to_deepgram(self, dg_ws):
        """Forward audio from the client WebSocket to Deepgram STT."""
        try:
            while True:
                data = await self.ws.receive_bytes()
                await dg_ws.send(data)
        except Exception as e:
            logger.info("Client audio stream ended: %s", type(e).__name__)
            # Send close signal to Deepgram
            try:
                await dg_ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass

    async def _listen_for_transcripts(self, dg_ws):
        """Listen for transcription results from Deepgram."""
        try:
            async for message in dg_ws:
                data = json.loads(message)

                # Only process final transcripts
                if data.get("type") == "Results" and data.get("is_final"):
                    transcript = (
                        data.get("channel", {})
                        .get("alternatives", [{}])[0]
                        .get("transcript", "")
                        .strip()
                    )
                    if transcript:
                        logger.info("Caller said: '%s' (state: %s)", transcript, self.session.state.value)
                        if not self._processing:
                            self._processing = True
                            try:
                                await self._process_caller_input(transcript)
                            finally:
                                self._processing = False

        except websockets.exceptions.ConnectionClosed:
            logger.info("Deepgram STT connection closed")
        except Exception as e:
            logger.error("Transcript listener error: %s", e)

    async def _process_caller_input(self, caller_text: str):
        """Process what the caller said through the triage pipeline."""
        # Check danger keywords
        danger_matches = check_danger_keywords(caller_text)
        if danger_matches:
            logger.warning("Danger keywords: %s", danger_matches)

        # Check escalation
        escalate, reason = should_escalate(
            self.session.turn_count,
            self.low_confidence_streak,
            caller_text,
            self.session.severity.value,
        )
        if escalate:
            logger.warning("Escalating: %s", reason)
            self.session.internal_notes.append(f"Escalated: {reason}")
            await self._speak(ESCALATION_TEXT)
            self.session.state = TriageState.CALL_END
            report = self.machine.get_triage_report()
            logger.info("TRIAGE REPORT: %s", json.dumps(report, indent=2))
            return

        # Get context for NLU
        question_context = self._get_current_question_text() or ""
        expected_fields = self._get_expected_fields()

        # NLU extraction via Groq
        nlu_result = await extract_structured_data(
            caller_text=caller_text,
            state=self.session.state,
            question_asked=question_context,
            expected_fields=expected_fields,
        )

        # Track confidence
        confidence = nlu_result.get("confidence", 0.8)
        if confidence < 0.6:
            self.low_confidence_streak += 1
        else:
            self.low_confidence_streak = 0

        # Chief complaint classification
        if self.session.state == TriageState.CHIEF_COMPLAINT:
            if not nlu_result.get("category"):
                category = await classify_complaint(caller_text)
                nlu_result["category"] = category
            nlu_result["complaint_text"] = caller_text

        # Protocol completion check
        if self.session.state in PROTOCOL_STATES:
            if is_protocol_complete(self.session.state, self.session.current_protocol_step + 1):
                nlu_result["protocol_complete"] = True

        # State machine transition
        new_state = self.machine.process_nlu_result(nlu_result)
        logger.info("State -> %s (severity: %s)", new_state.value, self.session.severity.value)

        # Generate and speak response
        response = self._get_response_for_state()
        if response:
            response = sanitize_response(response)
            await self._speak(response)

        # Check if done
        if self.session.state == TriageState.CALL_END:
            report = self.machine.get_triage_report()
            logger.info("TRIAGE COMPLETE: %s", json.dumps(report, indent=2))

    async def _speak(self, text: str):
        """Convert text to speech via Deepgram TTS REST API and send audio to client."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    DEEPGRAM_TTS_URL,
                    headers={
                        "Authorization": f"Token {settings.deepgram_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"text": text},
                    timeout=15.0,
                )

                if response.status_code == 200:
                    await self.ws.send_bytes(response.content)
                    logger.info("Spoke: %s", text[:80])
                else:
                    logger.error("TTS error: %s %s", response.status_code, response.text)

        except Exception as e:
            logger.error("TTS error: %s", e)
            # Fallback: send as text
            try:
                await self.ws.send_json({"type": "text", "message": text})
            except Exception:
                pass

    def _get_current_question_text(self) -> str | None:
        state = self.session.state
        step = self.session.current_protocol_step if state in PROTOCOL_STATES else 0
        return get_current_question(state, step, {"caller_name": self.session.caller_name})

    def _get_expected_fields(self) -> str:
        state = self.session.state
        questions = QUESTIONS.get(state, [])
        step = self.session.current_protocol_step if state in PROTOCOL_STATES else 0
        if step < len(questions):
            return questions[step].get("expect", "")
        return ""

    def _get_response_for_state(self) -> str | None:
        state = self.session.state

        if state == TriageState.CALL_END:
            return CALL_END_TEXT
        if state == TriageState.HUMAN_ESCALATION:
            return ESCALATION_TEXT
        if state == TriageState.PRE_ARRIVAL_INSTRUCTIONS:
            all_findings = {**self.session.danger_sign_findings, **self.session.protocol_findings}
            return get_pre_arrival_instructions(
                self.session.complaint_category, self.session.severity, all_findings,
            )

        step = self.session.current_protocol_step if state in PROTOCOL_STATES else 0
        question = get_current_question(state, step, {"caller_name": self.session.caller_name})
        if question is None and state == TriageState.CONSENT:
            return NO_CONSENT_TEXT
        return question

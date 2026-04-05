"""Voice triage pipeline using FastAPI WebSocket + Deepgram + Groq.

Single-port architecture: FastAPI handles both HTTP and WebSocket on port 8000.
No Pipecat transport needed — we manage the audio stream directly.

Flow:
  Client mic -> WebSocket -> Deepgram STT -> Triage Logic -> Deepgram TTS -> WebSocket -> Client speaker
"""

import asyncio
import json
import logging
import uuid

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)
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


class TriageCall:
    """Manages a single triage call session."""

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.session = TriageSession(session_id=str(uuid.uuid4()))
        self.machine = TriageStateMachine(self.session)
        self.low_confidence_streak = 0
        self._dg_client = None
        self._dg_connection = None
        self._processing = False

    async def run(self):
        """Run the full triage call."""
        logger.info("Call started: %s", self.session.session_id)

        # Send greeting via TTS
        greeting = get_current_question(TriageState.GREETING, 0, {"caller_name": ""})
        if greeting:
            await self._speak(greeting)

        # Set up Deepgram for live STT
        self._dg_client = DeepgramClient(settings.deepgram_api_key)
        self._dg_connection = self._dg_client.listen.asyncwebsocket.v("1")

        # Handle transcription results
        self._dg_connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        self._dg_connection.on(LiveTranscriptionEvents.Error, self._on_dg_error)

        options = LiveOptions(
            model="nova-2",
            language="en",
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=False,
            utterance_end_ms="1500",
            vad_events=True,
            endpointing=300,
        )

        if not await self._dg_connection.start(options):
            logger.error("Failed to connect to Deepgram STT")
            await self.ws.send_json({"type": "error", "message": "Failed to connect to speech service"})
            return

        logger.info("Deepgram STT connected")

        # Stream audio from client to Deepgram
        try:
            while True:
                data = await self.ws.receive_bytes()
                await self._dg_connection.send(data)
        except Exception as e:
            logger.info("Client disconnected: %s", e)
        finally:
            await self._dg_connection.finish()
            report = self.machine.get_triage_report()
            logger.info("CALL ENDED. Triage report: %s", json.dumps(report, indent=2))

    async def _on_transcript(self, connection, result, **kwargs):
        """Handle a transcription result from Deepgram."""
        try:
            transcript = result.channel.alternatives[0].transcript.strip()
            if not transcript or result.is_final is False:
                return

            logger.info("Caller said: '%s' (state: %s)", transcript, self.session.state.value)

            # Prevent re-entrant processing
            if self._processing:
                return
            self._processing = True

            try:
                await self._process_caller_input(transcript)
            finally:
                self._processing = False

        except Exception as e:
            logger.error("Error processing transcript: %s", e)

    async def _on_dg_error(self, connection, error, **kwargs):
        logger.error("Deepgram STT error: %s", error)

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
        """Convert text to speech via Deepgram TTS and send audio to client."""
        try:
            dg = DeepgramClient(settings.deepgram_api_key)
            options = {"model": "aura-asteria-en", "encoding": "linear16", "sample_rate": 16000}

            response = await dg.speak.asyncrest.v("1").stream_raw({"text": text}, options)

            # Send audio chunks to client
            audio_data = response.stream.read()
            if audio_data:
                # Send as binary WebSocket message
                await self.ws.send_bytes(audio_data)

            logger.info("Spoke: %s", text[:80])
        except Exception as e:
            logger.error("TTS error: %s", e)
            # Fallback: send text
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

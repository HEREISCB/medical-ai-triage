"""Voice triage agent using LiveKit Agents SDK.

LiveKit handles all the hard parts (WebRTC, audio, VAD, turn-taking).
We just plug in Deepgram STT/TTS and our triage logic.
"""

import json
import logging
import uuid

from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.stt import STT
from livekit.agents.tts import TTS
from livekit.plugins import deepgram, silero

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


class TriageAgent(Agent):
    """LiveKit Agent that runs the medical triage conversation."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a medical triage AI assistant. You ONLY ask questions to "
                "understand the emergency. You NEVER reveal medical conditions or "
                "diagnoses to the caller. You ask clear, simple questions and note "
                "findings internally."
            ),
        )
        self.triage_session = TriageSession(session_id=str(uuid.uuid4()))
        self.machine = TriageStateMachine(self.triage_session)
        self.low_confidence_streak = 0

    async def on_enter(self):
        """Called when the agent enters the session. Send greeting."""
        greeting = get_current_question(TriageState.GREETING, 0, {"caller_name": ""})
        if greeting:
            self.session.say(greeting)

    async def on_user_turn_completed(self, turn_ctx, message: ChatMessage):
        """Called when the user finishes speaking. Process through triage."""
        caller_text = message.text_content.strip()
        if not caller_text:
            return

        logger.info("Caller said: '%s' (state: %s)", caller_text, self.triage_session.state.value)

        # Check danger keywords
        danger_matches = check_danger_keywords(caller_text)
        if danger_matches:
            logger.warning("Danger keywords: %s", danger_matches)

        # Check escalation
        escalate, reason = should_escalate(
            self.triage_session.turn_count,
            self.low_confidence_streak,
            caller_text,
            self.triage_session.severity.value,
        )
        if escalate:
            logger.warning("Escalating: %s", reason)
            self.triage_session.internal_notes.append(f"Escalated: {reason}")
            turn_ctx.say(sanitize_response(ESCALATION_TEXT))
            self.triage_session.state = TriageState.CALL_END
            self._log_report()
            return

        # NLU extraction
        question_context = self._get_current_question_text() or ""
        expected_fields = self._get_expected_fields()

        nlu_result = await extract_structured_data(
            caller_text=caller_text,
            state=self.triage_session.state,
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
        if self.triage_session.state == TriageState.CHIEF_COMPLAINT:
            if not nlu_result.get("category"):
                category = await classify_complaint(caller_text)
                nlu_result["category"] = category
            nlu_result["complaint_text"] = caller_text

        # Protocol completion
        if self.triage_session.state in PROTOCOL_STATES:
            if is_protocol_complete(self.triage_session.state, self.triage_session.current_protocol_step + 1):
                nlu_result["protocol_complete"] = True

        # State machine transition
        new_state = self.machine.process_nlu_result(nlu_result)
        logger.info("State -> %s (severity: %s)", new_state.value, self.triage_session.severity.value)

        # Respond
        response = self._get_response_for_state()
        if response:
            response = sanitize_response(response)
            turn_ctx.say(response)

        if self.triage_session.state == TriageState.CALL_END:
            self._log_report()

    def _get_current_question_text(self) -> str | None:
        state = self.triage_session.state
        step = self.triage_session.current_protocol_step if state in PROTOCOL_STATES else 0
        return get_current_question(state, step, {"caller_name": self.triage_session.caller_name})

    def _get_expected_fields(self) -> str:
        state = self.triage_session.state
        questions = QUESTIONS.get(state, [])
        step = self.triage_session.current_protocol_step if state in PROTOCOL_STATES else 0
        if step < len(questions):
            return questions[step].get("expect", "")
        return ""

    def _get_response_for_state(self) -> str | None:
        state = self.triage_session.state
        if state == TriageState.CALL_END:
            return CALL_END_TEXT
        if state == TriageState.HUMAN_ESCALATION:
            return ESCALATION_TEXT
        if state == TriageState.PRE_ARRIVAL_INSTRUCTIONS:
            all_findings = {**self.triage_session.danger_sign_findings, **self.triage_session.protocol_findings}
            return get_pre_arrival_instructions(
                self.triage_session.complaint_category, self.triage_session.severity, all_findings,
            )
        step = self.triage_session.current_protocol_step if state in PROTOCOL_STATES else 0
        question = get_current_question(state, step, {"caller_name": self.triage_session.caller_name})
        if question is None and state == TriageState.CONSENT:
            return NO_CONSENT_TEXT
        return question

    def _log_report(self):
        report = self.machine.get_triage_report()
        logger.info("TRIAGE REPORT: %s", json.dumps(report, indent=2))


async def entrypoint(ctx: JobContext):
    """LiveKit agent entrypoint — called when a user joins a room."""
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(api_key=settings.deepgram_api_key),
        tts=deepgram.TTS(api_key=settings.deepgram_api_key, model="aura-asteria-en"),
        vad=silero.VAD.load(),
    )

    await session.start(agent=TriageAgent(), room=ctx.room)


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

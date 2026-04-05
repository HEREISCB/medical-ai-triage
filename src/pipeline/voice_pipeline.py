"""Main Pipecat voice pipeline for medical triage.

Orchestrates: Daily (transport) -> Deepgram STT -> Triage Logic -> Deepgram TTS

The triage logic is a custom Pipecat processor that:
1. Receives transcribed text from STT
2. Runs NLU extraction via Groq
3. Feeds structured data to the triage state machine
4. Gets the next question from the question bank
5. Optionally sanitizes through safety guardrails
6. Sends the question text to TTS
"""

import logging
import uuid

from pipecat.frames.frames import (
    EndFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from src.config import settings
from src.nlu.extractor import extract_structured_data, classify_complaint
from src.safety.guardrails import check_danger_keywords, sanitize_response, should_escalate
from src.safety.disclaimers import ESCALATION_TEXT, CALL_END_TEXT, NO_CONSENT_TEXT
from src.triage.state_machine import TriageSession, TriageStateMachine
from src.triage.states import TriageState, Severity
from src.triage.questions import get_current_question, is_protocol_complete, QUESTIONS
from src.triage.pre_arrival import get_pre_arrival_instructions

logger = logging.getLogger(__name__)


class TriageProcessor(FrameProcessor):
    """Custom Pipecat processor that handles the triage conversation flow.

    Sits between STT and TTS in the pipeline. Receives transcribed text,
    processes it through the triage state machine, and outputs the next
    question as text for TTS.
    """

    def __init__(self, session_id: str | None = None):
        super().__init__()
        sid = session_id or str(uuid.uuid4())
        self.session = TriageSession(session_id=sid)
        self.machine = TriageStateMachine(self.session)
        self.low_confidence_streak = 0
        self._greeting_sent = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # We got transcribed text from the caller
            caller_text = frame.text.strip()
            if not caller_text:
                await self.push_frame(frame, direction)
                return

            logger.info("Caller said: %s (state: %s)", caller_text, self.session.state.value)

            # Check for immediate danger keywords
            danger_matches = check_danger_keywords(caller_text)
            if danger_matches:
                logger.warning("Danger keywords detected: %s", danger_matches)

            # Check if we should escalate to human
            escalate, reason = should_escalate(
                self.session.turn_count,
                self.low_confidence_streak,
                caller_text,
                self.session.severity.value,
            )
            if escalate:
                logger.warning("Escalating call: %s", reason)
                self.session.state = TriageState.HUMAN_ESCALATION
                self.session.internal_notes.append(f"Escalated: {reason}")
                response = ESCALATION_TEXT
                await self.push_frame(TextFrame(text=response), FrameDirection.DOWNSTREAM)
                # Generate triage report
                report = self.machine.get_triage_report()
                logger.info("Triage report: %s", report)
                self.session.state = TriageState.CALL_END
                return

            # Get current question context for NLU
            question_context = self._get_current_question_text()
            expected_fields = self._get_expected_fields()

            # NLU extraction
            nlu_result = await extract_structured_data(
                caller_text=caller_text,
                state=self.session.state,
                question_asked=question_context or "",
                expected_fields=expected_fields,
            )

            # Track confidence
            confidence = nlu_result.get("confidence", 0.8)
            if confidence < 0.6:
                self.low_confidence_streak += 1
            else:
                self.low_confidence_streak = 0

            # Special handling for chief complaint classification
            if self.session.state == TriageState.CHIEF_COMPLAINT:
                if "category" not in nlu_result or not nlu_result.get("category"):
                    category = await classify_complaint(caller_text)
                    nlu_result["category"] = category
                nlu_result["complaint_text"] = caller_text

            # Handle protocol completion
            if self.session.state in (
                TriageState.MALARIA_PROTOCOL,
                TriageState.TRAUMA_PROTOCOL,
                TriageState.MATERNAL_PROTOCOL,
                TriageState.RESPIRATORY_PROTOCOL,
                TriageState.SNAKEBITE_PROTOCOL,
                TriageState.GENERAL_PROTOCOL,
            ):
                if is_protocol_complete(self.session.state, self.session.current_protocol_step + 1):
                    nlu_result["protocol_complete"] = True

            # Feed to state machine
            new_state = self.machine.process_nlu_result(nlu_result)
            logger.info("State transition -> %s (severity: %s)", new_state.value, self.session.severity.value)

            # Generate response based on new state
            response = self._get_response_for_state()
            if response:
                response = sanitize_response(response)
                await self.push_frame(TextFrame(text=response), FrameDirection.DOWNSTREAM)

            # Check if call is complete
            if self.session.state == TriageState.CALL_END:
                report = self.machine.get_triage_report()
                logger.info("TRIAGE COMPLETE. Report: %s", report)
                await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)

        else:
            await self.push_frame(frame, direction)

    async def send_greeting(self):
        """Send the initial greeting. Called when the call connects."""
        if self._greeting_sent:
            return
        self._greeting_sent = True
        greeting = get_current_question(
            TriageState.GREETING, 0, {"caller_name": ""}
        )
        if greeting:
            await self.push_frame(TextFrame(text=greeting), FrameDirection.DOWNSTREAM)

    def _get_current_question_text(self) -> str | None:
        """Get the text of the current question being asked."""
        state = self.session.state
        step = self.session.current_protocol_step if state in (
            TriageState.MALARIA_PROTOCOL,
            TriageState.TRAUMA_PROTOCOL,
            TriageState.MATERNAL_PROTOCOL,
            TriageState.RESPIRATORY_PROTOCOL,
            TriageState.SNAKEBITE_PROTOCOL,
            TriageState.GENERAL_PROTOCOL,
        ) else 0

        return get_current_question(
            state, step,
            {"caller_name": self.session.caller_name},
        )

    def _get_expected_fields(self) -> str:
        """Get the expected extraction fields for the current state."""
        state = self.session.state
        questions = QUESTIONS.get(state, [])
        step = self.session.current_protocol_step if state in (
            TriageState.MALARIA_PROTOCOL,
            TriageState.TRAUMA_PROTOCOL,
            TriageState.MATERNAL_PROTOCOL,
            TriageState.RESPIRATORY_PROTOCOL,
            TriageState.SNAKEBITE_PROTOCOL,
            TriageState.GENERAL_PROTOCOL,
        ) else 0

        if step < len(questions):
            return questions[step].get("expect", "")
        return ""

    def _get_response_for_state(self) -> str | None:
        """Get the appropriate response for the current state."""
        state = self.session.state

        if state == TriageState.CALL_END:
            return CALL_END_TEXT

        if state == TriageState.HUMAN_ESCALATION:
            return ESCALATION_TEXT

        if state == TriageState.PRE_ARRIVAL_INSTRUCTIONS:
            all_findings = {
                **self.session.danger_sign_findings,
                **self.session.protocol_findings,
            }
            instructions = get_pre_arrival_instructions(
                self.session.complaint_category,
                self.session.severity,
                all_findings,
            )
            return instructions

        # For all other states, get the next question
        step = 0
        if state in (
            TriageState.MALARIA_PROTOCOL,
            TriageState.TRAUMA_PROTOCOL,
            TriageState.MATERNAL_PROTOCOL,
            TriageState.RESPIRATORY_PROTOCOL,
            TriageState.SNAKEBITE_PROTOCOL,
            TriageState.GENERAL_PROTOCOL,
        ):
            step = self.session.current_protocol_step

        question = get_current_question(
            state, step,
            {"caller_name": self.session.caller_name},
        )

        if question is None and state == TriageState.CONSENT:
            return NO_CONSENT_TEXT

        return question


async def create_pipeline(room_url: str, token: str) -> PipelineTask:
    """Create and configure the full voice triage pipeline.

    Args:
        room_url: Daily room URL.
        token: Daily room token.

    Returns:
        Configured PipelineTask ready to run.
    """
    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Triage AI",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=None,  # Use Pipecat's default Silero VAD
            transcription_enabled=False,  # We use our own STT
        ),
    )

    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        model="nova-2",
        language="en",
    )

    tts = DeepgramTTSService(
        api_key=settings.deepgram_api_key,
        voice="aura-asteria-en",  # Calm, clear female voice
    )

    triage = TriageProcessor()

    pipeline = Pipeline([
        transport.input(),
        stt,
        triage,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    # Send greeting when participant joins
    @transport.event_handler("on_first_participant_joined")
    async def on_joined(transport, participant):
        logger.info("Caller joined: %s", participant.get("id", "unknown"))
        await triage.send_greeting()

    @transport.event_handler("on_participant_left")
    async def on_left(transport, participant, reason):
        logger.info("Caller left: %s (reason: %s)", participant.get("id", "unknown"), reason)
        report = triage.machine.get_triage_report()
        logger.info("Final triage report: %s", report)
        await task.queue_frame(EndFrame())

    return task


async def run_pipeline(room_url: str, token: str):
    """Run the voice triage pipeline."""
    runner = PipelineRunner()
    task = await create_pipeline(room_url, token)
    await runner.run(task)

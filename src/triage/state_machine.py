"""Core triage state machine.

This is a deterministic FSM that governs the entire triage conversation.
The LLM is NEVER used for triage decisions — only for understanding caller
speech (NLU) and generating natural responses (NLG).

CRITICAL RULE: The AI never reveals medical conditions to the caller.
It only asks questions and notes findings internally.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.triage.states import (
    ComplaintCategory,
    COMPLAINT_TO_PROTOCOL,
    Severity,
    TriageState,
)
from src.triage.severity import classify_danger_signs, PROTOCOL_CLASSIFIERS


@dataclass
class TriageSession:
    """Tracks the full state of a triage conversation."""

    session_id: str
    state: TriageState = TriageState.GREETING
    severity: Severity = Severity.PENDING
    complaint_category: ComplaintCategory | None = None

    # Internal findings - NEVER shown to caller
    danger_sign_findings: dict = field(default_factory=dict)
    protocol_findings: dict = field(default_factory=dict)
    chief_complaint_raw: str = ""
    internal_notes: list[str] = field(default_factory=list)

    # Caller info
    caller_language: str = "en"
    caller_name: str = ""
    patient_age: str = ""
    patient_relation: str = ""  # self, parent, bystander, etc.

    # Metadata
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turn_count: int = 0
    consent_given: bool = False

    # Protocol question tracking
    current_protocol_step: int = 0


class TriageStateMachine:
    """Deterministic finite state machine for medical triage.

    Flow:
    GREETING -> CONSENT -> CHIEF_COMPLAINT -> AIRWAY -> BREATHING ->
    CIRCULATION -> DISABILITY -> [protocol] -> PRE_ARRIVAL -> CALL_END

    At any point, if danger signs are detected -> RED -> HUMAN_ESCALATION
    """

    def __init__(self, session: TriageSession):
        self.session = session

    def process_nlu_result(self, nlu_result: dict) -> TriageState:
        """Process structured NLU output and transition to next state.

        Args:
            nlu_result: Structured data extracted from caller speech by NLU module.

        Returns:
            The new state after transition.
        """
        self.session.turn_count += 1
        state = self.session.state

        if state == TriageState.GREETING:
            return self._handle_greeting(nlu_result)
        elif state == TriageState.CONSENT:
            return self._handle_consent(nlu_result)
        elif state == TriageState.CHIEF_COMPLAINT:
            return self._handle_chief_complaint(nlu_result)
        elif state == TriageState.AIRWAY:
            return self._handle_airway(nlu_result)
        elif state == TriageState.BREATHING:
            return self._handle_breathing(nlu_result)
        elif state == TriageState.CIRCULATION:
            return self._handle_circulation(nlu_result)
        elif state == TriageState.DISABILITY:
            return self._handle_disability(nlu_result)
        elif state in (
            TriageState.MALARIA_PROTOCOL,
            TriageState.TRAUMA_PROTOCOL,
            TriageState.MATERNAL_PROTOCOL,
            TriageState.RESPIRATORY_PROTOCOL,
            TriageState.SNAKEBITE_PROTOCOL,
            TriageState.GENERAL_PROTOCOL,
        ):
            return self._handle_protocol(nlu_result)
        elif state == TriageState.PRE_ARRIVAL_INSTRUCTIONS:
            return self._handle_pre_arrival(nlu_result)
        elif state == TriageState.HUMAN_ESCALATION:
            return self._handle_escalation(nlu_result)
        else:
            return TriageState.CALL_END

    def _handle_greeting(self, nlu_result: dict) -> TriageState:
        if nlu_result.get("caller_name"):
            self.session.caller_name = nlu_result["caller_name"]
        self.session.state = TriageState.CONSENT
        return self.session.state

    def _handle_consent(self, nlu_result: dict) -> TriageState:
        if nlu_result.get("consent", False):
            self.session.consent_given = True
            self.session.state = TriageState.CHIEF_COMPLAINT
        else:
            # No consent - provide emergency numbers and end
            self.session.state = TriageState.CALL_END
        return self.session.state

    def _handle_chief_complaint(self, nlu_result: dict) -> TriageState:
        self.session.chief_complaint_raw = nlu_result.get("complaint_text", "")
        category = nlu_result.get("category")

        if category and category in ComplaintCategory.__members__.values():
            self.session.complaint_category = ComplaintCategory(category)
        else:
            self.session.complaint_category = ComplaintCategory.GENERAL

        if nlu_result.get("patient_age"):
            self.session.patient_age = nlu_result["patient_age"]
        if nlu_result.get("patient_relation"):
            self.session.patient_relation = nlu_result["patient_relation"]

        # Always go through danger signs first
        self.session.state = TriageState.AIRWAY
        return self.session.state

    def _handle_airway(self, nlu_result: dict) -> TriageState:
        self.session.danger_sign_findings["airway_compromised"] = nlu_result.get(
            "airway_compromised", False
        )
        self._check_immediate_red()
        if self.session.severity == Severity.RED:
            return self.session.state

        self.session.state = TriageState.BREATHING
        return self.session.state

    def _handle_breathing(self, nlu_result: dict) -> TriageState:
        self.session.danger_sign_findings["not_breathing"] = nlu_result.get(
            "not_breathing", False
        )
        self.session.danger_sign_findings["breathing_difficulty"] = nlu_result.get(
            "breathing_difficulty", False
        )
        self._check_immediate_red()
        if self.session.severity == Severity.RED:
            return self.session.state

        self.session.state = TriageState.CIRCULATION
        return self.session.state

    def _handle_circulation(self, nlu_result: dict) -> TriageState:
        self.session.danger_sign_findings["severe_bleeding"] = nlu_result.get(
            "severe_bleeding", False
        )
        self.session.danger_sign_findings["moderate_bleeding"] = nlu_result.get(
            "moderate_bleeding", False
        )
        self._check_immediate_red()
        if self.session.severity == Severity.RED:
            return self.session.state

        self.session.state = TriageState.DISABILITY
        return self.session.state

    def _handle_disability(self, nlu_result: dict) -> TriageState:
        self.session.danger_sign_findings["unconscious"] = nlu_result.get(
            "unconscious", False
        )
        self.session.danger_sign_findings["convulsing"] = nlu_result.get(
            "convulsing", False
        )
        self.session.danger_sign_findings["confused"] = nlu_result.get(
            "confused", False
        )
        self._check_immediate_red()
        if self.session.severity == Severity.RED:
            return self.session.state

        # Danger signs clear or yellow — proceed to condition protocol
        danger_severity = classify_danger_signs(self.session.danger_sign_findings)
        if danger_severity == Severity.YELLOW:
            self.session.severity = Severity.YELLOW

        protocol_state = COMPLAINT_TO_PROTOCOL.get(
            self.session.complaint_category, TriageState.GENERAL_PROTOCOL
        )
        self.session.state = protocol_state
        self.session.current_protocol_step = 0
        return self.session.state

    def _handle_protocol(self, nlu_result: dict) -> TriageState:
        """Handle condition-specific protocol questions."""
        # Merge findings
        self.session.protocol_findings.update(nlu_result.get("findings", {}))
        self.session.current_protocol_step += 1

        if nlu_result.get("protocol_complete", False):
            # Classify using the protocol-specific classifier
            protocol_name = self._get_protocol_name()
            classifier = PROTOCOL_CLASSIFIERS.get(protocol_name)
            if classifier:
                protocol_severity = classifier(self.session.protocol_findings)
                # Only upgrade severity, never downgrade
                if self.session.severity == Severity.PENDING:
                    self.session.severity = protocol_severity
                elif (
                    protocol_severity == Severity.RED
                    and self.session.severity != Severity.RED
                ):
                    self.session.severity = Severity.RED

            if self.session.severity == Severity.RED:
                self.session.state = TriageState.HUMAN_ESCALATION
            else:
                if self.session.severity == Severity.PENDING:
                    self.session.severity = Severity.GREEN
                self.session.state = TriageState.PRE_ARRIVAL_INSTRUCTIONS
        # else: stay in same protocol state, NLU pipeline will ask next question

        return self.session.state

    def _handle_pre_arrival(self, nlu_result: dict) -> TriageState:
        self.session.state = TriageState.CALL_END
        return self.session.state

    def _handle_escalation(self, nlu_result: dict) -> TriageState:
        self.session.state = TriageState.CALL_END
        return self.session.state

    def _check_immediate_red(self) -> None:
        """Check danger signs for immediate RED classification."""
        severity = classify_danger_signs(self.session.danger_sign_findings)
        if severity == Severity.RED:
            self.session.severity = Severity.RED
            self.session.state = TriageState.HUMAN_ESCALATION
            self.session.internal_notes.append(
                f"RED alert triggered at {self.session.state.value}: "
                f"{self.session.danger_sign_findings}"
            )

    def _get_protocol_name(self) -> str:
        """Get the protocol classifier name from current state."""
        state_to_name = {
            TriageState.MALARIA_PROTOCOL: "malaria",
            TriageState.TRAUMA_PROTOCOL: "trauma",
            TriageState.MATERNAL_PROTOCOL: "maternal",
            TriageState.RESPIRATORY_PROTOCOL: "respiratory",
            TriageState.SNAKEBITE_PROTOCOL: "snakebite",
            TriageState.GENERAL_PROTOCOL: "general",
        }
        return state_to_name.get(self.session.state, "general")

    def get_triage_report(self) -> dict:
        """Generate the behind-the-scenes triage report for medical professionals.

        This is NEVER shown to the caller.
        """
        return {
            "session_id": self.session.session_id,
            "severity": self.session.severity.value,
            "complaint_category": (
                self.session.complaint_category.value
                if self.session.complaint_category
                else "unknown"
            ),
            "chief_complaint": self.session.chief_complaint_raw,
            "danger_sign_findings": self.session.danger_sign_findings,
            "protocol_findings": self.session.protocol_findings,
            "internal_notes": self.session.internal_notes,
            "patient_info": {
                "name": self.session.caller_name,
                "age": self.session.patient_age,
                "relation": self.session.patient_relation,
            },
            "metadata": {
                "turns": self.session.turn_count,
                "started_at": self.session.started_at.isoformat(),
                "language": self.session.caller_language,
                "consent_given": self.session.consent_given,
            },
        }

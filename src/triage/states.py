from enum import Enum


class TriageState(str, Enum):
    """All possible states in the triage conversation flow."""

    # Entry states
    GREETING = "greeting"
    CONSENT = "consent"

    # Chief complaint
    CHIEF_COMPLAINT = "chief_complaint"

    # Danger signs (ABCDE - always runs first)
    AIRWAY = "airway"
    BREATHING = "breathing"
    CIRCULATION = "circulation"
    DISABILITY = "disability"

    # Condition-specific protocols
    MALARIA_PROTOCOL = "malaria_protocol"
    TRAUMA_PROTOCOL = "trauma_protocol"
    MATERNAL_PROTOCOL = "maternal_protocol"
    RESPIRATORY_PROTOCOL = "respiratory_protocol"
    SNAKEBITE_PROTOCOL = "snakebite_protocol"
    GENERAL_PROTOCOL = "general_protocol"

    # Resolution
    PRE_ARRIVAL_INSTRUCTIONS = "pre_arrival_instructions"
    HUMAN_ESCALATION = "human_escalation"
    CALL_END = "call_end"


class Severity(str, Enum):
    """WHO ETAT severity classification."""

    RED = "red"        # Emergency - immediate life threat
    YELLOW = "yellow"  # Priority - urgent, needs prompt attention
    GREEN = "green"    # Queue - non-urgent
    PENDING = "pending"  # Not yet classified


class ComplaintCategory(str, Enum):
    """Chief complaint categories that route to specific protocols."""

    MALARIA_FEVER = "malaria_fever"
    TRAUMA = "trauma"
    MATERNAL = "maternal"
    RESPIRATORY = "respiratory"
    SNAKEBITE = "snakebite"
    GENERAL = "general"


# Maps complaint categories to their protocol state
COMPLAINT_TO_PROTOCOL: dict[ComplaintCategory, TriageState] = {
    ComplaintCategory.MALARIA_FEVER: TriageState.MALARIA_PROTOCOL,
    ComplaintCategory.TRAUMA: TriageState.TRAUMA_PROTOCOL,
    ComplaintCategory.MATERNAL: TriageState.MATERNAL_PROTOCOL,
    ComplaintCategory.RESPIRATORY: TriageState.RESPIRATORY_PROTOCOL,
    ComplaintCategory.SNAKEBITE: TriageState.SNAKEBITE_PROTOCOL,
    ComplaintCategory.GENERAL: TriageState.GENERAL_PROTOCOL,
}

"""Safety guardrails for the triage system.

CORE RULE: The AI NEVER reveals medical conditions to the caller.
It asks questions, that's it. All medical analysis happens behind the scenes.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Patterns that should NEVER appear in caller-facing responses
FORBIDDEN_PATTERNS = [
    # Diagnostic language
    r"\b(you have|they have|he has|she has)\b.*\b(condition|disease|infection|syndrome)\b",
    r"\b(this (is|looks like|sounds like|could be|might be|appears to be))\b",
    r"\b(diagnosis|diagnose|diagnosed)\b",
    r"\b(i think (it'?s|this is|they have))\b",
    r"\b(suffering from|afflicted with)\b",
    r"\b(symptoms? (of|suggest|indicate|point to|consistent with))\b",
    # Specific condition names - never say these to the caller
    r"\b(malaria|stroke|heart attack|cardiac|sepsis|meningitis|eclampsia)\b",
    r"\b(pre-?eclampsia|hemorrhage|aneurysm|embolism|pneumonia)\b",
    r"\b(fracture|concussion|internal bleeding|organ failure)\b",
    r"\b(envenomation|neurotoxic|hemotoxic|cytotoxic)\b",
    # Medication/dosage language
    r"\b(take|administer|give|prescribe|dose|dosage|mg|milligram)\b.*\b(medicine|drug|tablet|pill|injection)\b",
    r"\b(paracetamol|ibuprofen|aspirin|antibiotic|antivenom|antimalarial)\b",
]

# Compile for performance
_FORBIDDEN_RE = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]

# Words that trigger immediate concern if in caller speech
DANGER_KEYWORDS = [
    "not breathing", "stopped breathing", "can't breathe", "cannot breathe",
    "choking", "unconscious", "unresponsive", "passed out", "not waking up",
    "so much blood", "bleeding heavily", "won't stop bleeding",
    "seizure", "convulsion", "shaking uncontrollably", "fitting",
    "blue lips", "turning blue", "grey", "cold and clammy",
    "no pulse", "heart stopped",
]


def check_response_safety(response_text: str) -> tuple[bool, str | None]:
    """Check if a response is safe to send to the caller.

    Returns:
        (is_safe, violation_description)
    """
    for pattern in _FORBIDDEN_RE:
        match = pattern.search(response_text)
        if match:
            violation = f"Forbidden pattern matched: '{match.group()}'"
            logger.warning("Response safety violation: %s", violation)
            return False, violation

    return True, None


def sanitize_response(response_text: str) -> str:
    """Remove any unsafe content from a response.

    If the response contains forbidden patterns, replace it with a safe fallback.
    """
    is_safe, violation = check_response_safety(response_text)
    if is_safe:
        return response_text

    logger.warning("Sanitizing unsafe response. Violation: %s", violation)
    return (
        "Thank you for that information. Let me ask you the next question "
        "so we can get help to you quickly."
    )


def check_danger_keywords(caller_text: str) -> list[str]:
    """Check caller's speech for immediate danger keywords.

    Returns list of matched danger keywords.
    """
    text_lower = caller_text.lower()
    matches = []
    for keyword in DANGER_KEYWORDS:
        if keyword in text_lower:
            matches.append(keyword)
    return matches


def should_escalate(
    turn_count: int,
    low_confidence_streak: int,
    caller_text: str,
    severity: str,
) -> tuple[bool, str]:
    """Determine if the call should be escalated to a human.

    Returns:
        (should_escalate, reason)
    """
    # RED severity -> always escalate
    if severity == "red":
        return True, "RED severity classification"

    # Caller requests human
    human_requests = [
        "talk to someone", "real person", "human", "let me speak",
        "transfer me", "operator", "agent",
    ]
    text_lower = caller_text.lower()
    for phrase in human_requests:
        if phrase in text_lower:
            return True, f"Caller requested human: '{phrase}'"

    # Too many low-confidence extractions
    if low_confidence_streak >= 3:
        return True, "3 consecutive low-confidence NLU extractions"

    # Call too long without resolution
    if turn_count > 20:
        return True, "Call exceeded 20 turns without classification"

    return False, ""

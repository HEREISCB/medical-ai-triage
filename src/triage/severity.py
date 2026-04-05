"""Deterministic severity classification.

The LLM NEVER decides severity. These are hard-coded clinical rules
based on WHO ETAT (Emergency Triage Assessment and Treatment).
"""

from src.triage.states import Severity


def classify_danger_signs(findings: dict) -> Severity:
    """Classify severity based on ABCDE danger sign findings.

    Any single RED trigger = immediate RED classification.
    These bypass the normal protocol flow entirely.
    """
    red_triggers = [
        findings.get("airway_compromised"),
        findings.get("not_breathing"),
        findings.get("severe_bleeding"),
        findings.get("unconscious"),
        findings.get("convulsing"),
    ]

    if any(red_triggers):
        return Severity.RED

    yellow_triggers = [
        findings.get("breathing_difficulty"),
        findings.get("moderate_bleeding"),
        findings.get("confused"),
        findings.get("high_fever"),
    ]

    if any(yellow_triggers):
        return Severity.YELLOW

    return Severity.PENDING


def classify_malaria(findings: dict) -> Severity:
    """Classify malaria/fever severity."""
    if any([
        findings.get("convulsions"),
        findings.get("unconscious"),
        findings.get("unable_to_drink"),
        findings.get("child_under_5") and findings.get("high_fever"),
    ]):
        return Severity.RED

    if any([
        findings.get("fever_over_3_days"),
        findings.get("vomiting_everything"),
        findings.get("very_weak"),
    ]):
        return Severity.YELLOW

    return Severity.GREEN


def classify_trauma(findings: dict) -> Severity:
    """Classify trauma/RTA severity."""
    if any([
        findings.get("unconscious"),
        findings.get("severe_bleeding"),
        findings.get("head_injury_with_confusion"),
        findings.get("suspected_spinal"),
        findings.get("chest_wound"),
    ]):
        return Severity.RED

    if any([
        findings.get("fracture_suspected"),
        findings.get("moderate_bleeding"),
        findings.get("multiple_injuries"),
        findings.get("abdominal_pain"),
    ]):
        return Severity.YELLOW

    return Severity.GREEN


def classify_maternal(findings: dict) -> Severity:
    """Classify maternal emergency severity."""
    if any([
        findings.get("seizures"),
        findings.get("heavy_bleeding"),
        findings.get("unconscious"),
        findings.get("cord_prolapse"),
        findings.get("severe_headache_with_blurred_vision"),
    ]):
        return Severity.RED

    if any([
        findings.get("regular_contractions"),
        findings.get("water_broken"),
        findings.get("moderate_bleeding"),
        findings.get("severe_abdominal_pain"),
    ]):
        return Severity.YELLOW

    return Severity.GREEN


def classify_respiratory(findings: dict) -> Severity:
    """Classify respiratory distress severity."""
    if any([
        findings.get("not_breathing"),
        findings.get("blue_lips"),
        findings.get("severe_chest_indrawing"),
        findings.get("unable_to_speak"),
    ]):
        return Severity.RED

    if any([
        findings.get("fast_breathing"),
        findings.get("wheezing"),
        findings.get("chest_pain"),
        findings.get("coughing_blood"),
    ]):
        return Severity.YELLOW

    return Severity.GREEN


def classify_snakebite(findings: dict) -> Severity:
    """Classify snakebite severity."""
    if any([
        findings.get("difficulty_breathing"),
        findings.get("swelling_spreading_fast"),
        findings.get("blurred_vision"),
        findings.get("unable_to_swallow"),
        findings.get("bleeding_from_gums"),
    ]):
        return Severity.RED

    if any([
        findings.get("significant_swelling"),
        findings.get("severe_pain"),
        findings.get("nausea_vomiting"),
        findings.get("bite_on_trunk_or_face"),
    ]):
        return Severity.YELLOW

    return Severity.GREEN


def classify_general(findings: dict) -> Severity:
    """Classify general/unspecified complaints."""
    if any([
        findings.get("unconscious"),
        findings.get("severe_pain"),
        findings.get("not_breathing"),
    ]):
        return Severity.RED

    if any([
        findings.get("moderate_pain"),
        findings.get("worsening_symptoms"),
        findings.get("unable_to_move"),
    ]):
        return Severity.YELLOW

    return Severity.GREEN


# Map protocol names to their classifiers
PROTOCOL_CLASSIFIERS = {
    "malaria": classify_malaria,
    "trauma": classify_trauma,
    "maternal": classify_maternal,
    "respiratory": classify_respiratory,
    "snakebite": classify_snakebite,
    "general": classify_general,
}

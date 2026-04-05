"""Question bank for triage states.

CRITICAL RULES:
1. Questions NEVER hint at or reveal medical conditions.
2. No legal disclaimers, no "I am an AI" nonsense.
3. The AI behaves like an emergency dispatcher — calm, direct, helpful.
4. First response to ANY caller: acknowledge their problem, then ask questions.
"""

from src.triage.states import TriageState


QUESTIONS: dict[str, list[dict]] = {
    # === GREETING — just acknowledge and start helping ===
    TriageState.GREETING: [
        {
            "text": (
                "Okay, I hear you. I'm going to help you right now. "
                "Can you tell me your name?"
            ),
            "expect": "caller_name",
        }
    ],
    # === CONSENT — skip the legal crap, just confirm they want help ===
    TriageState.CONSENT: [
        {
            "text": (
                "Alright {caller_name}, I need to ask you a few quick questions "
                "so we can get you the right help. Is that okay?"
            ),
            "expect": "consent",
        }
    ],
    # === CHIEF COMPLAINT ===
    TriageState.CHIEF_COMPLAINT: [
        {
            "text": "Tell me exactly what's happening right now. What do you see?",
            "expect": "complaint_text",
        },
        {
            "text": (
                "Is this happening to you, or to someone else? "
                "And roughly how old is the person?"
            ),
            "expect": "patient_relation,patient_age",
        },
    ],
    # === DANGER SIGNS (ABCDE) ===
    TriageState.AIRWAY: [
        {
            "text": "Can the person talk or make sounds right now? Can they swallow?",
            "expect": "airway_compromised",
        }
    ],
    TriageState.BREATHING: [
        {
            "text": (
                "Is the person breathing? Look at their chest — "
                "is it moving up and down?"
            ),
            "expect": "not_breathing,breathing_difficulty",
        }
    ],
    TriageState.CIRCULATION: [
        {
            "text": (
                "Is there any bleeding? If yes, how much — "
                "a small amount, or is it flowing or soaking through cloth?"
            ),
            "expect": "severe_bleeding,moderate_bleeding",
        }
    ],
    TriageState.DISABILITY: [
        {
            "text": (
                "Is the person awake and aware? "
                "Can they respond when you speak to them? "
                "Are they shaking or jerking?"
            ),
            "expect": "unconscious,convulsing,confused",
        }
    ],
    # === MALARIA/FEVER PROTOCOL ===
    TriageState.MALARIA_PROTOCOL: [
        {
            "text": "Does the person feel hot to the touch? How long have they been feeling this way?",
            "expect": "high_fever,fever_over_3_days",
        },
        {
            "text": "Have they been shaking or jerking at any point?",
            "expect": "convulsions",
        },
        {
            "text": "Can they drink water? Are they keeping it down or throwing up?",
            "expect": "unable_to_drink,vomiting_everything",
        },
        {
            "text": "How is their energy? Can they sit up, or are they too weak to move?",
            "expect": "very_weak",
        },
        {
            "text": None,
            "expect": "protocol_complete",
        },
    ],
    # === TRAUMA/RTA PROTOCOL ===
    TriageState.TRAUMA_PROTOCOL: [
        {
            "text": "What happened? Was it a fall, a vehicle accident, or something else?",
            "expect": "injury_type",
        },
        {
            "text": "Did they hit their head? Are they confused or saying things that don't make sense?",
            "expect": "head_injury_with_confusion",
        },
        {
            "text": "Can they move their arms and legs? Is anything bent or out of place?",
            "expect": "fracture_suspected,suspected_spinal",
        },
        {
            "text": "Is there any wound on the chest or belly?",
            "expect": "chest_wound,abdominal_pain",
        },
        {
            "text": "How many parts of the body are hurt?",
            "expect": "multiple_injuries",
        },
        {
            "text": None,
            "expect": "protocol_complete",
        },
    ],
    # === MATERNAL PROTOCOL ===
    TriageState.MATERNAL_PROTOCOL: [
        {
            "text": "How many months or weeks pregnant is she, roughly?",
            "expect": "weeks_pregnant",
        },
        {
            "text": "Is there any bleeding? How much — light spotting, or soaking through?",
            "expect": "heavy_bleeding,moderate_bleeding",
        },
        {
            "text": "Is she having contractions or strong belly pains? How often?",
            "expect": "regular_contractions,severe_abdominal_pain",
        },
        {
            "text": "Has her water broken? Is there fluid leaking?",
            "expect": "water_broken",
        },
        {
            "text": "Does she have a bad headache, blurry vision, or has she been shaking?",
            "expect": "severe_headache_with_blurred_vision,seizures",
        },
        {
            "text": None,
            "expect": "protocol_complete",
        },
    ],
    # === RESPIRATORY PROTOCOL ===
    TriageState.RESPIRATORY_PROTOCOL: [
        {
            "text": "Can the person speak full sentences, or do they stop to catch their breath?",
            "expect": "unable_to_speak",
        },
        {
            "text": "Look at their lips and fingernails — do they look blue or grey?",
            "expect": "blue_lips",
        },
        {
            "text": "Can you see the skin pulling in between the ribs when they breathe?",
            "expect": "severe_chest_indrawing",
        },
        {
            "text": "Are they breathing fast? Is there a whistling sound?",
            "expect": "fast_breathing,wheezing",
        },
        {
            "text": "Any pain in the chest? Are they coughing up blood?",
            "expect": "chest_pain,coughing_blood",
        },
        {
            "text": None,
            "expect": "protocol_complete",
        },
    ],
    # === SNAKEBITE PROTOCOL ===
    TriageState.SNAKEBITE_PROTOCOL: [
        {
            "text": "When did the bite happen? How long ago?",
            "expect": "time_since_bite",
        },
        {
            "text": "Where on the body is the bite?",
            "expect": "bite_on_trunk_or_face",
        },
        {
            "text": "Is the area around the bite swollen? Is the swelling spreading?",
            "expect": "significant_swelling,swelling_spreading_fast",
        },
        {
            "text": "Can the person see clearly? Can they swallow normally?",
            "expect": "blurred_vision,unable_to_swallow",
        },
        {
            "text": "Are their gums bleeding, or any bleeding from unusual places?",
            "expect": "bleeding_from_gums",
        },
        {
            "text": "Are they feeling sick? Is the pain very bad?",
            "expect": "nausea_vomiting,severe_pain",
        },
        {
            "text": None,
            "expect": "protocol_complete",
        },
    ],
    # === GENERAL PROTOCOL ===
    TriageState.GENERAL_PROTOCOL: [
        {
            "text": "On a scale of 1 to 10, how bad is the pain?",
            "expect": "severe_pain,moderate_pain",
        },
        {
            "text": "How long has this been going on? Is it getting worse or staying the same?",
            "expect": "worsening_symptoms",
        },
        {
            "text": "Can the person move around normally?",
            "expect": "unable_to_move",
        },
        {
            "text": None,
            "expect": "protocol_complete",
        },
    ],
}


def get_current_question(state: TriageState, step: int, session_data: dict = None) -> str | None:
    """Get the current question for the given state and step."""
    questions = QUESTIONS.get(state, [])
    if step >= len(questions):
        return None

    question = questions[step]
    text = question["text"]

    if text is None:
        return None

    if session_data:
        try:
            text = text.format(**session_data)
        except KeyError:
            pass

    return text


def is_protocol_complete(state: TriageState, step: int) -> bool:
    """Check if the current protocol step is the completion sentinel."""
    questions = QUESTIONS.get(state, [])
    if step >= len(questions):
        return True
    return questions[step]["text"] is None

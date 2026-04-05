"""Question bank for triage states.

CRITICAL RULE: Questions NEVER hint at or reveal medical conditions.
They are purely observational — asking the caller to describe what they SEE.
The AI is an interrogator, not an explainer.

Bad: "This sounds like it could be a stroke. Is the left side of their face drooping?"
Good: "Can you look at their face? Does one side look different from the other?"
"""

from src.triage.states import TriageState


# Each state maps to a list of questions asked in sequence.
# The NLU module picks the appropriate question based on current_protocol_step.

QUESTIONS: dict[str, list[dict]] = {
    # === ENTRY STATES ===
    TriageState.GREETING: [
        {
            "text": (
                "I'm here to help you. Take a deep breath. You're doing the right thing "
                "by reaching out. I'm going to ask you a few quick questions so we can "
                "get help to you as fast as possible. Can you tell me your name?"
            ),
            "expect": "caller_name",
        }
    ],
    TriageState.CONSENT: [
        {
            "text": (
                "Thank you, {caller_name}. Before we continue, I need you to know "
                "that I am an AI assistant. I will ask you questions to help get the "
                "right help to you quickly. This conversation may be recorded. "
                "If someone is in immediate danger right now, please call 999 or 112. "
                "Can we continue?"
            ),
            "expect": "consent",
        }
    ],
    # === CHIEF COMPLAINT ===
    TriageState.CHIEF_COMPLAINT: [
        {
            "text": (
                "Tell me what's happening right now. What do you see? "
                "Take your time — just describe what's going on."
            ),
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
            "text": (
                "Can the person talk or make sounds right now? "
                "Are they able to swallow?"
            ),
            "expect": "airway_compromised",
        }
    ],
    TriageState.BREATHING: [
        {
            "text": (
                "Is the person breathing? Can you watch their chest — "
                "is it moving up and down?"
            ),
            "expect": "not_breathing,breathing_difficulty",
        }
    ],
    TriageState.CIRCULATION: [
        {
            "text": (
                "Is there any bleeding? If yes, how much — "
                "is it a small amount, or is it flowing or soaking through cloth?"
            ),
            "expect": "severe_bleeding,moderate_bleeding",
        }
    ],
    TriageState.DISABILITY: [
        {
            "text": (
                "Is the person awake and aware of what's going on? "
                "Can they respond when you speak to them? "
                "Are they shaking or jerking?"
            ),
            "expect": "unconscious,convulsing,confused",
        }
    ],
    # === MALARIA/FEVER PROTOCOL ===
    TriageState.MALARIA_PROTOCOL: [
        {
            "text": "Does the person feel hot to the touch? For how long have they been feeling this way?",
            "expect": "high_fever,fever_over_3_days",
            "findings_key": "fever_duration",
        },
        {
            "text": "Have they been shaking or jerking at any point?",
            "expect": "convulsions",
        },
        {
            "text": "Can they drink water or other fluids? Are they keeping it down, or throwing up?",
            "expect": "unable_to_drink,vomiting_everything",
        },
        {
            "text": "How is their energy? Can they sit up or walk, or are they too weak to move?",
            "expect": "very_weak",
        },
        {
            "text": None,  # Sentinel: protocol complete
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
            "text": "Can they move their arms and legs? Is there anything that looks bent or out of place?",
            "expect": "fracture_suspected,suspected_spinal",
        },
        {
            "text": "Is there any wound on the chest or belly? Can you see anything coming out?",
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
            "text": "Is she having contractions or strong belly pains? How often are they coming?",
            "expect": "regular_contractions,severe_abdominal_pain",
        },
        {
            "text": "Has her water broken? Is there any fluid leaking?",
            "expect": "water_broken",
        },
        {
            "text": "Does she have a bad headache, or is her vision blurry? Has she been shaking?",
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
            "text": "Can the person speak full sentences, or do they have to stop to catch their breath?",
            "expect": "unable_to_speak",
        },
        {
            "text": "Look at their lips and fingernails — do they look blue or grey?",
            "expect": "blue_lips",
        },
        {
            "text": "Can you see the skin pulling in between the ribs or under the neck when they breathe?",
            "expect": "severe_chest_indrawing",
        },
        {
            "text": "Are they breathing fast? Do they have a whistling sound when they breathe?",
            "expect": "fast_breathing,wheezing",
        },
        {
            "text": "Is there any pain in the chest? Are they coughing up any blood?",
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
            "text": "Where on the body is the bite? Is it on the arm, leg, or somewhere else?",
            "expect": "bite_on_trunk_or_face",
        },
        {
            "text": "Is the area around the bite swollen? Is the swelling spreading or staying the same?",
            "expect": "significant_swelling,swelling_spreading_fast",
        },
        {
            "text": "Can the person see clearly? Can they swallow normally?",
            "expect": "blurred_vision,unable_to_swallow",
        },
        {
            "text": "Are their gums bleeding, or is there bleeding from anywhere unusual?",
            "expect": "bleeding_from_gums",
        },
        {
            "text": "Are they feeling sick to their stomach? Is the pain very bad?",
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
            "text": "On a scale of 1 to 10, how bad is the pain or discomfort?",
            "expect": "severe_pain,moderate_pain",
        },
        {
            "text": "How long has this been going on? Is it getting worse, staying the same, or getting better?",
            "expect": "worsening_symptoms",
        },
        {
            "text": "Can the person move around normally, or are they stuck in one position?",
            "expect": "unable_to_move",
        },
        {
            "text": None,
            "expect": "protocol_complete",
        },
    ],
}


def get_current_question(state: TriageState, step: int, session_data: dict = None) -> str | None:
    """Get the current question for the given state and step.

    Returns None if no more questions (protocol complete).
    """
    questions = QUESTIONS.get(state, [])
    if step >= len(questions):
        return None

    question = questions[step]
    text = question["text"]

    if text is None:
        return None  # Protocol complete sentinel

    # Template substitution for caller name etc.
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

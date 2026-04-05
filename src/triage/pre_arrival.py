"""Pre-arrival first-aid instructions.

These are simple, safe instructions given to the caller while help is on the way.
They NEVER mention the suspected condition — only what to DO.
"""

from src.triage.states import ComplaintCategory, Severity


def get_pre_arrival_instructions(
    category: ComplaintCategory | None,
    severity: Severity,
    findings: dict,
) -> str:
    """Get appropriate pre-arrival instructions based on findings.

    Instructions are action-oriented: what to DO, not what the condition IS.
    """
    instructions = []

    # Universal safety instructions
    instructions.append(
        "Stay with the person and keep them as comfortable as possible."
    )

    if severity == Severity.RED:
        instructions.append(
            "This is urgent. Please call 999 or 112 right now if you haven't already. "
            "I will tell you what to do while you wait for help."
        )

    if findings.get("not_breathing"):
        instructions.append(
            "If you know how to do CPR, start chest compressions now. "
            "Push hard and fast in the center of the chest. "
            "If you don't know CPR, I'll guide you: place the heel of your hand "
            "on the center of the chest, lock your other hand on top, and push "
            "down firmly about 30 times, then give 2 breaths into their mouth."
        )
    elif findings.get("airway_compromised"):
        instructions.append(
            "Gently tilt their head back and lift the chin to open the airway. "
            "If there is something visible in the mouth, carefully remove it. "
            "Do not put your fingers deep into the throat."
        )

    if findings.get("severe_bleeding") or findings.get("heavy_bleeding"):
        instructions.append(
            "Take a clean cloth, towel, or piece of clothing and press it firmly "
            "on the wound. Keep pressing and do not lift it to check. "
            "If it soaks through, add more cloth on top."
        )

    if findings.get("unconscious") and not findings.get("not_breathing"):
        instructions.append(
            "If the person is breathing but not awake, gently roll them onto their "
            "side. This helps keep the airway clear. Support the head."
        )

    if findings.get("convulsing"):
        instructions.append(
            "Do not hold them down or put anything in their mouth. "
            "Move any hard or sharp objects away from them. "
            "Place something soft under their head if you can. "
            "Time the shaking if possible — note when it started."
        )

    if findings.get("suspected_spinal"):
        instructions.append(
            "Do not move the person. Keep their head, neck, and back still. "
            "If they must be moved for safety, support the head and neck together "
            "and move them as one unit."
        )

    if category == ComplaintCategory.SNAKEBITE:
        instructions.append(
            "Keep the bitten area still and below the level of the heart. "
            "Do not cut the bite, do not suck out venom, do not apply ice. "
            "Remove any rings or tight items near the bite before swelling increases."
        )

    if category == ComplaintCategory.MATERNAL:
        if findings.get("heavy_bleeding"):
            instructions.append(
                "Have her lie down with her legs raised slightly. "
                "Do not insert anything. Keep her warm."
            )
        if findings.get("seizures"):
            instructions.append(
                "Protect her from injury during the shaking. "
                "After the shaking stops, roll her onto her left side."
            )

    # Always end with reassurance
    instructions.append(
        "You are doing great. Help is being arranged. Stay on the line with me."
    )

    return " ".join(instructions)

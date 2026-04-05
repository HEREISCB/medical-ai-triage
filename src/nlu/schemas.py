"""Pydantic schemas for structured NLU extraction.

These define what the LLM must extract from the caller's speech
for each triage state. The LLM outputs JSON matching these schemas.
"""

from pydantic import BaseModel, Field


class GreetingExtraction(BaseModel):
    caller_name: str = Field(default="", description="Caller's name if provided")


class ConsentExtraction(BaseModel):
    consent: bool = Field(description="Whether the caller consented to continue")


class ChiefComplaintExtraction(BaseModel):
    complaint_text: str = Field(description="Raw description of the emergency")
    category: str = Field(
        description=(
            "One of: malaria_fever, trauma, maternal, respiratory, snakebite, general"
        )
    )
    patient_age: str = Field(default="", description="Approximate age if mentioned")
    patient_relation: str = Field(
        default="self",
        description="Who is the patient: self, child, parent, spouse, stranger, other",
    )


class AirwayExtraction(BaseModel):
    airway_compromised: bool = Field(
        default=False,
        description="True if person cannot talk, make sounds, or swallow",
    )


class BreathingExtraction(BaseModel):
    not_breathing: bool = Field(default=False, description="True if not breathing at all")
    breathing_difficulty: bool = Field(
        default=False, description="True if breathing but with difficulty"
    )


class CirculationExtraction(BaseModel):
    severe_bleeding: bool = Field(
        default=False, description="True if heavy/uncontrolled bleeding"
    )
    moderate_bleeding: bool = Field(
        default=False, description="True if some bleeding but controlled"
    )


class DisabilityExtraction(BaseModel):
    unconscious: bool = Field(default=False, description="True if person is not responsive")
    convulsing: bool = Field(default=False, description="True if shaking/jerking")
    confused: bool = Field(default=False, description="True if disoriented/confused")


class ProtocolFindingsExtraction(BaseModel):
    """Generic findings extraction for protocol-specific questions.

    The findings dict keys match the severity classifier expected keys.
    """

    findings: dict[str, bool | str] = Field(
        default_factory=dict,
        description="Key-value findings extracted from caller response",
    )
    protocol_complete: bool = Field(
        default=False,
        description="True only when all protocol questions have been asked",
    )
    confidence: float = Field(
        default=0.8,
        description="Confidence score 0-1 for extraction accuracy",
    )

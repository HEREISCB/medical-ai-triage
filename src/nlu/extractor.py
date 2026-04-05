"""NLU extraction using Groq (Llama 3.3 70B).

Extracts structured data from caller speech. The LLM is used ONLY for
understanding natural language — never for medical decisions or generating
caller-facing responses (that's handled by the pipeline's response generator).
"""

import json
import logging

from groq import AsyncGroq

from src.config import settings
from src.nlu.prompts import SYSTEM_PROMPT, get_extraction_prompt, COMPLAINT_CLASSIFIER_PROMPT
from src.triage.states import TriageState

logger = logging.getLogger(__name__)

# Groq client - initialized lazily
_groq_client: AsyncGroq | None = None


def get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


async def extract_structured_data(
    caller_text: str,
    state: TriageState,
    question_asked: str,
    expected_fields: str,
) -> dict:
    """Extract structured data from caller's speech using Groq/Llama.

    Args:
        caller_text: Raw transcription from STT.
        state: Current triage state.
        question_asked: The question that was asked to the caller.
        expected_fields: Comma-separated field names to extract.

    Returns:
        Parsed JSON dict with extracted fields.
    """
    client = get_groq_client()

    extraction_prompt = get_extraction_prompt(
        state=state.value,
        question_context=question_asked,
        expected_fields=expected_fields,
    )

    try:
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{extraction_prompt}\n\nCaller said: \"{caller_text}\""},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        logger.info("NLU extraction for %s: %s", state.value, result)
        return result

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.error("NLU extraction failed for %s: %s", state.value, e)
        return {"confidence": 0.0, "error": str(e)}
    except Exception as e:
        logger.error("Groq API error: %s", e)
        return {"confidence": 0.0, "error": str(e)}


async def classify_complaint(complaint_text: str) -> str:
    """Classify a chief complaint into a triage protocol category.

    Returns one of: malaria_fever, trauma, maternal, respiratory, snakebite, general
    """
    client = get_groq_client()

    try:
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": COMPLAINT_CLASSIFIER_PROMPT},
                {"role": "user", "content": complaint_text},
            ],
            temperature=0.0,
            max_tokens=20,
        )

        category = response.choices[0].message.content.strip().lower()

        valid_categories = {
            "malaria_fever", "trauma", "maternal",
            "respiratory", "snakebite", "general",
        }
        if category not in valid_categories:
            logger.warning("Unknown complaint category '%s', defaulting to general", category)
            return "general"

        return category

    except Exception as e:
        logger.error("Complaint classification failed: %s", e)
        return "general"

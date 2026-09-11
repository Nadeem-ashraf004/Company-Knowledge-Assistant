from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import settings


@lru_cache
def get_llm_client() -> genai.Client:
    """
    Create and cache the Gemini client.
    """

    if not settings.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured."
        )

    return genai.Client(
        api_key=settings.GEMINI_API_KEY
    )


def generate_answer(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> str:
    """
    Generate a grounded answer using Gemini.
    """

    client = get_llm_client()

    response = client.models.generate_content(
        model=settings.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    print("[LLM] Response:", repr(response.text))

    if response.candidates:
        candidate = response.candidates[0]
        print("[LLM] Finish reason:", candidate.finish_reason)

    return response.text.strip()
"""
Gemini client for item classification.

Mirrors `entity_extraction/models.py`: Vertex AI is the primary path
(GOOGLE_APPLICATION_CREDENTIALS + PROJECT_ID + LOCATION). If those are not set
the client falls back to a Gemini Developer API key (GOOGLE_API_KEY).
"""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_MODEL = os.getenv("ITEM_GROUPING_MODEL", "gemini-2.5-pro")
THINKING_BUDGET = int(os.getenv("ITEM_GROUPING_THINKING_BUDGET", "250"))


def init_client():
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")
    if creds and project:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
        return genai.Client(vertexai=True, project=project, location=location)
    if os.getenv("GOOGLE_API_KEY"):
        return genai.Client()
    raise SystemExit(
        "No Gemini credentials. Set GOOGLE_APPLICATION_CREDENTIALS + PROJECT_ID + "
        "LOCATION for Vertex AI, or GOOGLE_API_KEY for the Developer API."
    )


def generate(client, prompt, model=DEFAULT_MODEL, retries=4):
    """One completion, with backoff on transient API errors."""
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
    )
    delay = 5
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
            return response.text or ""
        except Exception as exc:  # transient quota / 5xx
            if attempt == retries - 1:
                raise
            print(f"    [retry {attempt + 1}/{retries - 1}] {type(exc).__name__}: {exc}")
            time.sleep(delay)
            delay *= 2
    return ""

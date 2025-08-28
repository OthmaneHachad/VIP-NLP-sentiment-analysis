"""
We will run *sentiment classification with justification* on finance-related sentences.
"""
from dataclasses import dataclass

SYSTEM_SENTIMENT = (
    "You are a careful financial NLP assistant. "
    "Your task is to classify the SENTIMENT of a short finance-related sentence as one of: "
    "{labels}. "
    "Return a strict JSON object with keys: sentiment (string from the allowed set) and "
    "rationale (one sentence explaining the choice). Do not include any extra text."
)

USER_SENTIMENT_TEMPLATE = (
    "Classify the sentiment for the following finance-related sentence.\n\n"
    "SENTENCE: {text}\n"
    "Allowed labels: {labels}\n"
    "Output JSON ONLY."
)

DEFAULT_LABELS = ["negative", "neutral", "positive"]


def build_sentiment_prompts(text: str, labels=None):
    labels = labels or DEFAULT_LABELS
    system = SYSTEM_SENTIMENT.format(labels=", ".join(labels))
    user = USER_SENTIMENT_TEMPLATE.format(text=text, labels=", ".join(labels))
    return system, user
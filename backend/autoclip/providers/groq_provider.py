"""Groq provider using Groq's OpenAI-compatible API."""

from __future__ import annotations

from .base import ProviderStatus
from .openai_provider import OpenAIProvider

DEFAULT_MODEL = "openai/gpt-oss-20b"

SUGGESTED_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


class GroqProvider(OpenAIProvider):
    """Groq adapter backed by the OpenAI-compatible client."""

    name = "groq"
    requires_key = True

    def __init__(
        self,
        model: str = "",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(
            model or DEFAULT_MODEL,
            api_key=api_key,
            base_url=base_url or "https://api.groq.com/openai/v1",
        )

    async def health_check(self) -> ProviderStatus:
        status = await super().health_check()
        if not status.models:
            status.models = SUGGESTED_MODELS
        return status

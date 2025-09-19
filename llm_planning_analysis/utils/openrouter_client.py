"""Utilities for creating a configured OpenRouter client.

This module centralises the logic for instantiating the OpenAI Python
client so that every OpenAI-compatible request made by the repository is
routed through OpenRouter.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

from openai import OpenAI


def _optional_header(name: str, header: str, headers: Dict[str, str]) -> None:
    value = os.getenv(name)
    if value:
        headers[header] = value


@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenAI:
    """Return a cached OpenAI client configured for OpenRouter."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY must be set to use the OpenRouter client."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    default_headers: Dict[str, str] = {}
    _optional_header("OPENROUTER_REFERER", "HTTP-Referer", default_headers)
    _optional_header("OPENROUTER_APP_NAME", "X-Title", default_headers)

    return OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)


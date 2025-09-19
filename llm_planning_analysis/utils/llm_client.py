"""Utility helpers for constructing LLM clients.

This module centralises creation of the OpenAI SDK client so the rest of the
codebase can talk to OpenRouter without each caller re-declaring connection
parameters.  It reads credentials and optional metadata from environment
variables that mirror OpenRouter's configuration knobs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from openai.types.responses import Response


def _build_default_headers() -> Optional[Dict[str, str]]:
    """Construct optional headers required by OpenRouter.

    Both ``HTTP-Referer`` and ``X-Title`` are optional for self-hosted usage but
    recommended when routing traffic through OpenRouter.  They are exposed as
    environment variables so deployments can supply organisation-specific
    values without modifying source code.
    """

    headers: Dict[str, str] = {}

    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer

    title = os.getenv("OPENROUTER_TITLE") or os.getenv("OPENROUTER_X_TITLE")
    if title:
        headers["X-Title"] = title

    return headers or None


@lru_cache()
def get_llm_client() -> OpenAI:
    """Return a singleton OpenAI SDK client configured for OpenRouter."""

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Configure your OpenRouter API key "
            "in the environment before running the pipeline."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    default_headers = _build_default_headers()

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
    )


def prepare_responses_messages(
    messages: List[Dict[str, str]],
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Split chat-style messages into responses instructions and dialogue.

    The OpenAI Responses API accepts a dedicated ``instructions`` field in
    place of ``system`` messages.  This helper collapses any system-role
    content into a single instruction string while leaving the remainder of
    the conversation untouched.
    """

    instructions_parts: List[str] = []
    conversation: List[Dict[str, str]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            instructions_parts.append(content)
        else:
            conversation.append({"role": role, "content": content})

    instructions = "\n\n".join(instructions_parts) if instructions_parts else None

    if not conversation:
        conversation = [{"role": "user", "content": ""}]

    return instructions, conversation


def extract_response_text(response: Response) -> str:
    """Return the concatenated text output from a Responses result."""

    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    text_parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                text_parts.append(getattr(content, "text", ""))

    return "".join(text_parts)


def response_to_dict(response: Response) -> Dict[str, Any]:
    """Convert a Responses object into a JSON-serialisable dictionary.

    The helper also normalises usage statistics so downstream code can rely on
    the historical ``prompt_tokens``/``completion_tokens`` keys.
    """

    response_dict = response.model_dump(mode="json")
    usage = response_dict.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        usage.setdefault("prompt_tokens", prompt)
        usage.setdefault("completion_tokens", completion)

    return response_dict


__all__ = [
    "get_llm_client",
    "prepare_responses_messages",
    "extract_response_text",
    "response_to_dict",
]


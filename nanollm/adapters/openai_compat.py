"""OpenAI-compatible adapter for NanoLLM.

Handles: OpenAI, Groq, Together, Mistral, Deepseek, Perplexity,
Fireworks, OpenRouter, DeepInfra, Anyscale, xAI, Cerebras,
and any custom OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
from typing import Any

from .._types import _AttrDict


# Parameters the OpenAI chat completions API accepts (beyond model/messages/stream).
_CHAT_PARAMS = frozenset({
    "temperature",
    "top_p",
    "n",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "user",
    "tools",
    "tool_choice",
    "response_format",
    "seed",
    "service_tier",
    "parallel_tool_calls",
})


def build_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    stream: bool = False,
    extra_headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict:
    """Build an HTTP request dict for /chat/completions.

    Messages are passed through as-is since OpenAI format is the native format
    (including multimodal image_url content blocks).
    """
    url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    # Add stream_options for usage in streaming responses
    if stream:
        body["stream_options"] = {"include_usage": True}

    for key in _CHAT_PARAMS:
        if key in kwargs:
            body[key] = kwargs[key]

    return {"url": url, "headers": headers, "body": body}


def parse_response(raw: dict) -> dict:
    """Parse a chat completions response into a normalized dict.

    Returns:
        {
            "content": str | None,
            "finish_reason": str | None,
            "tool_calls": list[dict] | None,
            "usage": dict,
            "model": str,
        }
    """
    choice = raw.get("choices", [{}])[0]
    message = choice.get("message", {})

    usage_raw = raw.get("usage") or {}
    usage = {
        "prompt_tokens": usage_raw.get("prompt_tokens", 0),
        "completion_tokens": usage_raw.get("completion_tokens", 0),
        "total_tokens": usage_raw.get("total_tokens", 0),
    }

    # Wrap token detail sub-objects so they support attribute access
    completion_details = usage_raw.get("completion_tokens_details")
    if completion_details and isinstance(completion_details, dict):
        usage["completion_tokens_details"] = _AttrDict(completion_details)

    prompt_details = usage_raw.get("prompt_tokens_details")
    if prompt_details and isinstance(prompt_details, dict):
        usage["prompt_tokens_details"] = _AttrDict(prompt_details)

    return {
        "content": message.get("content"),
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": message.get("tool_calls"),
        "usage": usage,
        "model": raw.get("model", ""),
    }


def parse_stream_chunk(line: str) -> dict | None:
    """Parse a single SSE data line from a streaming response.

    Args:
        line: A line from the SSE stream (may or may not have 'data: ' prefix).

    Returns:
        Parsed chunk dict, or None for [DONE] / empty / unparseable lines.
    """
    line = line.strip()
    if not line:
        return None

    if line.startswith("data: "):
        line = line[6:]

    if line == "[DONE]":
        return None

    try:
        return json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None


# ── Embeddings ────────────────────────────────────────────────────────


def build_embedding_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    input: str | list[str],
    extra_headers: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict:
    """Build an HTTP request dict for /embeddings."""
    url = f"{base_url.rstrip('/')}/embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    body: dict[str, Any] = {
        "model": model,
        "input": input,
    }
    for key in ("encoding_format", "dimensions", "user"):
        if key in kwargs:
            body[key] = kwargs[key]

    return {"url": url, "headers": headers, "body": body}


def parse_embedding_response(raw: dict) -> dict:
    """Parse an embeddings response into a normalized dict.

    Returns:
        {
            "embeddings": list[list[float]],
            "usage": dict,
            "model": str,
        }
    """
    embeddings = [item["embedding"] for item in raw.get("data", [])]
    usage_raw = raw.get("usage") or {}
    usage = {
        "prompt_tokens": usage_raw.get("prompt_tokens", 0),
        "total_tokens": usage_raw.get("total_tokens", 0),
    }

    return {
        "embeddings": embeddings,
        "usage": usage,
        "model": raw.get("model", ""),
    }

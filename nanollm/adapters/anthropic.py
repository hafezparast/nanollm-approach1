"""Anthropic Messages API adapter for NanoLLM.

Converts between OpenAI message format and Anthropic's native format,
including multimodal content with images.
"""

from __future__ import annotations

import json
from typing import Any

from .._image import extract_image_url, to_anthropic_image


# Map Anthropic stop reasons to OpenAI finish reasons
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}

# Parameters to forward from kwargs to the Anthropic API
_ANTHROPIC_PARAMS = frozenset({
    "temperature",
    "top_p",
    "top_k",
    "metadata",
})


def _convert_content_block(block: dict) -> dict:
    """Convert a single OpenAI content block to Anthropic format."""
    block_type = block.get("type", "")

    if block_type == "text":
        return {"type": "text", "text": block.get("text", "")}

    if block_type == "image_url":
        url = extract_image_url(block)
        if url:
            return to_anthropic_image(url)
        # Fallback: skip unrecognized image block
        return {"type": "text", "text": "[image]"}

    # Pass through unknown block types as text
    return {"type": "text", "text": block.get("text", str(block))}


def _convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns:
        (system_prompt, converted_messages)
    """
    system_parts: list[str] = []
    converted: list[dict] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        # Extract system messages into top-level system parameter
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block["text"])
                    elif isinstance(block, str):
                        system_parts.append(block)
            continue

        # Convert content to Anthropic format
        if isinstance(content, str):
            anthropic_content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            anthropic_content = [_convert_content_block(b) for b in content]
        elif content is None:
            anthropic_content = [{"type": "text", "text": ""}]
        else:
            anthropic_content = [{"type": "text", "text": str(content)}]

        # Map tool role to user role with tool_result content
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            converted.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content if isinstance(content, str) else json.dumps(content),
                    }
                ],
            })
            continue

        # Handle assistant messages with tool_calls
        if role == "assistant" and msg.get("tool_calls"):
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": content if isinstance(content, str) else str(content)})
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": args_str}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            converted.append({"role": "assistant", "content": blocks})
            continue

        converted.append({
            "role": role,
            "content": anthropic_content,
        })

    system = "\n\n".join(system_parts) if system_parts else None
    return system, converted


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tools to Anthropic format."""
    anthropic_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
    return anthropic_tools


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
    """Build an HTTP request dict for Anthropic /messages."""
    url = f"{base_url.rstrip('/')}/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    system, converted_messages = _convert_messages(messages)

    body: dict[str, Any] = {
        "model": model,
        "messages": converted_messages,
        "max_tokens": kwargs.pop("max_tokens", kwargs.pop("max_completion_tokens", 4096)),
        "stream": stream,
    }

    if system:
        body["system"] = system

    # Handle response_format for JSON mode
    response_format = kwargs.pop("response_format", None)
    if response_format and response_format.get("type") == "json_object":
        json_instruction = "You must respond with valid JSON only. No other text."
        if body.get("system"):
            body["system"] = f"{body['system']}\n\n{json_instruction}"
        else:
            body["system"] = json_instruction

    # Map stop → stop_sequences
    stop = kwargs.pop("stop", None)
    if stop is not None:
        if isinstance(stop, str):
            body["stop_sequences"] = [stop]
        elif isinstance(stop, list):
            body["stop_sequences"] = stop

    # Convert tools
    tools = kwargs.pop("tools", None)
    if tools:
        body["tools"] = _convert_tools(tools)

    tool_choice = kwargs.pop("tool_choice", None)
    if tool_choice is not None:
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                # Anthropic doesn't have "none" — omit tools instead
                body.pop("tools", None)
            elif tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            body["tool_choice"] = {
                "type": "tool",
                "name": tool_choice["function"]["name"],
            }

    # Forward supported params
    for key in _ANTHROPIC_PARAMS:
        if key in kwargs:
            body[key] = kwargs[key]

    return {"url": url, "headers": headers, "body": body}


def parse_response(raw: dict) -> dict:
    """Parse an Anthropic messages response into a normalized dict.

    Returns:
        {
            "content": str | None,
            "finish_reason": str | None,
            "tool_calls": list[dict] | None,
            "usage": dict,
            "model": str,
        }
    """
    content_blocks = raw.get("content", [])

    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content_blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    content = "\n".join(text_parts) if text_parts else None
    stop_reason = raw.get("stop_reason", "")
    finish_reason = _STOP_REASON_MAP.get(stop_reason, stop_reason or None)

    usage_raw = raw.get("usage", {})
    usage = {
        "prompt_tokens": usage_raw.get("input_tokens", 0),
        "completion_tokens": usage_raw.get("output_tokens", 0),
        "total_tokens": usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
    }

    return {
        "content": content,
        "finish_reason": finish_reason,
        "tool_calls": tool_calls or None,
        "usage": usage,
        "model": raw.get("model", ""),
    }


def parse_stream_chunk(line: str) -> dict | None:
    """Parse a single SSE line from an Anthropic streaming response.

    Anthropic uses event-based SSE with types like:
    - message_start, content_block_start, content_block_delta,
      content_block_stop, message_delta, message_stop

    Returns an OpenAI-compatible chunk dict, or None if the line
    should be skipped.
    """
    line = line.strip()
    if not line:
        return None

    # Handle event: lines — store for context but don't emit
    if line.startswith("event:"):
        return None

    if not line.startswith("data:"):
        return None

    data_str = line[5:].strip()
    if not data_str:
        return None

    try:
        data = json.loads(data_str)
    except (json.JSONDecodeError, ValueError):
        return None

    event_type = data.get("type", "")

    if event_type == "content_block_delta":
        delta = data.get("delta", {})
        delta_type = delta.get("type", "")

        if delta_type == "text_delta":
            return {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta.get("text", "")},
                        "finish_reason": None,
                    }
                ],
            }

        if delta_type == "input_json_delta":
            # Tool call argument streaming
            return {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": data.get("index", 0),
                                    "function": {
                                        "arguments": delta.get("partial_json", ""),
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }

    if event_type == "content_block_start":
        block = data.get("content_block", {})
        if block.get("type") == "tool_use":
            return {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": data.get("index", 0),
                                    "id": block.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }

    if event_type == "message_delta":
        delta = data.get("delta", {})
        stop_reason = delta.get("stop_reason", "")
        finish_reason = _STOP_REASON_MAP.get(stop_reason, stop_reason or None)
        usage_raw = data.get("usage", {})

        chunk: dict[str, Any] = {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }

        if usage_raw:
            chunk["usage"] = {
                "prompt_tokens": usage_raw.get("input_tokens", 0),
                "completion_tokens": usage_raw.get("output_tokens", 0),
                "total_tokens": (
                    usage_raw.get("input_tokens", 0)
                    + usage_raw.get("output_tokens", 0)
                ),
            }

        return chunk

    if event_type == "message_start":
        message = data.get("message", {})
        usage_raw = message.get("usage", {})
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
            "model": message.get("model", ""),
        }
        if usage_raw:
            chunk["usage"] = {
                "prompt_tokens": usage_raw.get("input_tokens", 0),
                "completion_tokens": usage_raw.get("output_tokens", 0),
                "total_tokens": (
                    usage_raw.get("input_tokens", 0)
                    + usage_raw.get("output_tokens", 0)
                ),
            }
        return chunk

    if event_type == "message_stop":
        return None

    return None

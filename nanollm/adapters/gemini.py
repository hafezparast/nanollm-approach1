"""Google Gemini generateContent adapter for NanoLLM.

Translates between OpenAI message format and Gemini's native
generateContent / streamGenerateContent API. Supports multimodal
messages with images via to_gemini_image().
"""

from __future__ import annotations

import json
from typing import Any

from .._image import extract_image_url, to_gemini_image


# Gemini finish reason → OpenAI finish reason
_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
}


def _convert_content_block(block: dict) -> dict:
    """Convert a single OpenAI content block to a Gemini part."""
    if block.get("type") == "text":
        return {"text": block["text"]}

    url = extract_image_url(block)
    if url:
        return to_gemini_image(url)

    # Fallback: pass through as text
    return {"text": str(block)}


def _convert_message_content(content: str | list[dict]) -> list[dict]:
    """Convert OpenAI message content to Gemini parts list."""
    if isinstance(content, str):
        return [{"text": content}]

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append({"text": block})
        elif isinstance(block, dict):
            parts.append(_convert_content_block(block))
    return parts


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI tools format to Gemini function declarations."""
    declarations = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            decl: dict[str, Any] = {"name": fn["name"]}
            if "description" in fn:
                decl["description"] = fn["description"]
            if "parameters" in fn:
                decl["parameters"] = fn["parameters"]
            declarations.append(decl)
    if declarations:
        return [{"function_declarations": declarations}]
    return []


def _convert_messages(messages: list[dict]) -> tuple[list[dict] | None, list[dict]]:
    """Convert OpenAI messages to Gemini format.

    Returns:
        (system_instruction_parts, contents) where system_instruction_parts
        is a list of parts for the system_instruction field (or None),
        and contents is the Gemini contents array.
    """
    system_parts: list[dict] = []
    contents: list[dict] = []

    role_map = {"assistant": "model", "user": "user"}

    for msg in messages:
        role = msg.get("role", "user")

        # System messages become system_instruction
        if role == "system":
            text = msg.get("content", "")
            if isinstance(text, str):
                system_parts.append({"text": text})
            elif isinstance(text, list):
                system_parts.extend(_convert_message_content(text))
            continue

        gemini_role = role_map.get(role, "user")
        parts = _convert_message_content(msg.get("content", ""))

        # Group consecutive messages with the same role
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": gemini_role, "parts": parts})

    return (system_parts if system_parts else None, contents)


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
    """Build an HTTP request dict for Gemini generateContent.

    URL format: {base_url}/models/{model}:generateContent?key={api_key}
    For streaming: :streamGenerateContent?alt=sse&key={api_key}
    """
    base = base_url.rstrip("/")
    action = "streamGenerateContent" if stream else "generateContent"
    url = f"{base}/models/{model}:{action}?key={api_key}"
    if stream:
        url += "&alt=sse"

    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    system_parts, contents = _convert_messages(messages)

    body: dict[str, Any] = {"contents": contents}

    if system_parts:
        body["system_instruction"] = {"parts": system_parts}

    # Build generationConfig
    gen_config: dict[str, Any] = {}
    if "temperature" in kwargs:
        gen_config["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        gen_config["topP"] = kwargs["top_p"]
    if "top_k" in kwargs:
        gen_config["topK"] = kwargs["top_k"]
    if "max_tokens" in kwargs:
        gen_config["maxOutputTokens"] = kwargs["max_tokens"]
    if "n" in kwargs:
        gen_config["candidateCount"] = kwargs["n"]
    if "stop" in kwargs:
        stop = kwargs["stop"]
        gen_config["stopSequences"] = stop if isinstance(stop, list) else [stop]

    # response_format with json_object
    response_format = kwargs.get("response_format")
    if response_format and isinstance(response_format, dict):
        if response_format.get("type") == "json_object":
            gen_config["responseMimeType"] = "application/json"

    if gen_config:
        body["generationConfig"] = gen_config

    # Safety settings
    if "safety_settings" in kwargs:
        body["safetySettings"] = kwargs["safety_settings"]

    # Tools
    if "tools" in kwargs and kwargs["tools"]:
        body["tools"] = _convert_tools(kwargs["tools"])

    return {"url": url, "headers": headers, "body": body}


def parse_response(raw: dict) -> dict:
    """Parse a Gemini generateContent response into normalized dict.

    Returns:
        {
            "content": str | None,
            "finish_reason": str | None,
            "tool_calls": list[dict] | None,
            "usage": dict,
            "model": str,
        }
    """
    candidates = raw.get("candidates", [])
    candidate = candidates[0] if candidates else {}

    # Extract text from parts
    parts = candidate.get("content", {}).get("parts", [])
    text_parts = [p["text"] for p in parts if "text" in p]
    content = "".join(text_parts) if text_parts else None

    # Extract tool calls (function calls)
    tool_calls = None
    fn_parts = [p for p in parts if "functionCall" in p]
    if fn_parts:
        tool_calls = []
        for i, p in enumerate(fn_parts):
            fc = p["functionCall"]
            tool_calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": json.dumps(fc.get("args", {})),
                },
            })

    # Map finish reason
    gemini_reason = candidate.get("finishReason", "")
    finish_reason = _FINISH_REASON_MAP.get(gemini_reason, "stop")

    # Usage
    usage_meta = raw.get("usageMetadata", {})
    usage = {
        "prompt_tokens": usage_meta.get("promptTokenCount", 0),
        "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
        "total_tokens": usage_meta.get("totalTokenCount", 0),
    }

    return {
        "content": content,
        "finish_reason": finish_reason,
        "tool_calls": tool_calls,
        "usage": usage,
        "model": raw.get("modelVersion", ""),
    }


def parse_stream_chunk(line: str) -> dict | None:
    """Parse a single SSE data line from a Gemini streaming response.

    Gemini streams SSE with `data: {json}` lines. Each chunk has the
    same structure as a full response (candidates[].content.parts[].text).

    Returns an OpenAI-compatible chunk dict, or None for empty/unparseable lines.
    """
    line = line.strip()
    if not line:
        return None

    if line.startswith("data: "):
        line = line[6:]

    if line == "[DONE]":
        return None

    try:
        raw = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None

    candidates = raw.get("candidates", [])
    candidate = candidates[0] if candidates else {}

    parts = candidate.get("content", {}).get("parts", [])
    text_parts = [p["text"] for p in parts if "text" in p]
    content = "".join(text_parts) if text_parts else None

    # Map finish reason
    gemini_reason = candidate.get("finishReason")
    finish_reason = _FINISH_REASON_MAP.get(gemini_reason, None) if gemini_reason else None

    # Build OpenAI-compatible chunk
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content

    # Extract tool calls from stream
    fn_parts = [p for p in parts if "functionCall" in p]
    if fn_parts:
        tc_list = []
        for i, p in enumerate(fn_parts):
            fc = p["functionCall"]
            tc_list.append({
                "index": i,
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": json.dumps(fc.get("args", {})),
                },
            })
        delta["tool_calls"] = tc_list

    # Usage from stream chunk
    usage = None
    usage_meta = raw.get("usageMetadata")
    if usage_meta:
        usage = {
            "prompt_tokens": usage_meta.get("promptTokenCount", 0),
            "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
            "total_tokens": usage_meta.get("totalTokenCount", 0),
        }

    chunk: dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
        "model": raw.get("modelVersion", ""),
    }
    if usage:
        chunk["usage"] = usage

    return chunk

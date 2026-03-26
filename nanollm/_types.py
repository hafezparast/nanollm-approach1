"""Response types for NanoLLM.

Dataclasses with dual attribute/dict access to maintain compatibility
with litellm's response objects. Streaming chunks use plain dicts.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class _DictAccessMixin:
    """Allows dict-style access on dataclass instances."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class _AttrDict(dict):
    """A dict that also supports attribute access and __dict__.

    litellm returns token detail objects that crawl4ai accesses via
    .__dict__. This class wraps a plain dict so that both
    obj.__dict__ and obj.key work as expected.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @property
    def __dict__(self) -> dict:  # type: ignore[override]
        return dict(self)


# ── Chat completion types ─────────────────────────────────────────────


@dataclass
class Message(_DictAccessMixin):
    content: str | None = None
    role: str = "assistant"
    tool_calls: list[dict] | None = None
    function_call: dict | None = None


@dataclass
class Choice(_DictAccessMixin):
    message: Message = field(default_factory=Message)
    index: int = 0
    finish_reason: str | None = None


@dataclass
class Usage(_DictAccessMixin):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: Any = None
    prompt_tokens_details: Any = None


@dataclass
class ModelResponse(_DictAccessMixin):
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    choices: list[Choice] = field(default_factory=list)
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion"

    _hidden_params: dict = field(default_factory=dict, repr=False)


# ── Text completion types ─────────────────────────────────────────────


@dataclass
class TextChoice(_DictAccessMixin):
    text: str = ""
    index: int = 0
    finish_reason: str | None = None
    logprobs: Any = None


@dataclass
class TextCompletionResponse(_DictAccessMixin):
    id: str = field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    choices: list[TextChoice] = field(default_factory=list)
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "text_completion"

    _hidden_params: dict = field(default_factory=dict, repr=False)


# ── Embedding types ───────────────────────────────────────────────────


@dataclass
class EmbeddingData(_DictAccessMixin):
    embedding: list[float] = field(default_factory=list)
    index: int = 0
    object: str = "embedding"


@dataclass
class EmbeddingResponse(_DictAccessMixin):
    data: list[EmbeddingData] = field(default_factory=list)
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    object: str = "list"


# ── Factory helpers ───────────────────────────────────────────────────


def make_model_response(
    content: str,
    model: str = "",
    finish_reason: str = "stop",
    usage: dict | None = None,
    provider: str | None = None,
    tool_calls: list[dict] | None = None,
) -> ModelResponse:
    """Helper to build a ModelResponse from a completion result."""
    u = Usage(**(usage or {}))
    return ModelResponse(
        choices=[
            Choice(
                message=Message(content=content, role="assistant", tool_calls=tool_calls),
                index=0,
                finish_reason=finish_reason,
            )
        ],
        model=model,
        usage=u,
        _hidden_params={"custom_llm_provider": provider or ""},
    )


def make_embedding_response(
    embeddings: list[list[float]],
    model: str = "",
    usage: dict | None = None,
) -> EmbeddingResponse:
    """Helper to build an EmbeddingResponse."""
    data = [
        EmbeddingData(embedding=emb, index=i)
        for i, emb in enumerate(embeddings)
    ]
    return EmbeddingResponse(
        data=data,
        model=model,
        usage=Usage(**(usage or {})),
    )


def make_stream_chunk(
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    model: str = "",
    tool_calls: list[dict] | None = None,
) -> dict:
    """Build a streaming chunk as a plain dict (matches litellm's format)."""
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


# ── Stream chunk builder ──────────────────────────────────────────────


def stream_chunk_builder(
    chunks: list[dict],
    messages: list[dict] | None = None,
) -> ModelResponse:
    """Reassemble streaming chunks into a complete ModelResponse.

    Accumulates content, tool_calls, and usage from stream chunks
    into a single ModelResponse object.

    Args:
        chunks: List of streaming chunk dicts
        messages: Original messages (unused, kept for litellm compat)

    Returns:
        ModelResponse with accumulated content and usage
    """
    if not chunks:
        return ModelResponse()

    first = chunks[0]
    model = first.get("model", "")
    response_id = first.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")

    choice_data: dict[int, dict[str, Any]] = {}
    total_usage: dict[str, int] = {}

    for chunk in chunks:
        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]
            if isinstance(usage, dict):
                for k, v in usage.items():
                    if isinstance(v, (int, float)):
                        total_usage[k] = v

        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            if idx not in choice_data:
                choice_data[idx] = {
                    "content_parts": [],
                    "role": "assistant",
                    "tool_calls": {},
                    "finish_reason": None,
                }

            cd = choice_data[idx]
            delta = choice.get("delta", {})

            if "role" in delta:
                cd["role"] = delta["role"]
            if "content" in delta and delta["content"] is not None:
                cd["content_parts"].append(delta["content"])
            if choice.get("finish_reason"):
                cd["finish_reason"] = choice["finish_reason"]

            for tc in delta.get("tool_calls", []) or []:
                tc_idx = tc.get("index", 0)
                if tc_idx not in cd["tool_calls"]:
                    cd["tool_calls"][tc_idx] = {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    }
                existing = cd["tool_calls"][tc_idx]
                fn = tc.get("function", {})
                if fn.get("name"):
                    existing["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    existing["function"]["arguments"] += fn["arguments"]
                if tc.get("id"):
                    existing["id"] = tc["id"]

    choices = []
    for idx in sorted(choice_data.keys()):
        cd = choice_data[idx]
        content = "".join(cd["content_parts"]) or None
        tool_calls_list = (
            [cd["tool_calls"][i] for i in sorted(cd["tool_calls"].keys())]
            if cd["tool_calls"]
            else None
        )
        choices.append(
            Choice(
                message=Message(
                    content=content,
                    role=cd["role"],
                    tool_calls=tool_calls_list,
                ),
                index=idx,
                finish_reason=cd["finish_reason"],
            )
        )

    return ModelResponse(
        id=response_id,
        choices=choices,
        model=model,
        usage=Usage(**total_usage) if total_usage else Usage(),
    )

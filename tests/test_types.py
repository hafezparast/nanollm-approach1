"""Exhaustive tests for nanollm._types — response types, factories, and stream builder."""

from __future__ import annotations

import time

import pytest

from nanollm._types import (
    Choice,
    EmbeddingData,
    EmbeddingResponse,
    Message,
    ModelResponse,
    TextChoice,
    TextCompletionResponse,
    Usage,
    _AttrDict,
    _DictAccessMixin,
    make_embedding_response,
    make_model_response,
    make_stream_chunk,
    stream_chunk_builder,
)


# ════════════════════════════════════════════════════════════════════════
# _AttrDict
# ════════════════════════════════════════════════════════════════════════


class TestAttrDict:
    def test_attribute_read(self):
        d = _AttrDict({"a": 1, "b": 2})
        assert d.a == 1
        assert d.b == 2

    def test_dict_read(self):
        d = _AttrDict({"x": 42})
        assert d["x"] == 42

    def test_setattr(self):
        d = _AttrDict()
        d.foo = "bar"
        assert d["foo"] == "bar"
        assert d.foo == "bar"

    def test_missing_key_raises_attribute_error(self):
        d = _AttrDict()
        with pytest.raises(AttributeError):
            _ = d.nonexistent

    def test_dunder_dict_returns_plain_dict(self):
        d = _AttrDict({"k": "v"})
        result = d.__dict__
        assert isinstance(result, dict)
        assert result == {"k": "v"}
        # Must be a plain dict, NOT an _AttrDict
        assert type(result) is dict

    def test_dunder_dict_is_copy(self):
        d = _AttrDict({"k": 1})
        snapshot = d.__dict__
        d["k"] = 999
        # __dict__ is a new dict each time
        assert snapshot["k"] == 1

    def test_iteration(self):
        d = _AttrDict({"a": 1, "b": 2})
        assert set(d.keys()) == {"a", "b"}

    def test_len(self):
        assert len(_AttrDict({"a": 1, "b": 2, "c": 3})) == 3

    def test_empty(self):
        d = _AttrDict()
        assert len(d) == 0
        assert d.__dict__ == {}


# ════════════════════════════════════════════════════════════════════════
# _DictAccessMixin
# ════════════════════════════════════════════════════════════════════════


class TestDictAccessMixin:
    def test_getitem(self):
        msg = Message(content="hello")
        assert msg["content"] == "hello"

    def test_getitem_missing_raises(self):
        msg = Message()
        with pytest.raises(AttributeError):
            _ = msg["nonexistent"]

    def test_contains_true(self):
        msg = Message(content="hi")
        assert "content" in msg
        assert "role" in msg

    def test_contains_false(self):
        msg = Message()
        assert "banana" not in msg

    def test_get_existing(self):
        msg = Message(content="x")
        assert msg.get("content") == "x"

    def test_get_missing_default_none(self):
        msg = Message()
        assert msg.get("nonexistent") is None

    def test_get_missing_custom_default(self):
        msg = Message()
        assert msg.get("nonexistent", "fallback") == "fallback"


# ════════════════════════════════════════════════════════════════════════
# Message
# ════════════════════════════════════════════════════════════════════════


class TestMessage:
    def test_defaults(self):
        m = Message()
        assert m.content is None
        assert m.role == "assistant"
        assert m.tool_calls is None
        assert m.function_call is None

    def test_attribute_access(self):
        m = Message(content="hi", role="user")
        assert m.content == "hi"
        assert m.role == "user"

    def test_dict_access(self):
        m = Message(content="test")
        assert m["content"] == "test"
        assert m["role"] == "assistant"

    def test_get_method(self):
        m = Message(content="val")
        assert m.get("content") == "val"
        assert m.get("missing", "def") == "def"

    def test_tool_calls(self):
        tc = [{"id": "1", "function": {"name": "f", "arguments": "{}"}}]
        m = Message(tool_calls=tc)
        assert m.tool_calls == tc
        assert m["tool_calls"] == tc

    def test_function_call(self):
        fc = {"name": "fn", "arguments": "{}"}
        m = Message(function_call=fc)
        assert m.function_call == fc

    def test_content_none_explicit(self):
        m = Message(content=None)
        assert m.content is None

    def test_content_empty_string(self):
        m = Message(content="")
        assert m.content == ""

    def test_contains(self):
        m = Message(content="x")
        assert "content" in m
        assert "role" in m
        assert "nonexistent" not in m


# ════════════════════════════════════════════════════════════════════════
# Choice
# ════════════════════════════════════════════════════════════════════════


class TestChoice:
    def test_defaults(self):
        c = Choice()
        assert isinstance(c.message, Message)
        assert c.index == 0
        assert c.finish_reason is None

    def test_message_access(self):
        m = Message(content="hi")
        c = Choice(message=m, finish_reason="stop")
        assert c.message.content == "hi"
        assert c.finish_reason == "stop"

    def test_dict_access(self):
        c = Choice(index=2, finish_reason="length")
        assert c["index"] == 2
        assert c["finish_reason"] == "length"

    def test_custom_index(self):
        c = Choice(index=5)
        assert c.index == 5


# ════════════════════════════════════════════════════════════════════════
# Usage
# ════════════════════════════════════════════════════════════════════════


class TestUsage:
    def test_defaults(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.completion_tokens_details is None
        assert u.prompt_tokens_details is None

    def test_explicit_values(self):
        u = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20
        assert u.total_tokens == 30

    def test_dict_access(self):
        u = Usage(prompt_tokens=5)
        assert u["prompt_tokens"] == 5

    def test_token_details_with_attr_dict(self):
        details = _AttrDict({"reasoning_tokens": 50, "accepted_prediction_tokens": 10})
        u = Usage(completion_tokens_details=details)
        assert u.completion_tokens_details.reasoning_tokens == 50
        assert u.completion_tokens_details.__dict__ == {
            "reasoning_tokens": 50,
            "accepted_prediction_tokens": 10,
        }

    def test_token_details_none_guard_pattern(self):
        """Test the .__dict__ if x else {} pattern used by crawl4ai."""
        u = Usage(completion_tokens_details=None)
        x = u.completion_tokens_details
        result = x.__dict__ if x else {}
        assert result == {}

    def test_token_details_attr_dict_guard_pattern(self):
        details = _AttrDict({"reasoning_tokens": 100})
        u = Usage(completion_tokens_details=details)
        x = u.completion_tokens_details
        result = x.__dict__ if x else {}
        assert result == {"reasoning_tokens": 100}

    def test_prompt_tokens_details_with_attr_dict(self):
        details = _AttrDict({"cached_tokens": 25})
        u = Usage(prompt_tokens_details=details)
        assert u.prompt_tokens_details.cached_tokens == 25

    def test_prompt_tokens_details_none_guard(self):
        u = Usage(prompt_tokens_details=None)
        x = u.prompt_tokens_details
        result = x.__dict__ if x else {}
        assert result == {}

    def test_get_method(self):
        u = Usage(total_tokens=42)
        assert u.get("total_tokens") == 42
        assert u.get("nonexistent", -1) == -1


# ════════════════════════════════════════════════════════════════════════
# ModelResponse
# ════════════════════════════════════════════════════════════════════════


class TestModelResponse:
    def test_id_prefix(self):
        r = ModelResponse()
        assert r.id.startswith("chatcmpl-")

    def test_id_uniqueness(self):
        r1 = ModelResponse()
        r2 = ModelResponse()
        assert r1.id != r2.id

    def test_defaults(self):
        r = ModelResponse()
        assert r.choices == []
        assert r.model == ""
        assert isinstance(r.usage, Usage)
        assert r.object == "chat.completion"
        assert r._hidden_params == {}

    def test_created_timestamp(self):
        before = int(time.time())
        r = ModelResponse()
        after = int(time.time())
        assert before <= r.created <= after

    def test_choices_list(self):
        c = Choice(message=Message(content="hi"), finish_reason="stop")
        r = ModelResponse(choices=[c])
        assert len(r.choices) == 1
        assert r.choices[0].message.content == "hi"

    def test_model_field(self):
        r = ModelResponse(model="gpt-4o")
        assert r.model == "gpt-4o"

    def test_dict_access(self):
        r = ModelResponse(model="test")
        assert r["model"] == "test"
        assert r["object"] == "chat.completion"

    def test_hidden_params(self):
        r = ModelResponse(_hidden_params={"custom_llm_provider": "openai"})
        assert r._hidden_params["custom_llm_provider"] == "openai"

    def test_contains(self):
        r = ModelResponse()
        assert "id" in r
        assert "choices" in r
        assert "usage" in r
        assert "banana" not in r


# ════════════════════════════════════════════════════════════════════════
# TextChoice
# ════════════════════════════════════════════════════════════════════════


class TestTextChoice:
    def test_defaults(self):
        tc = TextChoice()
        assert tc.text == ""
        assert tc.index == 0
        assert tc.finish_reason is None
        assert tc.logprobs is None

    def test_explicit(self):
        tc = TextChoice(text="hello", index=1, finish_reason="stop")
        assert tc.text == "hello"
        assert tc.index == 1
        assert tc.finish_reason == "stop"

    def test_dict_access(self):
        tc = TextChoice(text="val")
        assert tc["text"] == "val"

    def test_logprobs(self):
        tc = TextChoice(logprobs={"tokens": ["a"], "offsets": [0]})
        assert tc.logprobs["tokens"] == ["a"]


# ════════════════════════════════════════════════════════════════════════
# TextCompletionResponse
# ════════════════════════════════════════════════════════════════════════


class TestTextCompletionResponse:
    def test_id_prefix(self):
        r = TextCompletionResponse()
        assert r.id.startswith("cmpl-")

    def test_object_field(self):
        r = TextCompletionResponse()
        assert r.object == "text_completion"

    def test_defaults(self):
        r = TextCompletionResponse()
        assert r.choices == []
        assert r.model == ""
        assert isinstance(r.usage, Usage)
        assert r._hidden_params == {}

    def test_created_timestamp(self):
        before = int(time.time())
        r = TextCompletionResponse()
        after = int(time.time())
        assert before <= r.created <= after

    def test_choices(self):
        tc = TextChoice(text="result", finish_reason="stop")
        r = TextCompletionResponse(choices=[tc])
        assert r.choices[0].text == "result"

    def test_dict_access(self):
        r = TextCompletionResponse(model="gpt-3.5-turbo-instruct")
        assert r["model"] == "gpt-3.5-turbo-instruct"
        assert r["object"] == "text_completion"

    def test_id_uniqueness(self):
        r1 = TextCompletionResponse()
        r2 = TextCompletionResponse()
        assert r1.id != r2.id


# ════════════════════════════════════════════════════════════════════════
# EmbeddingData
# ════════════════════════════════════════════════════════════════════════


class TestEmbeddingData:
    def test_defaults(self):
        e = EmbeddingData()
        assert e.embedding == []
        assert e.index == 0
        assert e.object == "embedding"

    def test_explicit(self):
        e = EmbeddingData(embedding=[0.1, 0.2, 0.3], index=2)
        assert e.embedding == [0.1, 0.2, 0.3]
        assert e.index == 2

    def test_dict_access(self):
        e = EmbeddingData(embedding=[1.0])
        assert e["embedding"] == [1.0]
        assert e["object"] == "embedding"


# ════════════════════════════════════════════════════════════════════════
# EmbeddingResponse
# ════════════════════════════════════════════════════════════════════════


class TestEmbeddingResponse:
    def test_defaults(self):
        r = EmbeddingResponse()
        assert r.data == []
        assert r.model == ""
        assert isinstance(r.usage, Usage)
        assert r.object == "list"

    def test_data_list(self):
        d1 = EmbeddingData(embedding=[0.1], index=0)
        d2 = EmbeddingData(embedding=[0.2], index=1)
        r = EmbeddingResponse(data=[d1, d2])
        assert len(r.data) == 2
        assert r.data[0].embedding == [0.1]
        assert r.data[1].index == 1

    def test_dict_access(self):
        r = EmbeddingResponse(model="text-embedding-3-small")
        assert r["model"] == "text-embedding-3-small"
        assert r["object"] == "list"


# ════════════════════════════════════════════════════════════════════════
# make_model_response
# ════════════════════════════════════════════════════════════════════════


class TestMakeModelResponse:
    def test_basic(self):
        r = make_model_response("Hello!")
        assert r.choices[0].message.content == "Hello!"
        assert r.choices[0].message.role == "assistant"
        assert r.choices[0].finish_reason == "stop"
        assert r.choices[0].index == 0

    def test_model_field(self):
        r = make_model_response("x", model="gpt-4o")
        assert r.model == "gpt-4o"

    def test_finish_reason(self):
        r = make_model_response("x", finish_reason="length")
        assert r.choices[0].finish_reason == "length"

    def test_with_usage(self):
        u = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        r = make_model_response("x", usage=u)
        assert r.usage.prompt_tokens == 10
        assert r.usage.completion_tokens == 20
        assert r.usage.total_tokens == 30

    def test_without_usage(self):
        r = make_model_response("x")
        assert r.usage.prompt_tokens == 0
        assert r.usage.completion_tokens == 0
        assert r.usage.total_tokens == 0

    def test_with_provider(self):
        r = make_model_response("x", provider="anthropic")
        assert r._hidden_params["custom_llm_provider"] == "anthropic"

    def test_without_provider(self):
        r = make_model_response("x")
        assert r._hidden_params["custom_llm_provider"] == ""

    def test_with_tool_calls(self):
        tc = [{"id": "call_1", "function": {"name": "search", "arguments": '{"q":"hi"}'}}]
        r = make_model_response("", tool_calls=tc)
        assert r.choices[0].message.tool_calls == tc

    def test_without_tool_calls(self):
        r = make_model_response("hi")
        assert r.choices[0].message.tool_calls is None

    def test_id_prefix(self):
        r = make_model_response("x")
        assert r.id.startswith("chatcmpl-")

    def test_object_field(self):
        r = make_model_response("x")
        assert r.object == "chat.completion"

    def test_created_is_recent(self):
        before = int(time.time())
        r = make_model_response("x")
        after = int(time.time())
        assert before <= r.created <= after

    def test_single_choice(self):
        r = make_model_response("x")
        assert len(r.choices) == 1


# ════════════════════════════════════════════════════════════════════════
# make_embedding_response
# ════════════════════════════════════════════════════════════════════════


class TestMakeEmbeddingResponse:
    def test_single_embedding(self):
        r = make_embedding_response([[0.1, 0.2, 0.3]])
        assert len(r.data) == 1
        assert r.data[0].embedding == [0.1, 0.2, 0.3]
        assert r.data[0].index == 0
        assert r.data[0].object == "embedding"

    def test_multiple_embeddings(self):
        embs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        r = make_embedding_response(embs)
        assert len(r.data) == 3
        for i, emb in enumerate(embs):
            assert r.data[i].embedding == emb
            assert r.data[i].index == i

    def test_empty_embeddings(self):
        r = make_embedding_response([])
        assert len(r.data) == 0

    def test_model_field(self):
        r = make_embedding_response([[0.1]], model="text-embedding-3-small")
        assert r.model == "text-embedding-3-small"

    def test_with_usage(self):
        u = {"prompt_tokens": 5, "total_tokens": 5}
        r = make_embedding_response([[0.1]], usage=u)
        assert r.usage.prompt_tokens == 5
        assert r.usage.total_tokens == 5

    def test_without_usage(self):
        r = make_embedding_response([[0.1]])
        assert r.usage.prompt_tokens == 0

    def test_object_field(self):
        r = make_embedding_response([[0.1]])
        assert r.object == "list"


# ════════════════════════════════════════════════════════════════════════
# make_stream_chunk
# ════════════════════════════════════════════════════════════════════════


class TestMakeStreamChunk:
    def test_content_only(self):
        c = make_stream_chunk(content="hi")
        assert c["choices"][0]["delta"]["content"] == "hi"
        assert "role" not in c["choices"][0]["delta"]
        assert c["choices"][0]["finish_reason"] is None

    def test_role_only(self):
        c = make_stream_chunk(role="assistant")
        assert c["choices"][0]["delta"]["role"] == "assistant"
        assert "content" not in c["choices"][0]["delta"]

    def test_finish_reason(self):
        c = make_stream_chunk(finish_reason="stop")
        assert c["choices"][0]["finish_reason"] == "stop"

    def test_model(self):
        c = make_stream_chunk(model="gpt-4o")
        assert c["model"] == "gpt-4o"

    def test_tool_calls(self):
        tc = [{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": ""}}]
        c = make_stream_chunk(tool_calls=tc)
        assert c["choices"][0]["delta"]["tool_calls"] == tc

    def test_empty_delta(self):
        c = make_stream_chunk()
        assert c["choices"][0]["delta"] == {}

    def test_id_prefix(self):
        c = make_stream_chunk()
        assert c["id"].startswith("chatcmpl-")

    def test_object_field(self):
        c = make_stream_chunk()
        assert c["object"] == "chat.completion.chunk"

    def test_created(self):
        before = int(time.time())
        c = make_stream_chunk()
        after = int(time.time())
        assert before <= c["created"] <= after

    def test_single_choice(self):
        c = make_stream_chunk(content="x")
        assert len(c["choices"]) == 1
        assert c["choices"][0]["index"] == 0

    def test_content_and_role_together(self):
        c = make_stream_chunk(content="hi", role="assistant")
        delta = c["choices"][0]["delta"]
        assert delta["content"] == "hi"
        assert delta["role"] == "assistant"


# ════════════════════════════════════════════════════════════════════════
# stream_chunk_builder
# ════════════════════════════════════════════════════════════════════════


class TestStreamChunkBuilder:
    def test_empty_chunks(self):
        r = stream_chunk_builder([])
        assert isinstance(r, ModelResponse)
        assert r.choices == []

    def test_single_chunk_with_content(self):
        chunk = make_stream_chunk(content="hello", role="assistant", model="gpt-4o")
        r = stream_chunk_builder([chunk])
        assert r.choices[0].message.content == "hello"
        assert r.choices[0].message.role == "assistant"
        assert r.model == "gpt-4o"

    def test_multi_chunk_content_assembly(self):
        chunks = [
            make_stream_chunk(content="Hello", role="assistant", model="gpt-4o"),
            make_stream_chunk(content=" "),
            make_stream_chunk(content="world!"),
            make_stream_chunk(finish_reason="stop"),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.content == "Hello world!"
        assert r.choices[0].finish_reason == "stop"

    def test_no_content_produces_none(self):
        chunks = [
            make_stream_chunk(role="assistant"),
            make_stream_chunk(finish_reason="stop"),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.content is None

    def test_uses_first_chunk_id(self):
        c1 = make_stream_chunk(content="a")
        c2 = make_stream_chunk(content="b")
        r = stream_chunk_builder([c1, c2])
        assert r.id == c1["id"]

    def test_uses_first_chunk_model(self):
        c1 = make_stream_chunk(content="a", model="gpt-4o")
        c2 = make_stream_chunk(content="b", model="gpt-4o-mini")
        r = stream_chunk_builder([c1, c2])
        assert r.model == "gpt-4o"

    def test_usage_from_final_chunk(self):
        c1 = make_stream_chunk(content="hello")
        c2 = make_stream_chunk(finish_reason="stop")
        c2["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        r = stream_chunk_builder([c1, c2])
        assert r.usage.prompt_tokens == 10
        assert r.usage.completion_tokens == 5
        assert r.usage.total_tokens == 15

    def test_no_usage_defaults_to_zero(self):
        chunks = [make_stream_chunk(content="hi")]
        r = stream_chunk_builder(chunks)
        assert r.usage.prompt_tokens == 0
        assert r.usage.completion_tokens == 0
        assert r.usage.total_tokens == 0

    def test_tool_calls_accumulation(self):
        chunks = [
            make_stream_chunk(
                role="assistant",
                tool_calls=[{
                    "index": 0,
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "search", "arguments": ""},
                }],
            ),
            make_stream_chunk(
                tool_calls=[{
                    "index": 0,
                    "function": {"name": "", "arguments": '{"q":'},
                }],
            ),
            make_stream_chunk(
                tool_calls=[{
                    "index": 0,
                    "function": {"name": "", "arguments": '"hello"}'},
                }],
            ),
            make_stream_chunk(finish_reason="tool_calls"),
        ]
        r = stream_chunk_builder(chunks)
        tc = r.choices[0].message.tool_calls
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["id"] == "call_abc"
        assert tc[0]["function"]["name"] == "search"
        assert tc[0]["function"]["arguments"] == '{"q":"hello"}'

    def test_multiple_tool_calls_different_indices(self):
        chunks = [
            make_stream_chunk(
                role="assistant",
                tool_calls=[{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "func_a", "arguments": "{}"},
                }],
            ),
            make_stream_chunk(
                tool_calls=[{
                    "index": 1,
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "func_b", "arguments": "{}"},
                }],
            ),
            make_stream_chunk(finish_reason="tool_calls"),
        ]
        r = stream_chunk_builder(chunks)
        tc = r.choices[0].message.tool_calls
        assert len(tc) == 2
        assert tc[0]["function"]["name"] == "func_a"
        assert tc[1]["function"]["name"] == "func_b"
        assert tc[0]["id"] == "call_1"
        assert tc[1]["id"] == "call_2"

    def test_no_tool_calls_produces_none(self):
        chunks = [
            make_stream_chunk(content="hi", role="assistant"),
            make_stream_chunk(finish_reason="stop"),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.tool_calls is None

    def test_multiple_choices(self):
        """Chunks with different choice indices produce multiple choices."""
        chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "gpt-4o",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "A"}, "finish_reason": None},
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "gpt-4o",
                "choices": [
                    {"index": 1, "delta": {"role": "assistant", "content": "B"}, "finish_reason": None},
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "gpt-4o",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                    {"index": 1, "delta": {}, "finish_reason": "stop"},
                ],
            },
        ]
        r = stream_chunk_builder(chunks)
        assert len(r.choices) == 2
        assert r.choices[0].message.content == "A"
        assert r.choices[0].index == 0
        assert r.choices[1].message.content == "B"
        assert r.choices[1].index == 1

    def test_messages_param_ignored(self):
        """The messages parameter is accepted but unused (litellm compat)."""
        chunks = [make_stream_chunk(content="hi")]
        r = stream_chunk_builder(chunks, messages=[{"role": "user", "content": "test"}])
        assert r.choices[0].message.content == "hi"

    def test_usage_ignores_non_numeric(self):
        """Non-numeric usage values are skipped."""
        c = make_stream_chunk(content="x")
        c["usage"] = {"prompt_tokens": 10, "extra_info": "not_a_number"}
        r = stream_chunk_builder([c])
        assert r.usage.prompt_tokens == 10

    def test_usage_none_ignored(self):
        """usage=None in a chunk is skipped."""
        c = make_stream_chunk(content="x")
        c["usage"] = None
        r = stream_chunk_builder([c])
        assert r.usage.prompt_tokens == 0

    def test_tool_call_id_updated_later(self):
        """If tool call id arrives in a later chunk, it overwrites."""
        chunks = [
            make_stream_chunk(
                tool_calls=[{
                    "index": 0,
                    "id": "",
                    "type": "function",
                    "function": {"name": "f", "arguments": ""},
                }],
            ),
            make_stream_chunk(
                tool_calls=[{
                    "index": 0,
                    "id": "call_final",
                    "function": {"name": "", "arguments": "{}"},
                }],
            ),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.tool_calls[0]["id"] == "call_final"

    def test_finish_reason_from_last_occurrence(self):
        """finish_reason is taken from the last chunk that sets it."""
        chunks = [
            make_stream_chunk(content="hi"),
            make_stream_chunk(finish_reason="stop"),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].finish_reason == "stop"

    def test_role_defaults_to_assistant(self):
        """If no role is in any delta, default is assistant."""
        c = {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }
        r = stream_chunk_builder([c])
        assert r.choices[0].message.role == "assistant"

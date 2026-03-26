"""Exhaustive tests for litellm compatibility shim.

Covers: import patterns, exception classes, drop_params/set_verbose proxy,
response access patterns, streaming chunk access, Phase 1 exports,
type exports, and provider string parsing.
"""

from __future__ import annotations

import importlib
import sys

import pytest


# ── Import patterns used by crawl4ai ──────────────────────────────────


class TestImportPatterns:
    def test_from_nanollm_import_completion(self):
        from nanollm import completion
        assert callable(completion)

    def test_from_nanollm_import_acompletion(self):
        from nanollm import acompletion
        assert callable(acompletion)

    def test_from_litellm_import_completion(self):
        from litellm import completion
        assert callable(completion)

    def test_from_litellm_import_acompletion(self):
        from litellm import acompletion
        assert callable(acompletion)

    def test_import_litellm_dot_completion(self):
        import litellm
        assert callable(litellm.completion)

    def test_import_litellm_dot_acompletion(self):
        import litellm
        assert callable(litellm.acompletion)

    def test_from_litellm_import_batch_completion(self):
        from litellm import batch_completion
        assert callable(batch_completion)

    def test_from_litellm_import_embedding(self):
        from litellm import embedding
        assert callable(embedding)

    def test_from_litellm_import_aembedding(self):
        from litellm import aembedding
        assert callable(aembedding)

    def test_from_litellm_import_text_completion(self):
        from litellm import text_completion
        assert callable(text_completion)

    def test_from_litellm_import_atext_completion(self):
        from litellm import atext_completion
        assert callable(atext_completion)

    def test_from_litellm_import_stream_chunk_builder(self):
        from litellm import stream_chunk_builder
        assert callable(stream_chunk_builder)


# ── Exception classes importable from litellm.exceptions ──────────────


class TestExceptionImports:
    EXCEPTION_NAMES = [
        "APIConnectionError",
        "APIError",
        "AuthenticationError",
        "BadGatewayError",
        "BadRequestError",
        "BudgetExceededError",
        "ContentPolicyViolationError",
        "ContextWindowExceededError",
        "InternalServerError",
        "InvalidRequestError",
        "JSONSchemaValidationError",
        "NanoLLMException",
        "NotFoundError",
        "OpenAIError",
        "PermissionDeniedError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
        "UnsupportedParamsError",
    ]

    @pytest.mark.parametrize("name", EXCEPTION_NAMES)
    def test_exception_importable_from_litellm_exceptions(self, name):
        from litellm import exceptions
        cls = getattr(exceptions, name)
        assert isinstance(cls, type)

    @pytest.mark.parametrize("name", EXCEPTION_NAMES)
    def test_exception_importable_from_litellm_top_level(self, name):
        import litellm
        cls = getattr(litellm, name)
        assert isinstance(cls, type)

    def test_all_19_exceptions_in_litellm_exceptions(self):
        from litellm import exceptions
        assert len(exceptions.__all__) == 19

    def test_openai_error_is_alias_for_nanollm_exception(self):
        from litellm.exceptions import OpenAIError, NanoLLMException
        assert OpenAIError is NanoLLMException

    def test_bad_request_is_subclass_of_invalid_request(self):
        from litellm.exceptions import BadRequestError, InvalidRequestError
        assert issubclass(BadRequestError, InvalidRequestError)

    def test_internal_server_error_is_subclass_of_api_error(self):
        from litellm.exceptions import InternalServerError, APIError
        assert issubclass(InternalServerError, APIError)

    def test_context_window_is_subclass_of_invalid_request(self):
        from litellm.exceptions import ContextWindowExceededError, InvalidRequestError
        assert issubclass(ContextWindowExceededError, InvalidRequestError)


# ── drop_params/set_verbose proxy ────────────────────────────────────


class TestModuleLevelProxy:
    def test_drop_params_read_from_litellm(self):
        import litellm
        import nanollm
        original = nanollm.drop_params
        try:
            nanollm.drop_params = True
            assert litellm.drop_params is True
            nanollm.drop_params = False
            assert litellm.drop_params is False
        finally:
            nanollm.drop_params = original

    def test_drop_params_set_from_litellm(self):
        import litellm
        import nanollm
        original = nanollm.drop_params
        try:
            litellm.drop_params = True
            assert nanollm.drop_params is True
            litellm.drop_params = False
            assert nanollm.drop_params is False
        finally:
            nanollm.drop_params = original

    def test_set_verbose_read_from_litellm(self):
        import litellm
        import nanollm
        original = nanollm.set_verbose
        try:
            nanollm.set_verbose = True
            assert litellm.set_verbose is True
        finally:
            nanollm.set_verbose = original

    def test_set_verbose_set_from_litellm(self):
        import litellm
        import nanollm
        original = nanollm.set_verbose
        try:
            litellm.set_verbose = True
            assert nanollm.set_verbose is True
        finally:
            nanollm.set_verbose = original


# ── Response access patterns ──────────────────────────────────────────


class TestResponseAccessPatterns:
    def test_model_response_attribute_access(self):
        from nanollm._types import make_model_response
        resp = make_model_response(content="Hello", model="gpt-4o")
        assert resp.choices[0].message.content == "Hello"

    def test_model_response_dict_access(self):
        from nanollm._types import make_model_response
        resp = make_model_response(content="Hello", model="gpt-4o")
        assert resp["choices"][0]["message"]["content"] == "Hello"

    def test_usage_attribute_access(self):
        from nanollm._types import make_model_response
        resp = make_model_response(
            content="Hi",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.usage.total_tokens == 15

    def test_usage_dict_access(self):
        from nanollm._types import make_model_response
        resp = make_model_response(
            content="Hi",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert resp["usage"]["prompt_tokens"] == 10

    def test_finish_reason_access(self):
        from nanollm._types import make_model_response
        resp = make_model_response(content="Hi", finish_reason="stop")
        assert resp.choices[0].finish_reason == "stop"

    def test_model_field(self):
        from nanollm._types import make_model_response
        resp = make_model_response(content="Hi", model="gpt-4o")
        assert resp.model == "gpt-4o"

    def test_tool_calls_access(self):
        from nanollm._types import make_model_response
        tc = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        resp = make_model_response(content=None, tool_calls=tc)
        assert resp.choices[0].message.tool_calls == tc

    def test_hidden_params(self):
        from nanollm._types import make_model_response
        resp = make_model_response(content="Hi", provider="openai")
        assert resp._hidden_params["custom_llm_provider"] == "openai"

    def test_get_method_with_default(self):
        from nanollm._types import make_model_response
        resp = make_model_response(content="Hi")
        assert resp.get("nonexistent", "default") == "default"

    def test_contains_check(self):
        from nanollm._types import make_model_response
        resp = make_model_response(content="Hi")
        assert "choices" in resp
        assert "nonexistent" not in resp


# ── Streaming chunk access patterns ──────────────────────────────────


class TestStreamingChunkAccess:
    def test_chunk_dict_access(self):
        from nanollm._types import make_stream_chunk
        chunk = make_stream_chunk(content="Hi")
        assert chunk["choices"][0]["delta"].get("content") == "Hi"

    def test_chunk_finish_reason(self):
        from nanollm._types import make_stream_chunk
        chunk = make_stream_chunk(finish_reason="stop")
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_chunk_role(self):
        from nanollm._types import make_stream_chunk
        chunk = make_stream_chunk(role="assistant")
        assert chunk["choices"][0]["delta"]["role"] == "assistant"

    def test_chunk_tool_calls(self):
        from nanollm._types import make_stream_chunk
        tc = [{"index": 0, "function": {"arguments": '{"x'}}]
        chunk = make_stream_chunk(tool_calls=tc)
        assert chunk["choices"][0]["delta"]["tool_calls"] == tc

    def test_chunk_none_content_omitted(self):
        from nanollm._types import make_stream_chunk
        chunk = make_stream_chunk()
        assert "content" not in chunk["choices"][0]["delta"]


# ── stream_chunk_builder ──────────────────────────────────────────────


class TestStreamChunkBuilder:
    def test_assembles_content(self):
        from nanollm._types import stream_chunk_builder
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "Hel"}}], "model": "gpt-4o", "id": "x"},
            {"choices": [{"index": 0, "delta": {"content": "lo"}}], "model": "gpt-4o", "id": "x"},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "model": "gpt-4o", "id": "x"},
        ]
        result = stream_chunk_builder(chunks)
        assert result.choices[0].message.content == "Hello"
        assert result.choices[0].finish_reason == "stop"

    def test_assembles_tool_calls(self):
        from nanollm._types import stream_chunk_builder
        chunks = [
            {
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "fn", "arguments": ""}}]
                }}],
                "id": "x",
            },
            {
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": '{"k":'}}]
                }}],
                "id": "x",
            },
            {
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": '"v"}'}}]
                }}],
                "id": "x",
            },
        ]
        result = stream_chunk_builder(chunks)
        tc = result.choices[0].message.tool_calls
        assert tc[0]["function"]["name"] == "fn"
        assert tc[0]["function"]["arguments"] == '{"k":"v"}'

    def test_empty_chunks(self):
        from nanollm._types import stream_chunk_builder
        result = stream_chunk_builder([])
        assert result.choices == []

    def test_usage_from_chunks(self):
        from nanollm._types import stream_chunk_builder
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "x"}}], "id": "x"},
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "id": "x",
            },
        ]
        result = stream_chunk_builder(chunks)
        assert result.usage.prompt_tokens == 10
        assert result.usage.total_tokens == 15


# ── Phase 1 exports ──────────────────────────────────────────────────


class TestPhase1Exports:
    def test_embedding_from_litellm(self):
        from litellm import embedding
        assert callable(embedding)

    def test_text_completion_from_litellm(self):
        from litellm import text_completion
        assert callable(text_completion)

    def test_stream_chunk_builder_from_litellm(self):
        from litellm import stream_chunk_builder
        assert callable(stream_chunk_builder)

    def test_aembedding_from_litellm(self):
        from litellm import aembedding
        assert callable(aembedding)

    def test_atext_completion_from_litellm(self):
        from litellm import atext_completion
        assert callable(atext_completion)


# ── Type exports ──────────────────────────────────────────────────────


class TestTypeExports:
    def test_model_response_from_litellm(self):
        from litellm import ModelResponse
        assert ModelResponse is not None

    def test_text_completion_response_from_litellm(self):
        from litellm import TextCompletionResponse
        assert TextCompletionResponse is not None

    def test_embedding_response_from_litellm(self):
        from litellm import EmbeddingResponse
        assert EmbeddingResponse is not None

    def test_model_response_instantiable(self):
        from litellm import ModelResponse
        resp = ModelResponse()
        assert resp.object == "chat.completion"

    def test_text_completion_response_instantiable(self):
        from litellm import TextCompletionResponse
        resp = TextCompletionResponse()
        assert resp.object == "text_completion"

    def test_embedding_response_instantiable(self):
        from litellm import EmbeddingResponse
        resp = EmbeddingResponse()
        assert resp.object == "list"


# ── AttrDict ──────────────────────────────────────────────────────────


class TestAttrDict:
    def test_attribute_access(self):
        from nanollm._types import _AttrDict
        d = _AttrDict({"foo": 1, "bar": 2})
        assert d.foo == 1
        assert d.bar == 2

    def test_dict_access(self):
        from nanollm._types import _AttrDict
        d = _AttrDict({"foo": 1})
        assert d["foo"] == 1

    def test_dunder_dict(self):
        from nanollm._types import _AttrDict
        d = _AttrDict({"reasoning_tokens": 3})
        assert d.__dict__ == {"reasoning_tokens": 3}

    def test_set_attribute(self):
        from nanollm._types import _AttrDict
        d = _AttrDict()
        d.foo = 42
        assert d["foo"] == 42

    def test_missing_key_raises_attribute_error(self):
        from nanollm._types import _AttrDict
        d = _AttrDict()
        with pytest.raises(AttributeError):
            _ = d.nonexistent


# ── Provider string parsing ──────────────────────────────────────────


class TestProviderStringParsing:
    PROVIDERS = [
        ("openai/gpt-4o", "openai", "gpt-4o"),
        ("anthropic/claude-3-opus", "anthropic", "claude-3-opus"),
        ("gemini/gemini-pro", "gemini", "gemini-pro"),
        ("groq/llama-3-70b", "groq", "llama-3-70b"),
        ("together/meta-llama/llama-3-70b", "together", "meta-llama/llama-3-70b"),
        ("ollama/llama3", "ollama", "llama3"),
        ("mistral/mistral-large", "mistral", "mistral-large"),
        ("deepseek/deepseek-chat", "deepseek", "deepseek-chat"),
        ("perplexity/pplx-70b-chat", "perplexity", "pplx-70b-chat"),
        ("fireworks/accounts/fireworks/models/llama", "fireworks", "accounts/fireworks/models/llama"),
        ("openrouter/meta-llama/llama-3-70b", "openrouter", "meta-llama/llama-3-70b"),
        ("deepinfra/meta-llama/llama-3-70b", "deepinfra", "meta-llama/llama-3-70b"),
        ("azure/gpt-4", "azure", "gpt-4"),
        ("bedrock/anthropic.claude-v2", "bedrock", "anthropic.claude-v2"),
        ("vertex_ai/gemini-pro", "vertex_ai", "gemini-pro"),
        ("xai/grok-1", "xai", "grok-1"),
        ("cerebras/llama3.1-8b", "cerebras", "llama3.1-8b"),
    ]

    @pytest.mark.parametrize("model_str,expected_provider,expected_model", PROVIDERS)
    def test_parse_model_string(self, model_str, expected_provider, expected_model):
        from nanollm._router import parse_model_string
        provider, model_id = parse_model_string(model_str)
        assert provider == expected_provider
        assert model_id == expected_model

    def test_no_slash_defaults_to_openai(self):
        from nanollm._router import parse_model_string
        provider, model_id = parse_model_string("gpt-4o")
        assert provider == "openai"
        assert model_id == "gpt-4o"

    def test_trailing_slash(self):
        from nanollm._router import parse_model_string
        provider, model_id = parse_model_string("openai/")
        assert provider == "openai"
        assert model_id == ""


# ── Provider config lookup ────────────────────────────────────────────


class TestProviderConfigLookup:
    def test_known_providers(self):
        from nanollm._config import get_provider_config
        for name in ["openai", "anthropic", "gemini", "groq", "together",
                      "ollama", "azure", "bedrock", "vertex_ai"]:
            cfg = get_provider_config(name)
            assert cfg.name is not None

    def test_unknown_provider_raises(self):
        from nanollm._config import get_provider_config
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider_config("nonexistent_provider")

    def test_together_ai_alias(self):
        from nanollm._config import get_provider_config
        cfg = get_provider_config("together_ai")
        assert cfg.name == "together"

    def test_ollama_chat_alias(self):
        from nanollm._config import get_provider_config
        cfg = get_provider_config("ollama_chat")
        assert cfg.name == "ollama"

"""Exhaustive tests for all 7 nanollm adapters.

Tests build_request, parse_response, parse_stream_chunk, and embedding
functions for each adapter using realistic mock data.
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock
import os

import pytest

from nanollm.adapters import openai_compat, anthropic, gemini, ollama, azure_openai, bedrock, vertex


# ═══════════════════════════════════════════════════════════════════════
# OpenAI-Compatible Adapter
# ═══════════════════════════════════════════════════════════════════════


class TestOpenAICompatBuildRequest:
    def test_basic_url_construction(self):
        result = openai_compat.build_request(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result["url"] == "https://api.openai.com/v1/chat/completions"

    def test_trailing_slash_stripped(self):
        result = openai_compat.build_request(
            base_url="https://api.openai.com/v1/",
            api_key="sk-test",
            model="gpt-4o",
            messages=[],
        )
        assert result["url"] == "https://api.openai.com/v1/chat/completions"

    def test_headers_include_auth(self):
        result = openai_compat.build_request(
            base_url="https://api.openai.com/v1",
            api_key="sk-test123",
            model="gpt-4o",
            messages=[],
        )
        assert result["headers"]["Authorization"] == "Bearer sk-test123"
        assert result["headers"]["Content-Type"] == "application/json"

    def test_body_contains_model_and_messages(self):
        msgs = [{"role": "user", "content": "Hi"}]
        result = openai_compat.build_request(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            messages=msgs,
        )
        assert result["body"]["model"] == "gpt-4o"
        assert result["body"]["messages"] == msgs

    def test_stream_false_by_default(self):
        result = openai_compat.build_request(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            messages=[],
        )
        assert result["body"]["stream"] is False

    def test_stream_true_adds_stream_options(self):
        result = openai_compat.build_request(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            messages=[],
            stream=True,
        )
        assert result["body"]["stream"] is True
        assert result["body"]["stream_options"] == {"include_usage": True}

    def test_extra_headers_merged(self):
        result = openai_compat.build_request(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            messages=[],
            extra_headers={"X-Custom": "value"},
        )
        assert result["headers"]["X-Custom"] == "value"

    def test_chat_params_forwarded(self):
        result = openai_compat.build_request(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            messages=[],
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        assert result["body"]["temperature"] == 0.5
        assert result["body"]["top_p"] == 0.9
        assert result["body"]["max_tokens"] == 100
        assert len(result["body"]["tools"]) == 1

    def test_multimodal_messages_passthrough(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "https://img.jpg"}},
                ],
            }
        ]
        result = openai_compat.build_request(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            messages=msgs,
        )
        assert result["body"]["messages"] == msgs

    def test_response_format_forwarded(self):
        result = openai_compat.build_request(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            messages=[],
            response_format={"type": "json_object"},
        )
        assert result["body"]["response_format"] == {"type": "json_object"}


class TestOpenAICompatParseResponse:
    def test_extract_content(self):
        raw = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o",
        }
        result = openai_compat.parse_response(raw)
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["model"] == "gpt-4o"

    def test_extract_usage(self):
        raw = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = openai_compat.parse_response(raw)
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_extract_tool_calls(self):
        raw = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = openai_compat.parse_response(raw)
        assert result["content"] is None
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_empty_choices(self):
        raw = {"choices": [{}], "usage": {}}
        result = openai_compat.parse_response(raw)
        assert result["content"] is None
        assert result["finish_reason"] is None

    def test_completion_tokens_details_wrapped_as_attrdict(self):
        raw = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "completion_tokens_details": {"reasoning_tokens": 3},
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        }
        result = openai_compat.parse_response(raw)
        details = result["usage"]["completion_tokens_details"]
        assert details.reasoning_tokens == 3
        assert details.__dict__ == {"reasoning_tokens": 3}
        prompt_details = result["usage"]["prompt_tokens_details"]
        assert prompt_details.cached_tokens == 2


class TestOpenAICompatParseStreamChunk:
    def test_content_delta(self):
        chunk_json = json.dumps({
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        })
        result = openai_compat.parse_stream_chunk(chunk_json)
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_finish_delta(self):
        chunk_json = json.dumps({
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        })
        result = openai_compat.parse_stream_chunk(chunk_json)
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_done_sentinel(self):
        assert openai_compat.parse_stream_chunk("[DONE]") is None

    def test_empty_line(self):
        assert openai_compat.parse_stream_chunk("") is None

    def test_data_prefix_stripped(self):
        chunk_json = json.dumps({"choices": [{"delta": {"content": "hi"}}]})
        line = f"data: {chunk_json}"
        result = openai_compat.parse_stream_chunk(line)
        assert result["choices"][0]["delta"]["content"] == "hi"

    def test_invalid_json_returns_none(self):
        assert openai_compat.parse_stream_chunk("{bad json}") is None

    def test_tool_call_delta(self):
        chunk_json = json.dumps({
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"ci'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        })
        result = openai_compat.parse_stream_chunk(chunk_json)
        assert result["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"ci'


class TestOpenAICompatEmbedding:
    def test_build_embedding_request(self):
        result = openai_compat.build_embedding_request(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="text-embedding-ada-002",
            input=["Hello world"],
        )
        assert result["url"] == "https://api.openai.com/v1/embeddings"
        assert result["body"]["model"] == "text-embedding-ada-002"
        assert result["body"]["input"] == ["Hello world"]

    def test_parse_embedding_response(self):
        raw = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"},
            ],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
            "model": "text-embedding-ada-002",
        }
        result = openai_compat.parse_embedding_response(raw)
        assert result["embeddings"] == [[0.1, 0.2, 0.3]]
        assert result["usage"]["prompt_tokens"] == 5
        assert result["model"] == "text-embedding-ada-002"


# ═══════════════════════════════════════════════════════════════════════
# Anthropic Adapter
# ═══════════════════════════════════════════════════════════════════════


class TestAnthropicBuildRequest:
    def test_url_construction(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="sk-ant-test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result["url"] == "https://api.anthropic.com/v1/messages"

    def test_headers_use_x_api_key(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="sk-ant-test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result["headers"]["x-api-key"] == "sk-ant-test"
        assert result["headers"]["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in result["headers"]

    def test_system_message_extracted(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert result["body"]["system"] == "You are helpful"
        assert len(result["body"]["messages"]) == 1

    def test_max_tokens_default(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result["body"]["max_tokens"] == 4096

    def test_max_tokens_override(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1000,
        )
        assert result["body"]["max_tokens"] == 1000

    def test_stop_sequences_from_string(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
            stop="\n",
        )
        assert result["body"]["stop_sequences"] == ["\n"]

    def test_stop_sequences_from_list(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
            stop=["\n", "END"],
        )
        assert result["body"]["stop_sequences"] == ["\n", "END"]

    def test_tools_converted(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        assert result["body"]["tools"][0]["name"] == "get_weather"
        assert "input_schema" in result["body"]["tools"][0]

    def test_response_format_json_adds_system_instruction(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
            response_format={"type": "json_object"},
        )
        assert "JSON" in result["body"]["system"]

    @patch("nanollm._image.download_image_as_base64")
    def test_multimodal_messages_converted(self, mock_dl):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "https://img.jpg"}},
                ],
            }
        ]
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=msgs,
        )
        content = result["body"]["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"

    def test_stream_flag(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        assert result["body"]["stream"] is True

    def test_tool_choice_auto(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "f1"}}],
            tool_choice="auto",
        )
        assert result["body"]["tool_choice"] == {"type": "auto"}

    def test_tool_choice_none_removes_tools(self):
        result = anthropic.build_request(
            base_url="https://api.anthropic.com/v1",
            api_key="test",
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "f1"}}],
            tool_choice="none",
        )
        assert "tools" not in result["body"]


class TestAnthropicParseResponse:
    def test_text_response(self):
        raw = {
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-3-opus-20240229",
        }
        result = anthropic.parse_response(raw)
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_tool_use_response(self):
        raw = {
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "model": "claude-3-opus-20240229",
        }
        result = anthropic.parse_response(raw)
        assert result["content"] == "Let me check the weather."
        assert result["finish_reason"] == "tool_calls"
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"city": "NYC"}

    def test_empty_content(self):
        raw = {"content": [], "stop_reason": "end_turn", "usage": {}}
        result = anthropic.parse_response(raw)
        assert result["content"] is None
        assert result["tool_calls"] is None


class TestAnthropicParseStreamChunk:
    def test_content_block_delta(self):
        line = 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}'
        result = anthropic.parse_stream_chunk(line)
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_message_delta_with_stop(self):
        line = 'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}'
        result = anthropic.parse_stream_chunk(line)
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_message_start(self):
        line = 'data: {"type":"message_start","message":{"model":"claude-3","usage":{"input_tokens":10,"output_tokens":0}}}'
        result = anthropic.parse_stream_chunk(line)
        assert result["choices"][0]["delta"]["role"] == "assistant"
        assert result["model"] == "claude-3"

    def test_event_line_skipped(self):
        assert anthropic.parse_stream_chunk("event: content_block_delta") is None

    def test_empty_line_skipped(self):
        assert anthropic.parse_stream_chunk("") is None

    def test_message_stop_returns_none(self):
        line = 'data: {"type":"message_stop"}'
        assert anthropic.parse_stream_chunk(line) is None

    def test_content_block_start_tool_use(self):
        line = 'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather"}}'
        result = anthropic.parse_stream_chunk(line)
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["id"] == "toolu_01"

    def test_input_json_delta(self):
        line = 'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\""}}'
        result = anthropic.parse_stream_chunk(line)
        tc = result["choices"][0]["delta"]["tool_calls"][0]
        assert '{"city"' in tc["function"]["arguments"]


# ═══════════════════════════════════════════════════════════════════════
# Gemini Adapter
# ═══════════════════════════════════════════════════════════════════════


class TestGeminiBuildRequest:
    def test_url_construction_non_stream(self):
        result = gemini.build_request(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="gemini-key",
            model="gemini-pro",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert "generateContent" in result["url"]
        assert "key=gemini-key" in result["url"]
        assert "streamGenerateContent" not in result["url"]

    def test_url_construction_stream(self):
        result = gemini.build_request(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="gemini-key",
            model="gemini-pro",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        assert "streamGenerateContent" in result["url"]
        assert "alt=sse" in result["url"]

    def test_system_instruction_extracted(self):
        result = gemini.build_request(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="key",
            model="gemini-pro",
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert "system_instruction" in result["body"]
        assert result["body"]["system_instruction"]["parts"][0]["text"] == "Be helpful"

    def test_generation_config_mapped(self):
        result = gemini.build_request(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="key",
            model="gemini-pro",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=200,
        )
        gen = result["body"]["generationConfig"]
        assert gen["temperature"] == 0.7
        assert gen["topP"] == 0.9
        assert gen["topK"] == 40
        assert gen["maxOutputTokens"] == 200

    def test_json_response_format(self):
        result = gemini.build_request(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="key",
            model="gemini-pro",
            messages=[{"role": "user", "content": "Hi"}],
            response_format={"type": "json_object"},
        )
        assert result["body"]["generationConfig"]["responseMimeType"] == "application/json"

    def test_tools_converted(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]
        result = gemini.build_request(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="key",
            model="gemini-pro",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        decls = result["body"]["tools"][0]["function_declarations"]
        assert decls[0]["name"] == "search"

    def test_messages_role_mapped(self):
        result = gemini.build_request(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="key",
            model="gemini-pro",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        )
        contents = result["body"]["contents"]
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"


class TestGeminiParseResponse:
    def test_text_response(self):
        raw = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello!"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
            "modelVersion": "gemini-pro",
        }
        result = gemini.parse_response(raw)
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10

    def test_tool_call_response(self):
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "search", "args": {"q": "weather"}}}
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
        }
        result = gemini.parse_response(raw)
        assert result["tool_calls"][0]["function"]["name"] == "search"

    def test_empty_candidates(self):
        raw = {"candidates": [], "usageMetadata": {}}
        result = gemini.parse_response(raw)
        assert result["content"] is None


class TestGeminiParseStreamChunk:
    def test_text_chunk(self):
        raw = json.dumps({
            "candidates": [{"content": {"parts": [{"text": "Hi"}]}}],
        })
        result = gemini.parse_stream_chunk(raw)
        assert result["choices"][0]["delta"]["content"] == "Hi"

    def test_finish_reason(self):
        raw = json.dumps({
            "candidates": [
                {"content": {"parts": [{"text": "Done"}]}, "finishReason": "STOP"}
            ],
        })
        result = gemini.parse_stream_chunk(raw)
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_empty_line(self):
        assert gemini.parse_stream_chunk("") is None

    def test_done_line(self):
        assert gemini.parse_stream_chunk("[DONE]") is None

    def test_usage_in_chunk(self):
        raw = json.dumps({
            "candidates": [{"content": {"parts": [{"text": "x"}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        })
        result = gemini.parse_stream_chunk(raw)
        assert result["usage"]["prompt_tokens"] == 10

    def test_function_call_in_stream(self):
        raw = json.dumps({
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "fn1", "args": {"x": 1}}}]
                    }
                }
            ],
        })
        result = gemini.parse_stream_chunk(raw)
        assert result["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "fn1"


# ═══════════════════════════════════════════════════════════════════════
# Ollama Adapter
# ═══════════════════════════════════════════════════════════════════════


class TestOllamaBuildRequest:
    def test_default_base_url(self):
        result = ollama.build_request(
            model="llama3",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result["url"] == "http://localhost:11434/v1/chat/completions"

    def test_custom_base_url(self):
        result = ollama.build_request(
            base_url="http://remote:11434/v1",
            model="llama3",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result["url"] == "http://remote:11434/v1/chat/completions"

    def test_no_auth_header(self):
        result = ollama.build_request(
            model="llama3",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert "Authorization" not in result["headers"]

    def test_parse_response_delegates_to_openai_compat(self):
        raw = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        result = ollama.parse_response(raw)
        assert result["content"] == "Hi"

    def test_parse_stream_chunk_delegates(self):
        chunk_json = json.dumps({
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        })
        result = ollama.parse_stream_chunk(chunk_json)
        assert result is not None


class TestOllamaEmbedding:
    def test_build_embedding_request_default_url(self):
        result = ollama.build_embedding_request(
            model="nomic-embed-text",
            input=["Hello"],
        )
        assert result["url"] == "http://localhost:11434/v1/embeddings"
        assert "Authorization" not in result["headers"]


# ═══════════════════════════════════════════════════════════════════════
# Azure OpenAI Adapter
# ═══════════════════════════════════════════════════════════════════════


class TestAzureOpenAIBuildRequest:
    def test_url_includes_api_version(self):
        result = azure_openai.build_request(
            base_url="https://myresource.openai.azure.com/openai/deployments/gpt-4",
            api_key="azure-key",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert "api-version=" in result["url"]
        assert "/chat/completions" in result["url"]

    def test_custom_api_version(self):
        result = azure_openai.build_request(
            base_url="https://myresource.openai.azure.com/openai/deployments/gpt-4",
            api_key="azure-key",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            api_version="2024-01-01",
        )
        assert "api-version=2024-01-01" in result["url"]

    def test_auth_uses_api_key_header(self):
        result = azure_openai.build_request(
            base_url="https://myresource.openai.azure.com/openai/deployments/gpt-4",
            api_key="azure-key-123",
            model="gpt-4",
            messages=[],
        )
        assert result["headers"]["api-key"] == "azure-key-123"
        assert "Authorization" not in result["headers"]

    def test_stream_options_added(self):
        result = azure_openai.build_request(
            base_url="https://resource.openai.azure.com/openai/deployments/gpt-4",
            api_key="key",
            model="gpt-4",
            messages=[],
            stream=True,
        )
        assert result["body"]["stream"] is True
        assert result["body"]["stream_options"] == {"include_usage": True}

    def test_chat_params_forwarded(self):
        result = azure_openai.build_request(
            base_url="https://resource.openai.azure.com/openai/deployments/gpt-4",
            api_key="key",
            model="gpt-4",
            messages=[],
            temperature=0.5,
            max_tokens=100,
        )
        assert result["body"]["temperature"] == 0.5
        assert result["body"]["max_tokens"] == 100

    def test_parse_response_delegates(self):
        raw = {
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = azure_openai.parse_response(raw)
        assert result["content"] == "Hello"


class TestAzureOpenAIEmbedding:
    def test_build_embedding_request(self):
        result = azure_openai.build_embedding_request(
            base_url="https://resource.openai.azure.com/openai/deployments/embed",
            api_key="key",
            model="text-embedding-ada-002",
            input=["test"],
        )
        assert "/embeddings" in result["url"]
        assert "api-version=" in result["url"]
        assert result["headers"]["api-key"] == "key"


# ═══════════════════════════════════════════════════════════════════════
# Bedrock Adapter
# ═══════════════════════════════════════════════════════════════════════


class TestBedrockBuildRequest:
    @patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "AKID",
        "AWS_SECRET_ACCESS_KEY": "SECRET",
        "AWS_REGION": "us-west-2",
    })
    @patch("nanollm.adapters.bedrock._get_credentials", return_value=("AKID", "SECRET", None))
    def test_url_construction(self, mock_creds):
        result = bedrock.build_request(
            model="anthropic.claude-v2",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert "bedrock-runtime.us-west-2.amazonaws.com" in result["url"]
        assert "anthropic.claude-v2/converse" in result["url"]

    @patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "AKID",
        "AWS_SECRET_ACCESS_KEY": "SECRET",
        "AWS_REGION": "us-east-1",
    })
    @patch("nanollm.adapters.bedrock._get_credentials", return_value=("AKID", "SECRET", None))
    def test_stream_url(self, mock_creds):
        result = bedrock.build_request(
            model="anthropic.claude-v2",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        assert "converse-stream" in result["url"]

    @patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "AKID",
        "AWS_SECRET_ACCESS_KEY": "SECRET",
        "AWS_REGION": "us-east-1",
    })
    @patch("nanollm.adapters.bedrock._get_credentials", return_value=("AKID", "SECRET", None))
    def test_system_message_extracted(self, mock_creds):
        result = bedrock.build_request(
            model="anthropic.claude-v2",
            messages=[
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert result["body"]["system"] == [{"text": "Be concise"}]

    @patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "AKID",
        "AWS_SECRET_ACCESS_KEY": "SECRET",
        "AWS_REGION": "us-east-1",
    })
    @patch("nanollm.adapters.bedrock._get_credentials", return_value=("AKID", "SECRET", None))
    def test_inference_config_mapped(self, mock_creds):
        result = bedrock.build_request(
            model="anthropic.claude-v2",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
        )
        ic = result["body"]["inferenceConfig"]
        assert ic["maxTokens"] == 100
        assert ic["temperature"] == 0.5
        assert ic["topP"] == 0.9

    @patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "AKID",
        "AWS_SECRET_ACCESS_KEY": "SECRET",
        "AWS_REGION": "us-east-1",
    })
    @patch("nanollm.adapters.bedrock._get_credentials", return_value=("AKID", "SECRET", None))
    def test_sigv4_headers_added(self, mock_creds):
        result = bedrock.build_request(
            model="anthropic.claude-v2",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert "Authorization" in result["headers"]
        assert "X-Amz-Date" in result["headers"]
        assert result["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256")


class TestBedrockParseResponse:
    def test_text_response(self):
        raw = {
            "output": {"message": {"content": [{"text": "Hello!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
        result = bedrock.parse_response(raw)
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_empty_content(self):
        raw = {"output": {"message": {"content": []}}, "stopReason": "end_turn", "usage": {}}
        result = bedrock.parse_response(raw)
        assert result["content"] is None

    def test_max_tokens_stop(self):
        raw = {
            "output": {"message": {"content": [{"text": "partial"}]}},
            "stopReason": "max_tokens",
            "usage": {},
        }
        result = bedrock.parse_response(raw)
        assert result["finish_reason"] == "length"


class TestBedrockParseStreamChunk:
    def test_content_block_delta(self):
        event = json.dumps({"contentBlockDelta": {"delta": {"text": "Hello"}}})
        result = bedrock.parse_stream_chunk(event)
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_message_stop(self):
        event = json.dumps({"messageStop": {"stopReason": "end_turn"}})
        result = bedrock.parse_stream_chunk(event)
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_metadata_usage(self):
        event = json.dumps({
            "metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}
        })
        result = bedrock.parse_stream_chunk(event)
        assert result["usage"]["prompt_tokens"] == 10

    def test_empty_line(self):
        assert bedrock.parse_stream_chunk("") is None

    def test_invalid_json(self):
        assert bedrock.parse_stream_chunk("{bad") is None

    def test_irrelevant_event(self):
        event = json.dumps({"messageStart": {"role": "assistant"}})
        assert bedrock.parse_stream_chunk(event) is None


# ═══════════════════════════════════════════════════════════════════════
# Vertex AI Adapter
# ═══════════════════════════════════════════════════════════════════════


class TestVertexBuildRequest:
    @patch.dict(os.environ, {
        "VERTEX_PROJECT": "my-project",
        "VERTEX_LOCATION": "us-central1",
    })
    def test_url_construction(self):
        result = vertex.build_request(
            model="gemini-pro",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="token",
        )
        assert "us-central1-aiplatform.googleapis.com" in result["url"]
        assert "my-project" in result["url"]
        assert "generateContent" in result["url"]

    @patch.dict(os.environ, {
        "VERTEX_PROJECT": "my-project",
        "VERTEX_LOCATION": "us-central1",
    })
    def test_stream_url(self):
        result = vertex.build_request(
            model="gemini-pro",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="token",
            stream=True,
        )
        assert "streamGenerateContent" in result["url"]

    @patch.dict(os.environ, {
        "VERTEX_PROJECT": "my-project",
        "VERTEX_LOCATION": "us-central1",
    })
    def test_auth_header(self):
        result = vertex.build_request(
            model="gemini-pro",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="my-token",
        )
        assert result["headers"]["Authorization"] == "Bearer my-token"

    @patch.dict(os.environ, {
        "VERTEX_PROJECT": "my-project",
        "VERTEX_LOCATION": "us-central1",
    })
    def test_system_instruction(self):
        result = vertex.build_request(
            model="gemini-pro",
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ],
            api_key="token",
        )
        assert "systemInstruction" in result["body"]

    @patch.dict(os.environ, {
        "VERTEX_PROJECT": "my-project",
        "VERTEX_LOCATION": "us-central1",
    })
    def test_generation_config(self):
        result = vertex.build_request(
            model="gemini-pro",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="token",
            temperature=0.5,
            max_tokens=100,
        )
        gen = result["body"]["generationConfig"]
        assert gen["temperature"] == 0.5
        assert gen["maxOutputTokens"] == 100

    def test_missing_project_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure VERTEX_PROJECT and GOOGLE_CLOUD_PROJECT are unset
            env = {k: v for k, v in os.environ.items()
                   if k not in ("VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT")}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="project"):
                    vertex.build_request(
                        model="gemini-pro",
                        messages=[{"role": "user", "content": "Hi"}],
                        api_key="token",
                    )


class TestVertexParseResponse:
    def test_text_response(self):
        raw = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello from Vertex"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
            "modelVersion": "gemini-pro",
        }
        result = vertex.parse_response(raw)
        assert result["content"] == "Hello from Vertex"
        assert result["finish_reason"] == "stop"

    def test_empty_candidates(self):
        raw = {"candidates": []}
        result = vertex.parse_response(raw)
        assert result["content"] is None


class TestVertexParseStreamChunk:
    def test_text_chunk(self):
        raw = json.dumps({
            "candidates": [{"content": {"parts": [{"text": "Hi"}]}}],
        })
        # Vertex stream may have array prefixes
        result = vertex.parse_stream_chunk(raw)
        assert result is not None
        assert result["choices"][0]["delta"].get("content") == "Hi"

    def test_empty_line(self):
        assert vertex.parse_stream_chunk("") is None

    def test_array_prefix_stripped(self):
        raw = json.dumps({"candidates": [{"content": {"parts": [{"text": "X"}]}}]})
        line = f"[{raw}"
        result = vertex.parse_stream_chunk(line)
        assert result["choices"][0]["delta"]["content"] == "X"

    def test_finish_reason(self):
        raw = json.dumps({
            "candidates": [
                {"content": {"parts": [{"text": "end"}]}, "finishReason": "STOP"}
            ],
        })
        result = vertex.parse_stream_chunk(raw)
        assert result["choices"][0]["finish_reason"] == "stop"

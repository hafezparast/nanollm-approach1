"""Model string routing — parse provider/model strings and resolve adapters.

Handles the "provider/model-name" convention used across the codebase,
dynamically loads the correct adapter, and applies any runtime overrides
(api_key, base_url, etc.).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ._config import ProviderConfig, get_provider_config

if TYPE_CHECKING:
    from .adapters._base import BaseAdapter


# Providers whose model names can themselves contain "/"
# (e.g. "openrouter/meta-llama/llama-3-70b", "bedrock/anthropic.claude-v2")
_SLASH_IN_MODEL: frozenset[str] = frozenset({
    "together", "together_ai",
    "openrouter",
    "bedrock",
    "vertex_ai",
    "deepinfra",
    "fireworks", "fireworks_ai",
    "anyscale",
})

# Cached adapter singletons: adapter module name -> instance
_adapter_cache: dict[str, "_ModuleAdapter"] = {}


def parse_model_string(model: str) -> tuple[str, str]:
    """Split a "provider/model-name" string into (provider, model_id).

    Rules:
        - If no "/" is present, default provider is "openai".
        - For providers in _SLASH_IN_MODEL, only the first "/" is used
          as the separator; the rest are part of the model id.
        - Otherwise, split on the first "/" as well.

    Examples:
        >>> parse_model_string("gpt-4o")
        ("openai", "gpt-4o")
        >>> parse_model_string("anthropic/claude-3-opus")
        ("anthropic", "claude-3-opus")
        >>> parse_model_string("openrouter/meta-llama/llama-3-70b")
        ("openrouter", "meta-llama/llama-3-70b")
    """
    if "/" not in model:
        return "openai", model

    provider, _, model_id = model.partition("/")
    provider = provider.strip().lower()
    model_id = model_id.strip()

    if not model_id:
        # Trailing slash with no model, e.g. "openai/"
        return provider, ""

    return provider, model_id


class _ModuleAdapter:
    """Wraps a module with build_request/parse_response functions as an adapter.

    Adapters are stateless modules with module-level functions. This thin
    wrapper gives them a uniform object interface for the router.
    """

    def __init__(self, module):
        self._mod = module

    def build_request(self, *args, **kwargs):
        return self._mod.build_request(*args, **kwargs)

    def parse_response(self, *args, **kwargs):
        return self._mod.parse_response(*args, **kwargs)

    def parse_stream_chunk(self, *args, **kwargs):
        return self._mod.parse_stream_chunk(*args, **kwargs)

    def build_embedding_request(self, *args, **kwargs):
        return self._mod.build_embedding_request(*args, **kwargs)

    def parse_embedding_response(self, *args, **kwargs):
        return self._mod.parse_embedding_response(*args, **kwargs)

    @staticmethod
    def filter_params(kwargs: dict, supported: frozenset) -> dict:
        return {k: v for k, v in kwargs.items() if k in supported}


def get_adapter(provider: str) -> _ModuleAdapter:
    """Dynamically import and cache the adapter for a provider.

    Each adapter module lives at ``nanollm.adapters.<adapter_name>`` and
    exposes module-level functions: build_request, parse_response, etc.

    Args:
        provider: Provider name used to look up the adapter module name.

    Returns:
        Cached adapter wrapper.
    """
    config = get_provider_config(provider)
    adapter_name = config.adapter

    if adapter_name in _adapter_cache:
        return _adapter_cache[adapter_name]

    module = importlib.import_module(f".adapters.{adapter_name}", package="nanollm")
    instance = _ModuleAdapter(module)
    _adapter_cache[adapter_name] = instance
    return instance


def resolve(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
) -> tuple[str, str, "BaseAdapter", ProviderConfig]:
    """Full resolution chain: model string -> (provider, model_id, adapter, config).

    Parses the model string, loads the adapter, and applies any runtime
    overrides for base_url / api_base.

    Args:
        model: Model string, e.g. "openai/gpt-4o" or "gpt-4o".
        api_key: Optional API key override (not baked into config;
            passed through for the caller to use).
        base_url: Optional base URL override.
        api_base: Alias for base_url (litellm compat).

    Returns:
        Tuple of (provider_name, model_id, adapter_instance, provider_config).

    Raises:
        ValueError: If the provider is unknown or required config is missing.
    """
    provider, model_id = parse_model_string(model)
    config = get_provider_config(provider)
    adapter = get_adapter(provider)

    # Apply base_url overrides (api_base is a litellm-compat alias)
    effective_base_url = base_url or api_base or config.base_url

    if not effective_base_url and provider not in ("bedrock", "vertex_ai"):
        raise ValueError(
            f"Provider {provider!r} requires a base_url but none was provided. "
            f"Pass base_url= or set it in the provider config."
        )

    # If overridden, build a new config with the updated base_url
    if effective_base_url != config.base_url:
        config = ProviderConfig(
            name=config.name,
            base_url=effective_base_url,
            api_key_env=config.api_key_env,
            adapter=config.adapter,
            supported_params=config.supported_params,
            auth_header=config.auth_header,
            auth_prefix=config.auth_prefix,
            extra_headers=config.extra_headers,
        )

    return provider, model_id, adapter, config

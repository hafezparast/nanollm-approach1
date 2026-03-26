"""litellm compatibility shim.

Re-exports nanollm's public API under the litellm namespace so that
existing code using `from litellm import completion` works unchanged.
"""

from nanollm import (
    __version__,
    acompletion,
    aembedding,
    atext_completion,
    batch_completion,
    completion,
    embedding,
    stream_chunk_builder,
    text_completion,
    EmbeddingResponse,
    ModelResponse,
    TextCompletionResponse,
)
from nanollm import drop_params, set_verbose
from nanollm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    BudgetExceededError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    InvalidRequestError,
    JSONSchemaValidationError,
    NanoLLMException,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnsupportedParamsError,
)

import nanollm as _nanollm
import sys


class _LitellmModule(sys.modules[__name__].__class__):
    @property
    def drop_params(self):
        return _nanollm.drop_params

    @drop_params.setter
    def drop_params(self, value):
        _nanollm.drop_params = value

    @property
    def set_verbose(self):
        return _nanollm.set_verbose

    @set_verbose.setter
    def set_verbose(self, value):
        _nanollm.set_verbose = value

sys.modules[__name__].__class__ = _LitellmModule

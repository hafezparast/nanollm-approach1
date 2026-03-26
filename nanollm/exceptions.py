"""NanoLLM exception hierarchy.

Maps HTTP status codes to specific exception types for consistent
error handling across all providers. Hierarchy designed for backward
compatibility — catching a parent class catches all children.
"""

from __future__ import annotations


class NanoLLMException(Exception):
    """Base exception for all NanoLLM errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        llm_provider: str | None = None,
        model: str | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(message)


# litellm compat alias
OpenAIError = NanoLLMException


class RateLimitError(NanoLLMException):
    """HTTP 429 — rate limit exceeded."""


class AuthenticationError(NanoLLMException):
    """HTTP 401 — invalid or missing credentials."""


class InvalidRequestError(NanoLLMException):
    """HTTP 400/422 — malformed request or invalid parameters."""


class BadRequestError(InvalidRequestError):
    """HTTP 400 — alias for InvalidRequestError (litellm compat)."""


class NotFoundError(NanoLLMException):
    """HTTP 404 — requested resource not found."""


class PermissionDeniedError(NanoLLMException):
    """HTTP 403 — access denied."""


class APIError(NanoLLMException):
    """Generic API error for unexpected status codes."""


class InternalServerError(APIError):
    """HTTP 500 — server-side error."""


class BadGatewayError(APIError):
    """HTTP 502 — bad gateway."""


class ServiceUnavailableError(APIError):
    """HTTP 503 — provider temporarily unavailable."""


class APIConnectionError(NanoLLMException):
    """Connection error — network unreachable, DNS failure, etc."""


class Timeout(NanoLLMException):
    """HTTP 408 or request timeout."""


class ContextWindowExceededError(InvalidRequestError):
    """Input exceeds the model's context window."""


class ContentPolicyViolationError(InvalidRequestError):
    """Response blocked by content policy / safety filter."""


class UnsupportedParamsError(InvalidRequestError):
    """Request contains parameters not supported by the provider."""


class BudgetExceededError(NanoLLMException):
    """Budget limit exceeded."""


class JSONSchemaValidationError(InvalidRequestError):
    """JSON schema validation failed."""


# Status code → exception class mapping
STATUS_CODE_MAP: dict[int, type[NanoLLMException]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    408: Timeout,
    422: InvalidRequestError,
    429: RateLimitError,
    500: InternalServerError,
    502: BadGatewayError,
    503: ServiceUnavailableError,
}


def raise_for_status(
    status_code: int,
    body: dict | str,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Raise the appropriate exception for an HTTP error response."""
    if 200 <= status_code < 300:
        return

    if isinstance(body, dict):
        message = (
            body.get("error", {}).get("message")
            or body.get("error", {}).get("msg")
            or body.get("message")
            or body.get("detail")
            or str(body)
        )
    else:
        message = str(body)

    exc_class = STATUS_CODE_MAP.get(status_code, APIError)
    raise exc_class(
        message=message,
        status_code=status_code,
        llm_provider=provider,
        model=model,
    )

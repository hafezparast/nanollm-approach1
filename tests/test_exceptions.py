"""Exhaustive tests for nanollm.exceptions — hierarchy, STATUS_CODE_MAP, raise_for_status."""

from __future__ import annotations

import pytest

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
    STATUS_CODE_MAP,
    ServiceUnavailableError,
    Timeout,
    UnsupportedParamsError,
    raise_for_status,
)


# ════════════════════════════════════════════════════════════════════════
# NanoLLMException base
# ════════════════════════════════════════════════════════════════════════


class TestNanoLLMException:
    def test_construction_defaults(self):
        e = NanoLLMException("something broke")
        assert e.message == "something broke"
        assert e.status_code is None
        assert e.llm_provider is None
        assert e.model is None

    def test_construction_all_fields(self):
        e = NanoLLMException("err", status_code=500, llm_provider="openai", model="gpt-4o")
        assert e.message == "err"
        assert e.status_code == 500
        assert e.llm_provider == "openai"
        assert e.model == "gpt-4o"

    def test_str(self):
        e = NanoLLMException("test msg")
        assert str(e) == "test msg"

    def test_is_exception(self):
        assert issubclass(NanoLLMException, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(NanoLLMException):
            raise NanoLLMException("boom")


# ════════════════════════════════════════════════════════════════════════
# OpenAIError alias
# ════════════════════════════════════════════════════════════════════════


class TestOpenAIError:
    def test_is_nanollm_exception(self):
        assert OpenAIError is NanoLLMException

    def test_catching_openai_error_catches_nanollm(self):
        with pytest.raises(OpenAIError):
            raise NanoLLMException("test")

    def test_catching_nanollm_catches_openai(self):
        with pytest.raises(NanoLLMException):
            raise OpenAIError("test")


# ════════════════════════════════════════════════════════════════════════
# Individual exception classes — construction + attributes
# ════════════════════════════════════════════════════════════════════════


class TestRateLimitError:
    def test_construction(self):
        e = RateLimitError("rate limited", status_code=429, llm_provider="openai")
        assert e.message == "rate limited"
        assert e.status_code == 429

    def test_isinstance(self):
        assert issubclass(RateLimitError, NanoLLMException)

    def test_str(self):
        assert str(RateLimitError("slow down")) == "slow down"


class TestAuthenticationError:
    def test_construction(self):
        e = AuthenticationError("bad key", status_code=401)
        assert e.status_code == 401

    def test_isinstance(self):
        assert issubclass(AuthenticationError, NanoLLMException)


class TestInvalidRequestError:
    def test_construction(self):
        e = InvalidRequestError("invalid", status_code=422)
        assert e.message == "invalid"

    def test_isinstance(self):
        assert issubclass(InvalidRequestError, NanoLLMException)


class TestBadRequestError:
    def test_construction(self):
        e = BadRequestError("bad request", status_code=400)
        assert e.status_code == 400

    def test_is_invalid_request_error(self):
        assert issubclass(BadRequestError, InvalidRequestError)

    def test_is_nanollm_exception(self):
        assert issubclass(BadRequestError, NanoLLMException)

    def test_catching_invalid_request_catches_bad_request(self):
        with pytest.raises(InvalidRequestError):
            raise BadRequestError("bad")


class TestNotFoundError:
    def test_construction(self):
        e = NotFoundError("not found", status_code=404)
        assert e.status_code == 404

    def test_isinstance(self):
        assert issubclass(NotFoundError, NanoLLMException)


class TestPermissionDeniedError:
    def test_construction(self):
        e = PermissionDeniedError("denied", status_code=403)
        assert e.status_code == 403

    def test_isinstance(self):
        assert issubclass(PermissionDeniedError, NanoLLMException)


class TestAPIError:
    def test_construction(self):
        e = APIError("api error", status_code=500)
        assert e.message == "api error"

    def test_isinstance(self):
        assert issubclass(APIError, NanoLLMException)


class TestInternalServerError:
    def test_construction(self):
        e = InternalServerError("server down", status_code=500)
        assert e.status_code == 500

    def test_is_api_error(self):
        assert issubclass(InternalServerError, APIError)

    def test_is_nanollm_exception(self):
        assert issubclass(InternalServerError, NanoLLMException)

    def test_catching_api_error_catches_internal(self):
        with pytest.raises(APIError):
            raise InternalServerError("500")


class TestBadGatewayError:
    def test_construction(self):
        e = BadGatewayError("bad gateway", status_code=502)
        assert e.status_code == 502

    def test_is_api_error(self):
        assert issubclass(BadGatewayError, APIError)


class TestServiceUnavailableError:
    def test_construction(self):
        e = ServiceUnavailableError("unavailable", status_code=503)
        assert e.status_code == 503

    def test_is_api_error(self):
        assert issubclass(ServiceUnavailableError, APIError)


class TestAPIConnectionError:
    def test_construction(self):
        e = APIConnectionError("no connection")
        assert e.message == "no connection"

    def test_isinstance(self):
        assert issubclass(APIConnectionError, NanoLLMException)


class TestTimeout:
    def test_construction(self):
        e = Timeout("timed out", status_code=408)
        assert e.status_code == 408

    def test_isinstance(self):
        assert issubclass(Timeout, NanoLLMException)


class TestContextWindowExceededError:
    def test_construction(self):
        e = ContextWindowExceededError("too long")
        assert e.message == "too long"

    def test_is_invalid_request_error(self):
        assert issubclass(ContextWindowExceededError, InvalidRequestError)

    def test_is_nanollm_exception(self):
        assert issubclass(ContextWindowExceededError, NanoLLMException)


class TestContentPolicyViolationError:
    def test_construction(self):
        e = ContentPolicyViolationError("blocked")
        assert e.message == "blocked"

    def test_is_invalid_request_error(self):
        assert issubclass(ContentPolicyViolationError, InvalidRequestError)


class TestUnsupportedParamsError:
    def test_construction(self):
        e = UnsupportedParamsError("unsupported param: foo")
        assert e.message == "unsupported param: foo"

    def test_is_invalid_request_error(self):
        assert issubclass(UnsupportedParamsError, InvalidRequestError)


class TestBudgetExceededError:
    def test_construction(self):
        e = BudgetExceededError("budget exceeded")
        assert e.message == "budget exceeded"

    def test_isinstance(self):
        assert issubclass(BudgetExceededError, NanoLLMException)


class TestJSONSchemaValidationError:
    def test_construction(self):
        e = JSONSchemaValidationError("schema failed")
        assert e.message == "schema failed"

    def test_is_invalid_request_error(self):
        assert issubclass(JSONSchemaValidationError, InvalidRequestError)


# ════════════════════════════════════════════════════════════════════════
# Inheritance / backward compat — catching parent catches child
# ════════════════════════════════════════════════════════════════════════


class TestInheritanceHierarchy:
    def test_nanollm_catches_all(self):
        """NanoLLMException catches every exception in the module."""
        for exc_class in [
            RateLimitError, AuthenticationError, InvalidRequestError,
            BadRequestError, NotFoundError, PermissionDeniedError,
            APIError, InternalServerError, BadGatewayError,
            ServiceUnavailableError, APIConnectionError, Timeout,
            ContextWindowExceededError, ContentPolicyViolationError,
            UnsupportedParamsError, BudgetExceededError, JSONSchemaValidationError,
        ]:
            with pytest.raises(NanoLLMException):
                raise exc_class("test")

    def test_invalid_request_catches_children(self):
        for exc_class in [
            BadRequestError, ContextWindowExceededError,
            ContentPolicyViolationError, UnsupportedParamsError,
            JSONSchemaValidationError,
        ]:
            with pytest.raises(InvalidRequestError):
                raise exc_class("test")

    def test_api_error_catches_children(self):
        for exc_class in [InternalServerError, BadGatewayError, ServiceUnavailableError]:
            with pytest.raises(APIError):
                raise exc_class("test")

    def test_bad_request_not_caught_by_api_error(self):
        """BadRequestError is NOT an APIError."""
        with pytest.raises(BadRequestError):
            try:
                raise BadRequestError("test")
            except APIError:
                pytest.fail("BadRequestError should not be caught by APIError")

    def test_rate_limit_not_caught_by_invalid_request(self):
        with pytest.raises(RateLimitError):
            try:
                raise RateLimitError("test")
            except InvalidRequestError:
                pytest.fail("RateLimitError should not be caught by InvalidRequestError")


# ════════════════════════════════════════════════════════════════════════
# STATUS_CODE_MAP
# ════════════════════════════════════════════════════════════════════════


class TestStatusCodeMap:
    def test_400(self):
        assert STATUS_CODE_MAP[400] is BadRequestError

    def test_401(self):
        assert STATUS_CODE_MAP[401] is AuthenticationError

    def test_403(self):
        assert STATUS_CODE_MAP[403] is PermissionDeniedError

    def test_404(self):
        assert STATUS_CODE_MAP[404] is NotFoundError

    def test_408(self):
        assert STATUS_CODE_MAP[408] is Timeout

    def test_422(self):
        assert STATUS_CODE_MAP[422] is InvalidRequestError

    def test_429(self):
        assert STATUS_CODE_MAP[429] is RateLimitError

    def test_500(self):
        assert STATUS_CODE_MAP[500] is InternalServerError

    def test_502(self):
        assert STATUS_CODE_MAP[502] is BadGatewayError

    def test_503(self):
        assert STATUS_CODE_MAP[503] is ServiceUnavailableError

    def test_unknown_code_not_in_map(self):
        assert 999 not in STATUS_CODE_MAP

    def test_200_not_in_map(self):
        assert 200 not in STATUS_CODE_MAP


# ════════════════════════════════════════════════════════════════════════
# raise_for_status
# ════════════════════════════════════════════════════════════════════════


class TestRaiseForStatus:
    # ── 2xx no-raise ──

    def test_200_no_raise(self):
        raise_for_status(200, {"ok": True})  # should not raise

    def test_201_no_raise(self):
        raise_for_status(201, "created")

    def test_204_no_raise(self):
        raise_for_status(204, "")

    def test_299_no_raise(self):
        raise_for_status(299, {})

    # ── dict body: nested error.message ──

    def test_dict_error_message(self):
        body = {"error": {"message": "Invalid API key"}}
        with pytest.raises(AuthenticationError, match="Invalid API key") as exc_info:
            raise_for_status(401, body, provider="openai", model="gpt-4o")
        e = exc_info.value
        assert e.status_code == 401
        assert e.llm_provider == "openai"
        assert e.model == "gpt-4o"

    # ── dict body: nested error.msg ──

    def test_dict_error_msg(self):
        body = {"error": {"msg": "Bad request format"}}
        with pytest.raises(BadRequestError, match="Bad request format"):
            raise_for_status(400, body)

    # ── dict body: flat message ──

    def test_dict_flat_message(self):
        body = {"message": "Not found"}
        with pytest.raises(NotFoundError, match="Not found"):
            raise_for_status(404, body)

    # ── dict body: detail ──

    def test_dict_detail(self):
        body = {"detail": "Unprocessable entity"}
        with pytest.raises(InvalidRequestError, match="Unprocessable entity"):
            raise_for_status(422, body)

    # ── dict body: fallback to str(body) ──

    def test_dict_fallback_str(self):
        body = {"unknown_key": "some value"}
        with pytest.raises(BadRequestError):
            raise_for_status(400, body)

    # ── string body ──

    def test_string_body(self):
        with pytest.raises(InternalServerError, match="Internal Server Error"):
            raise_for_status(500, "Internal Server Error")

    # ── status codes mapping ──

    def test_400_raises_bad_request(self):
        with pytest.raises(BadRequestError):
            raise_for_status(400, "bad")

    def test_401_raises_authentication(self):
        with pytest.raises(AuthenticationError):
            raise_for_status(401, "unauth")

    def test_403_raises_permission_denied(self):
        with pytest.raises(PermissionDeniedError):
            raise_for_status(403, "denied")

    def test_404_raises_not_found(self):
        with pytest.raises(NotFoundError):
            raise_for_status(404, "missing")

    def test_408_raises_timeout(self):
        with pytest.raises(Timeout):
            raise_for_status(408, "timeout")

    def test_422_raises_invalid_request(self):
        with pytest.raises(InvalidRequestError):
            raise_for_status(422, "invalid")

    def test_429_raises_rate_limit(self):
        with pytest.raises(RateLimitError):
            raise_for_status(429, "slow down")

    def test_500_raises_internal_server(self):
        with pytest.raises(InternalServerError):
            raise_for_status(500, "server error")

    def test_502_raises_bad_gateway(self):
        with pytest.raises(BadGatewayError):
            raise_for_status(502, "bad gateway")

    def test_503_raises_service_unavailable(self):
        with pytest.raises(ServiceUnavailableError):
            raise_for_status(503, "unavailable")

    def test_unknown_status_raises_api_error(self):
        with pytest.raises(APIError) as exc_info:
            raise_for_status(599, "unknown error")
        assert exc_info.value.status_code == 599

    # ── provider and model propagated ──

    def test_provider_and_model_set(self):
        with pytest.raises(RateLimitError) as exc_info:
            raise_for_status(429, "limit", provider="anthropic", model="claude-3")
        e = exc_info.value
        assert e.llm_provider == "anthropic"
        assert e.model == "claude-3"

    def test_provider_none_default(self):
        with pytest.raises(APIError) as exc_info:
            raise_for_status(500, "err")
        assert exc_info.value.llm_provider is None
        assert exc_info.value.model is None

    # ── error message extraction priority ──

    def test_error_message_priority_over_flat_message(self):
        """error.message takes priority over top-level message."""
        body = {"error": {"message": "nested"}, "message": "flat"}
        with pytest.raises(BadRequestError, match="nested"):
            raise_for_status(400, body)

    def test_error_msg_priority_over_detail(self):
        """error.msg takes priority over detail."""
        body = {"error": {"msg": "nested_msg"}, "detail": "detail_msg"}
        with pytest.raises(BadRequestError, match="nested_msg"):
            raise_for_status(400, body)

    def test_empty_error_dict_falls_through(self):
        """If error is an empty dict, falls through to flat message."""
        body = {"error": {}, "message": "flat"}
        with pytest.raises(BadRequestError, match="flat"):
            raise_for_status(400, body)

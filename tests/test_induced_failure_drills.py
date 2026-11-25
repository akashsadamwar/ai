"""Drill complete provider failure at every model-backed GenAI entry point.

Issue #195: fallback proves an alternative works, not that anyone is told when
none does. Each drill makes every provider in an operation's chain fail the same
way and checks three things:

* the caller gets a distinguishable service error, not an empty success;
* the warning log names each model that failed, and why;
* neither the response nor the logs carry provider detail or the user's text.

The failure modes are the ones issue #195 lists: a decommissioned model (404),
a revoked key (401), a lapsed subscription (403), an exhausted quota (429), a
degraded provider (503) and a timeout.
"""
import logging
from unittest.mock import Mock

import httpx
import pytest
from google.genai.errors import ClientError, ServerError
from groq import (
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from langchain_core.runnables import RunnableLambda

import lambda_function as handlers
import services.classification_service as classification
import services.skill_suggestion_service as skill_suggestion
import utils
from utils import client as provider_clients
from utils import model_fallback, search_orgs, subject_generator


pytestmark = pytest.mark.unit

# Text the drill puts into provider errors and requests. Neither may reach the
# caller, and the user's text may not reach a log line.
_PROVIDER_DETAIL = "Simulated provider failure"
_USER_TEXT = "I need help studying for a math exam before Friday."

_OPERATIONS = (
    ("predict_category", "CATEGORY_PREDICTION_UNAVAILABLE"),
    ("suggest_skills", "SKILL_SUGGESTION_UNAVAILABLE"),
    ("generate_answer", "ANSWER_GENERATION_FAILED"),
    ("generate_subject", None),
    ("search_orgs", "ORG_SEARCH_UNAVAILABLE"),
)
_LOG_GAPS = {
    "generate_subject": "Subject generator still prints unstructured provider errors",
    "search_orgs": "Organization search warnings name providers only and lack error_class",
}

_GROQ_ERRORS = {
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    429: RateLimitError,
    503: InternalServerError,
}


def _provider_error(provider, status):
    """The exception each SDK raises for a given HTTP status, or a timeout."""
    request = httpx.Request("POST", "https://example.invalid/model")
    if status is None:
        if provider == "groq":
            return APITimeoutError(request=request)
        return httpx.ReadTimeout(_PROVIDER_DETAIL, request=request)

    if provider == "gemini":
        error_type = ServerError if status >= 500 else ClientError
        return error_type(status, {"error": {"message": _PROVIDER_DETAIL}})

    response = httpx.Response(status, request=request)
    return _GROQ_ERRORS[status](_PROVIDER_DETAIL, response=response, body=None)


@pytest.fixture(
    params=[
        (404, "model_not_found"),
        (401, "auth_or_entitlement"),
        (403, "auth_or_entitlement"),
        (429, "rate_limited"),
        (503, "timeout_or_unavailable"),
        (None, "timeout_or_unavailable"),
    ],
    ids=[
        "model-not-found", "auth", "subscription-lapsed",
        "rate-limit", "provider-unavailable", "timeout",
    ],
)
def failure(request):
    return request.param


def _failing_sdk_clients(monkeypatch, module, fail):
    """Replace the Groq and Gemini SDK clients a service module imported."""
    groq_client = Mock()
    gemini_client = Mock()
    groq_client.chat.completions.create.side_effect = (
        lambda **args: fail("groq", args["model"])
    )
    gemini_client.models.generate_content.side_effect = (
        lambda **args: fail("gemini", args["model"])
    )
    monkeypatch.setattr(module, "client", groq_client)
    monkeypatch.setattr(module, "_use_groq", True)
    monkeypatch.setattr(module, "_gemini_client", gemini_client)


def _event(operation):
    event = {
        "subject": "Math tutoring",
        "description": _USER_TEXT,
        "location": "United States",
    }
    if operation == "suggest_skills":
        # The web app reaches skill suggestion through predict_category, so
        # the drill does too.
        event["mode"] = "volunteer_skills"
    return event


def _handler(operation):
    if operation == "suggest_skills":
        return handlers.predict_category_handler
    return getattr(handlers, f"{operation}_handler")


def _expected_attempts(operation):
    if operation in ("predict_category", "suggest_skills", "generate_answer"):
        return [
            (target.provider, target.model)
            for target in model_fallback._DEFAULT_MODEL_CHAIN
        ]
    if operation == "generate_subject":
        return [
            ("groq", subject_generator.GROQ_MODEL),
            ("gemini", subject_generator.GEMINI_MODEL),
        ]
    return [
        ("groq", provider_clients.GROQ_MODEL),
        ("gemini", provider_clients.GEMINI_MODEL),
    ]


@pytest.fixture
def drill(monkeypatch, failure):
    def run(operation):
        status, _ = failure
        attempted = []

        def fail(provider, model):
            attempted.append((provider, model))
            raise _provider_error(provider, status)

        if operation == "predict_category":
            _failing_sdk_clients(monkeypatch, classification, fail)
        elif operation == "suggest_skills":
            _failing_sdk_clients(monkeypatch, skill_suggestion, fail)
        elif operation == "generate_answer":
            def create_model(target, **kwargs):
                model = Mock()
                model.invoke.side_effect = lambda _: fail(target.provider, target.model)
                return model

            monkeypatch.setattr(utils, "create_chat_model", create_model)
        elif operation == "generate_subject":
            for provider, model_name in (
                ("groq", subject_generator.GROQ_MODEL),
                ("gemini", subject_generator.GEMINI_MODEL),
            ):
                model = Mock()
                model.invoke.side_effect = (
                    lambda _, p=provider, m=model_name: fail(p, m)
                )
                monkeypatch.setattr(subject_generator, f"{provider}_llm", model)
                monkeypatch.setattr(subject_generator, f"_use_{provider}", True)
        else:
            monkeypatch.setattr(
                search_orgs, "load_llm",
                lambda: RunnableLambda(lambda _: fail("groq", provider_clients.GROQ_MODEL)),
            )
            monkeypatch.setattr(
                search_orgs, "load_fallback_llm",
                lambda: RunnableLambda(
                    lambda _: fail("gemini", provider_clients.GEMINI_MODEL)
                ),
            )

        response = _handler(operation)(_event(operation), None)
        expected = _expected_attempts(operation)
        if attempted != expected:
            pytest.fail(f"Expected provider attempts {expected}, got {attempted}")
        return response, expected

    return run


def _operation_params(gaps):
    """One param per operation; a known gap is a strict xfail with its reason.

    Strict, so that fixing a gap fails the suite until its entry is removed.
    """
    params = []
    for operation, code in _OPERATIONS:
        marks = ()
        if operation in gaps:
            marks = pytest.mark.xfail(
                strict=True, raises=AssertionError, reason=gaps[operation]
            )
        params.append(pytest.param(operation, code, id=operation, marks=marks))
    return params


@pytest.mark.parametrize(
    ("operation", "expected_code"),
    _operation_params(
        {"generate_subject": "Subject generation returns a successful truncated description"}
    ),
)
def test_total_provider_failure_returns_service_error(drill, operation, expected_code):
    """Provider outages must be distinguishable from successful responses."""
    response, _ = drill(operation)

    if expected_code is None:
        # The team still needs to agree on the exact Subject error code.
        assert 500 <= response["statusCode"] < 600
        return
    assert response["statusCode"] == 502
    assert response["body"] != []
    assert response["body"]["code"] == expected_code
    # The web client renders an empty category list as General, so an outage
    # must not carry one. (search_orgs keeps `organizations: []` on purpose:
    # its consumer reads that field, and the code above marks the outage.)
    assert "categories" not in response["body"]
    assert "answer" not in response["body"]


@pytest.mark.parametrize(("operation", "_code"), _operation_params(gaps={}))
def test_total_provider_failure_hides_provider_detail(drill, operation, _code):
    """The caller learns that a provider failed, never which one or why.

    GenAI Services Specification section 3.5: an error response must not
    carry provider or driver detail.
    """
    response, targets = drill(operation)

    body = repr(response["body"])
    assert _PROVIDER_DETAIL not in body
    for _, model in targets:
        assert model not in body


@pytest.mark.parametrize(
    ("operation", "_code"), _operation_params(gaps=_LOG_GAPS),
)
def test_total_provider_failure_logs_each_model(drill, operation, _code, failure, caplog):
    """A provider warning identifies each model and its failure class."""
    with caplog.at_level(logging.WARNING):
        _, targets = drill(operation)

    _, expected_class = failure
    warnings = [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    for provider, model in targets:
        matches = [
            line for line in warnings
            if line.startswith("MODEL_FAILOVER ")
            and f"provider={provider} model={model} " in line
        ]
        assert len(matches) == 1, (provider, model, warnings)
        # On test without #241, model attribution is still required. Once
        # the classifier is present, the same drill also requires its label.
        if hasattr(model_fallback, "classify_provider_error"):
            assert f"error_class={expected_class} " in matches[0]


def _all_log_output(drill, operation, caplog, capsys):
    """Everything an operation wrote to CloudWatch during a drill.

    Handlers log with both the logging module and print, and Lambda sends
    both to CloudWatch, so both are collected.
    """
    with caplog.at_level(logging.DEBUG):
        drill(operation)
    printed = capsys.readouterr()
    logged = "\n".join(record.getMessage() for record in caplog.records)
    return "\n".join((logged, printed.out, printed.err))


@pytest.mark.parametrize(("operation", "_code"), _operation_params(gaps={}))
def test_total_provider_failure_keeps_user_text_out_of_logs(
    drill, operation, _code, caplog, capsys,
):
    """An outage must not copy the person's own words into CloudWatch."""
    assert _USER_TEXT not in _all_log_output(drill, operation, caplog, capsys)


@pytest.mark.parametrize(
    ("operation", "_code"),
    _operation_params({
        "generate_subject": "Subject generator prints the provider's error message",
        "search_orgs": "Organization search prints the provider's error message",
    }),
)
def test_total_provider_failure_logs_no_provider_message(
    drill, operation, _code, caplog, capsys,
):
    """Logs keep the error type, status and code, not the provider's message.

    A provider's message can quote the prompt, and so the user's text. The
    model chain logs describe_model_error() for this reason; every service
    should do the same.
    """
    assert _PROVIDER_DETAIL not in _all_log_output(drill, operation, caplog, capsys)

"""The HTTP service. Implements proposal 009 (the MVP service); grounding is ADR-011.

`POST /ask` answers a question from the notes, or refuses; `GET /health` says the service is up;
`GET /topics` lists the notebook titles (no note content).
`create_app` wires a `QAService`, a `RateLimiter` and the shared UI secret into a FastAPI app
(tests pass fakes); `build_app` is the factory the server runs: it loads the built index and the
ONNX model once, reads `OPENAI_API_KEY` from the environment only, and fails at startup if
anything is missing. The key is never logged, never returned and never part of an error message.
Each accepted question is counted against the caller's daily limit before any model call; the 16th
question of a day gets 429."""

import time
from collections.abc import Sequence
from typing import Annotated

import structlog
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from noterecall.api_models import AskRequest, AskResponse, ErrorBody, TopicsResponse
from noterecall.config import ApiConfig
from noterecall.generate.rewrite import RewriteError
from noterecall.llm.client import BudgetExceededError, LLMError
from noterecall.serve.ratelimit import RateLimiter, resolve_client_ip
from noterecall.serve.service import QAService

log = structlog.get_logger()
UI_SECRET_HEADER = "X-UI-Secret"  # noqa: S105 - a header name, not a secret
UI_IP_HEADER = "X-UI-Client-IP"


def error(status: int, detail: str, retry_after_s: int | None = None) -> JSONResponse:
    """A JSON error with a short, generic message."""
    headers = {"Retry-After": str(retry_after_s)} if retry_after_s is not None else None
    body = ErrorBody(detail=detail, retry_after_s=retry_after_s)
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


def create_app(  # noqa: PLR0913, PLR0917 - the parts of one app, all needed
    service: QAService,
    limiter: RateLimiter,
    ui_secret: str | None,
    limits: ApiConfig,
    facts: dict[str, object],
    topics: Sequence[str] = (),
) -> FastAPI:
    """The FastAPI app around a service."""
    app = FastAPI(title="NoteRecall", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", **facts}

    @app.get("/topics", response_model=TopicsResponse)
    def topics_list() -> TopicsResponse:
        return TopicsResponse(count=len(topics), notebooks=list(topics))

    @app.post("/ask", response_model=AskResponse, responses={429: {"model": ErrorBody}})
    def ask(
        body: AskRequest,
        request: Request,
        forwarded_for: Annotated[str | None, Header(alias="X-Forwarded-For")] = None,
        ui_client_ip: Annotated[str | None, Header(alias=UI_IP_HEADER)] = None,
        ui_header_secret: Annotated[str | None, Header(alias=UI_SECRET_HEADER)] = None,
    ) -> AskResponse | JSONResponse:
        question = body.question.strip()
        if not question:
            return error(422, "the question is empty")
        if len(question) > limits.max_question_chars:
            return error(422, f"the question is longer than {limits.max_question_chars} characters")
        peer = request.client.host if request.client else None
        ip = resolve_client_ip(forwarded_for, ui_client_ip, ui_header_secret, ui_secret, peer)
        decision = limiter.check(ip)
        if not decision.allowed:
            return error(429, "daily limit reached", decision.retry_after_s)
        return _answer(service, body, question)

    return app


def _answer(service: QAService, body: AskRequest, question: str) -> AskResponse | JSONResponse:
    started = time.perf_counter()
    try:
        response = service.ask(body.model_copy(update={"question": question}))
    except RewriteError:
        return error(502, "the follow-up question could not be rewritten")
    except BudgetExceededError:
        return error(503, "the service's model budget is used up")
    except LLMError:
        return error(502, "the language model is unavailable")
    log.info(
        "ask",
        status=response.status.value,
        reason=response.refusal_reason.value if response.refusal_reason else None,
        history_turns=len(body.history),
        claims=len(response.claims),
        latency_s=round(time.perf_counter() - started, 3),
    )
    return response


def build_app() -> FastAPI:
    """The factory the server runs (`uvicorn noterecall.api:build_app --factory`)."""
    from noterecall.serve.runtime import build_service  # noqa: PLC0415 - heavy imports at startup

    return build_service()

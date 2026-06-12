# -*- coding: utf-8 -*-
"""
Agent Observability — lightweight OpenTelemetry-style tracing.

Provides:
  - Trace / Span data model with trace_id propagation
  - Token counting (prompt / completion / total)
  - Latency histogram per span
  - Structured JSON logging of completed traces
  - Thread-safe span tree

Usage:
    from infrastructure.tracing import Tracer, get_tracer

    tracer = get_tracer()

    with tracer.start_span("chat.request", attributes={"mode": "kb"}) as span:
        ...
        with span.start_child("llm.call", attributes={"model": "gemini"}) as child:
            child.set_tokens(prompt=100, completion=500)
        span.set_output("answer text")

No external collector required — traces are emitted as structured log lines.
Replace with real OTLP exporter when ready.
"""

import contextvars
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---- Context variable for implicit trace propagation ----
_current_span: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "current_span", default=None
)


@dataclass
class Span:
    """A single observable unit of work."""

    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    status: str = "ok"  # ok | error
    attributes: Dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    output: Optional[str] = None
    error: Optional[str] = None
    children: List["Span"] = field(default_factory=list)

    # ---- Token accounting ----

    def set_tokens(self, *, prompt: int = 0, completion: int = 0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion

    def accumulate_child_tokens(self) -> None:
        """Roll up token counts from all children."""
        for child in self.children:
            child.accumulate_child_tokens()
            self.prompt_tokens += child.prompt_tokens
            self.completion_tokens += child.completion_tokens
            self.total_tokens += child.total_tokens

    # ---- Child spans ----

    def start_child(self, name: str, **attributes) -> "Span":
        child = Span(
            name=name,
            trace_id=self.trace_id,
            parent_id=self.span_id,
            attributes=attributes,
        )
        self.children.append(child)
        return child

    # ---- Lifecycle ----

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        self.end_time = time.monotonic()
        self.status = status
        if error:
            self.error = str(error)[:2000]

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def set_output(self, output: str) -> None:
        # Truncate to avoid bloating logs
        self.output = output[:500] if output else None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.attributes:
            d["attributes"] = self.attributes
        if self.error:
            d["error"] = self.error
        if self.output:
            d["output_preview"] = self.output
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


class Tracer:
    """
    Lightweight tracer that emits structured log lines.

    Thread-safe via contextvars. Each async task gets its own span stack.
    """

    def __init__(self, service_name: str = "docmind"):
        self.service_name = service_name

    def start_span(self, name: str, **attributes) -> Span:
        parent = _current_span.get()
        if parent:
            span = parent.start_child(name, **attributes)
        else:
            span = Span(
                name=name,
                trace_id=uuid.uuid4().hex[:32],
                attributes=attributes,
            )
        return span

    @contextmanager
    def span(self, name: str, **attributes):
        """Context manager that auto-finishes the span and logs it."""
        s = self.start_span(name, **attributes)
        token = _current_span.set(s)
        try:
            yield s
            s.finish()
        except Exception as exc:
            s.finish(status="error", error=exc)
            raise
        finally:
            _current_span.reset(token)
            # Log root spans (no parent) when they complete
            if s.parent_id is None:
                s.accumulate_child_tokens()
                logger.info(
                    f"[trace] {s.name} trace_id={s.trace_id} "
                    f"duration={s.duration_ms:.0f}ms "
                    f"tokens={s.total_tokens} "
                    f"status={s.status}"
                )
                if s.status == "error" or logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[trace.detail] {s.to_dict()}")

    def current_span(self) -> Optional[Span]:
        return _current_span.get()


# ---- Global singleton ----
_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer

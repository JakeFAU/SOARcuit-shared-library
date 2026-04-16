from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from opentelemetry.util.types import AttributeValue


def get_tracer(name: str) -> trace.Tracer:
    """Return a module-scoped tracer."""

    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    """Return a module-scoped meter."""

    return metrics.get_meter(name)


def concrete_attributes(
    attributes: Mapping[str, AttributeValue | None] | None = None,
) -> dict[str, AttributeValue]:
    """Drop ``None`` values before sending attributes to telemetry APIs."""

    if attributes is None:
        return {}
    return {key: value for key, value in attributes.items() if value is not None}


def set_span_attributes(
    span: Span,
    attributes: Mapping[str, AttributeValue | None],
) -> None:
    """Set only the span attributes that have concrete values."""

    for key, value in concrete_attributes(attributes).items():
        span.set_attribute(key, value)


def mark_span_ok(span: Span) -> None:
    """Mark a span as successful."""

    span.set_status(Status(StatusCode.OK))


def mark_span_error(span: Span, exc: BaseException) -> None:
    """Mark a span as failed and attach the exception details."""

    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


@contextmanager
def traced_span(
    name: str,
    *,
    tracer_name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Mapping[str, AttributeValue | None] | None = None,
) -> Iterator[Span]:
    """Create a span, apply common attributes, and enforce OK/ERROR status."""

    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes is not None:
            set_span_attributes(span, attributes)
        try:
            yield span
        except Exception as exc:
            mark_span_error(span, exc)
            raise
        else:
            mark_span_ok(span)

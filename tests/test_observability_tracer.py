from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.trace import Span, StatusCode
from shared.observability.tracer import (
    concrete_attributes,
    get_meter,
    get_tracer,
    mark_span_error,
    mark_span_ok,
    set_span_attributes,
    traced_span,
)


def test_get_tracer():
    tracer = get_tracer("test")
    assert tracer is not None

def test_get_meter():
    meter = get_meter("test")
    assert meter is not None

def test_concrete_attributes():
    attrs = {"a": 1, "b": None, "c": "val"}
    concrete = concrete_attributes(attrs)
    assert concrete == {"a": 1, "c": "val"}
    assert concrete_attributes(None) == {}

def test_set_span_attributes():
    span = MagicMock(spec=Span)
    attrs = {"a": 1, "b": None}
    set_span_attributes(span, attrs)
    span.set_attribute.assert_called_once_with("a", 1)

def test_mark_span_ok():
    span = MagicMock(spec=Span)
    mark_span_ok(span)
    span.set_status.assert_called_once()
    status = span.set_status.call_args[0][0]
    assert status.status_code == StatusCode.OK

def test_mark_span_error():
    span = MagicMock(spec=Span)
    exc = ValueError("test error")
    mark_span_error(span, exc)
    span.record_exception.assert_called_once_with(exc)
    span.set_status.assert_called_once()
    status = span.set_status.call_args[0][0]
    assert status.status_code == StatusCode.ERROR
    assert status.description == "test error"

def test_traced_span_success():
    with patch("shared.observability.tracer.get_tracer") as mock_get_tracer:
        mock_tracer = MagicMock()
        mock_span = MagicMock(spec=Span)
        mock_get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        with traced_span("test_span", tracer_name="test_tracer", attributes={"a": 1}) as span:
            assert span == mock_span
            
        mock_span.set_attribute.assert_called_with("a", 1)
        # Should be marked OK at the end
        mock_span.set_status.assert_called()
        status = mock_span.set_status.call_args[0][0]
        assert status.status_code == StatusCode.OK

def test_traced_span_error():
    with patch("shared.observability.tracer.get_tracer") as mock_get_tracer:
        mock_tracer = MagicMock()
        mock_span = MagicMock(spec=Span)
        mock_get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        with pytest.raises(ValueError):
            with traced_span("test_span", tracer_name="test_tracer") as span:
                raise ValueError("oops")
        
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called()
        status = mock_span.set_status.call_args[0][0]
        assert status.status_code == StatusCode.ERROR

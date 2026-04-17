import pytest
from datetime import datetime, UTC
from uuid import UUID, uuid4
from enum import StrEnum
from shared.domain.validation import (
    require_text, require_float, require_int, require_uuid, 
    require_datetime, require_enum, require_mapping, require_tags,
    require_embedding
)
from shared.errors import ValidationError

class ExampleEnum(StrEnum):
    A = "a"
    B = "b"

def test_require_text():
    assert require_text(" hello ", "field") == "hello"
    with pytest.raises(ValidationError):
        require_text(123, "field")
    with pytest.raises(ValidationError):
        require_text("  ", "field")

def test_require_float():
    assert require_float("1.5", "field") == 1.5
    assert require_float(1, "field") == 1.0
    with pytest.raises(ValidationError):
        require_float("abc", "field")
    with pytest.raises(ValidationError):
        require_float(0.5, "field", minimum=1.0)
    with pytest.raises(ValidationError):
        require_float(2.5, "field", maximum=2.0)

def test_require_int():
    assert require_int("10", "field") == 10
    assert require_int(5, "field", minimum=0) == 5
    with pytest.raises(ValidationError):
        require_int("abc", "field")
    with pytest.raises(ValidationError):
        require_int(-1, "field", minimum=0)

def test_require_uuid():
    u = uuid4()
    assert require_uuid(u, "field") == u
    assert require_uuid(str(u), "field") == u
    with pytest.raises(ValidationError):
        require_uuid("invalid", "field")

def test_require_datetime():
    dt = datetime(2023, 1, 1, tzinfo=UTC)
    assert require_datetime(dt, "field") == dt
    assert require_datetime(dt.timestamp(), "field") == dt
    assert require_datetime("2023-01-01T00:00:00+00:00", "field") == dt
    
    dt_naive = datetime(2023, 1, 1)
    assert require_datetime(dt_naive, "field").tzinfo == UTC
    
    with pytest.raises(ValidationError):
        require_datetime("not a date", "field")

def test_require_enum():
    assert require_enum(ExampleEnum, "a", "field") == ExampleEnum.A
    assert require_enum(ExampleEnum, ExampleEnum.B, "field") == ExampleEnum.B
    with pytest.raises(ValidationError):
        require_enum(ExampleEnum, "c", "field")

def test_require_mapping():
    assert require_mapping({"a": 1}, "field") == {"a": 1}
    assert require_mapping(None, "field") == {}
    with pytest.raises(ValidationError):
        require_mapping("not a mapping", "field")

def test_require_tags():
    assert require_tags(["a", " b ", "a"]) == ["a", "b"]
    assert require_tags(None) == []
    with pytest.raises(ValidationError):
        require_tags("not a list")

def test_require_embedding():
    assert require_embedding([1.0, 2.0], dimensions=2) == [1.0, 2.0]
    assert require_embedding(None, dimensions=2) is None
    with pytest.raises(ValidationError):
        require_embedding([1.0], dimensions=2)
    with pytest.raises(ValidationError):
        require_embedding(["a"], dimensions=1)

def test_normalize_embedding_items():
    from shared.domain.validation import _normalize_embedding_items
    
    # Nested list
    assert _normalize_embedding_items([[1.0, 2.0]]) == [1.0, 2.0]
    
    # Mocking tolist()
    class MockArray:
        def tolist(self): return [1.0, 2.0]
    assert _normalize_embedding_items(MockArray()) == [1.0, 2.0]

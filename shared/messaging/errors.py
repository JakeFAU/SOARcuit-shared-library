"""Shared parsing and normalization error taxonomy."""

from shared.errors import ValidationError


class PayloadDecodeError(ValidationError):
    """Raised when raw message bytes cannot be decoded into a payload object."""


class UnsupportedMessageTypeError(ValidationError):
    """Raised when an explicit message type is present but unsupported."""


class UnsupportedPayloadShapeError(ValidationError):
    """Raised when the payload shape cannot be normalized into a known contract."""


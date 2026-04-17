from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# <kind>:<actor>:<verb>[:<variant>]
CANONICAL_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<kind>[a-z]+):(?P<actor>[a-z0-9_-]+):(?P<verb>[a-z0-9_-]+)(?::(?P<variant>[a-z0-9_-]+))?$"
)


class InvalidCanonicalNameError(ValueError):
    """Raised when a string does not match the canonical naming convention."""


@dataclass(frozen=True, slots=True)
class CanonicalName:
    """
    Structured representation of a SOARcuit action or step name.
    Follows: <kind>:<actor>:<verb>[:<variant>]
    """

    raw: str
    kind: str
    actor: str
    verb: str
    variant: str | None = None

    @property
    def operation(self) -> str:
        return self.verb

    @classmethod
    def parse(cls, name: str) -> CanonicalName:
        """
        Parse a colon-delimited string into a structured CanonicalName.
        Raises InvalidCanonicalNameError if the format is incorrect.
        """
        match = CANONICAL_NAME_PATTERN.match(name)
        if not match:
            raise InvalidCanonicalNameError(
                f"Name '{name}' does not match format <kind>:<actor>:<verb>[:<variant>]"
            )

        groups = match.groupdict()
        return cls(
            raw=name,
            kind=groups["kind"],
            actor=groups["actor"],
            verb=groups["verb"],
            variant=groups["variant"],
        )

    def __str__(self) -> str:
        return self.raw

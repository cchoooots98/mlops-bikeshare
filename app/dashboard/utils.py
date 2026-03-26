"""Shared utilities for the dashboard package."""

from __future__ import annotations

_VALID_IDENTIFIER_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


def validate_pg_identifier(name: str) -> str:
    """Raise ValueError if name contains characters outside [A-Za-z0-9_].

    Used to safely interpolate schema/table names into SQL strings.
    """
    if not name:
        raise ValueError("SQL identifier must be non-empty")
    if not (name[0].isalpha() or name[0] == "_"):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    if any(c not in _VALID_IDENTIFIER_CHARS for c in name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name

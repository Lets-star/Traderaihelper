from __future__ import annotations

from typing import Any


__all__ = ["clamp", "safe_div"]


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp ``x`` to the inclusive range [lo, hi].

    If the bounds are provided in reverse order, they will be normalised so that
    ``lo`` is always less than or equal to ``hi``.
    """
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, x))


def safe_div(a: Any, b: Any, default: float = 0.0) -> float:
    """Safely divide ``a`` by ``b``.

    Returns ``default`` when ``b`` is zero or when either operand cannot be
    interpreted as a float.
    """
    try:
        numerator = float(a)
        denominator = float(b)
    except (TypeError, ValueError):
        return default

    if denominator == 0:
        return default
    return numerator / denominator

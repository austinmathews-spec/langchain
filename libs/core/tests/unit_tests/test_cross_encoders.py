"""Tests for `langchain_core.cross_encoders`."""

from __future__ import annotations

import pytest
from typing_extensions import override

from langchain_core.cross_encoders import BaseCrossEncoder


class _FakeCrossEncoder(BaseCrossEncoder):
    """Returns the length of the concatenated pair as the "score"."""

    @override
    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(a) + len(b)) for a, b in text_pairs]


def test_base_cross_encoder_is_abstract() -> None:
    """`BaseCrossEncoder` cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseCrossEncoder()  # type: ignore[abstract]


def test_subclass_without_score_cannot_instantiate() -> None:
    """Subclasses missing `score` remain abstract."""

    class IncompleteEncoder(BaseCrossEncoder):
        pass

    with pytest.raises(TypeError):
        IncompleteEncoder()  # type: ignore[abstract]


def test_concrete_subclass_returns_scores() -> None:
    """A concrete subclass can be instantiated and called."""
    encoder = _FakeCrossEncoder()
    pairs = [("foo", "bar"), ("hello", "world!")]

    scores = encoder.score(pairs)

    assert scores == [6.0, 11.0]
    assert all(isinstance(s, float) for s in scores)


def test_score_with_empty_input() -> None:
    """Scoring an empty input list returns an empty list."""
    encoder = _FakeCrossEncoder()
    assert encoder.score([]) == []


def test_score_preserves_order_and_length() -> None:
    """Output length and order match the input pairs."""
    encoder = _FakeCrossEncoder()
    pairs = [("a", "bbbb"), ("cc", "dd"), ("", "")]

    scores = encoder.score(pairs)

    assert len(scores) == len(pairs)
    assert scores == [5.0, 4.0, 0.0]

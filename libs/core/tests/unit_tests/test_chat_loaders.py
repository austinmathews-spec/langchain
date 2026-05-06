"""Tests for `langchain_core.chat_loaders`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typing_extensions import override

from langchain_core.chat_loaders import BaseChatLoader
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.chat_sessions import ChatSession


class _FakeChatLoader(BaseChatLoader):
    """Concrete `BaseChatLoader` that yields a fixed set of sessions."""

    def __init__(self, sessions: list[ChatSession]) -> None:
        self._sessions = sessions
        self.lazy_load_calls = 0

    @override
    def lazy_load(self) -> Iterator[ChatSession]:
        self.lazy_load_calls += 1
        yield from self._sessions


def _make_session(text: str) -> ChatSession:
    return {"messages": [HumanMessage(content=text), AIMessage(content=f"re:{text}")]}


def test_base_chat_loader_is_abstract() -> None:
    """`BaseChatLoader` cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseChatLoader()  # type: ignore[abstract]


def test_subclass_without_lazy_load_cannot_instantiate() -> None:
    """Subclasses that do not implement `lazy_load` are still abstract."""

    class IncompleteLoader(BaseChatLoader):
        pass

    with pytest.raises(TypeError):
        IncompleteLoader()  # type: ignore[abstract]


def test_load_returns_list_from_lazy_load() -> None:
    """`load` should consume `lazy_load` and return a list."""
    sessions = [_make_session("hi"), _make_session("there")]
    loader = _FakeChatLoader(sessions)

    result = loader.load()

    assert isinstance(result, list)
    assert result == sessions
    assert loader.lazy_load_calls == 1


def test_load_with_empty_iterator() -> None:
    """`load` should return an empty list when `lazy_load` yields nothing."""
    loader = _FakeChatLoader([])
    assert loader.load() == []


def test_load_consumes_iterator_each_call() -> None:
    """Each call to `load` should re-invoke `lazy_load`."""
    loader = _FakeChatLoader([_make_session("a")])

    first = loader.load()
    second = loader.load()

    assert first == second
    assert loader.lazy_load_calls == 2


def test_lazy_load_returns_iterator() -> None:
    """`lazy_load` returns an iterator that can be consumed lazily."""
    sessions = [_make_session("one"), _make_session("two")]
    loader = _FakeChatLoader(sessions)

    iterator = loader.lazy_load()
    # Confirm iterator protocol (not eagerly materialized).
    assert iter(iterator) is iterator
    materialized = list(iterator)
    assert materialized == sessions

"""Tests for `langchain_core.chat_sessions`."""

from __future__ import annotations

from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def test_chat_session_is_typeddict_with_total_false() -> None:
    """`ChatSession` is a `TypedDict` and all keys are optional."""
    # `TypedDict` exposes `__total__` reflecting the `total=` argument
    assert ChatSession.__total__ is False
    # No required keys; all keys should be optional.
    assert ChatSession.__optional_keys__ == frozenset({"messages", "functions"})
    assert ChatSession.__required_keys__ == frozenset()


def test_chat_session_empty_construction() -> None:
    """Empty `ChatSession` literal should be allowed."""
    session: ChatSession = {}
    assert session == {}
    assert "messages" not in session
    assert "functions" not in session


def test_chat_session_with_messages() -> None:
    """A `ChatSession` may contain only messages."""
    messages: list[BaseMessage] = [
        HumanMessage(content="hello"),
        AIMessage(content="world"),
    ]
    session: ChatSession = {"messages": messages}
    assert list(session["messages"]) == messages
    assert "functions" not in session


def test_chat_session_with_messages_and_functions() -> None:
    """A `ChatSession` may contain both messages and function specs."""
    messages: list[BaseMessage] = [HumanMessage(content="hi")]
    functions = [{"name": "do_thing", "parameters": {"type": "object"}}]
    session: ChatSession = {"messages": messages, "functions": functions}
    assert session["messages"] == messages
    assert session["functions"] == functions


def test_chat_session_with_only_functions() -> None:
    """A `ChatSession` may contain only functions (since `total=False`)."""
    functions = [{"name": "fn"}]
    session: ChatSession = {"functions": functions}
    assert session["functions"] == functions
    assert "messages" not in session


def test_chat_session_annotations_present() -> None:
    """Both annotated fields should be exposed via `__annotations__`."""
    assert set(ChatSession.__annotations__) == {"messages", "functions"}

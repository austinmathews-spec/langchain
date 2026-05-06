"""Tests for `langchain_core.stores`.

Covers behaviors not exercised by ``tests/unit_tests/stores/test_in_memory.py``:

* ``BaseStore`` abstract contract.
* Default async fallback (``amget`` / ``amset`` / ``amdelete`` / ``ayield_keys``)
  via ``run_in_executor`` for stores that only implement the sync API.
* ``InMemoryByteStore`` round-trip.
* ``InvalidKeyException`` is a ``LangChainException`` subclass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typing_extensions import override

from langchain_core.exceptions import LangChainException
from langchain_core.stores import (
    BaseStore,
    ByteStore,
    InMemoryByteStore,
    InMemoryStore,
    InvalidKeyException,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class SyncOnlyStore(BaseStore[str, int]):
    """A minimal sync-only store used to exercise the default async fallbacks."""

    def __init__(self) -> None:
        self.store: dict[str, int] = {}
        self.mget_calls = 0
        self.mset_calls = 0
        self.mdelete_calls = 0
        self.yield_calls = 0

    @override
    def mget(self, keys: Sequence[str]) -> list[int | None]:
        self.mget_calls += 1
        return [self.store.get(k) for k in keys]

    @override
    def mset(self, key_value_pairs: Sequence[tuple[str, int]]) -> None:
        self.mset_calls += 1
        for k, v in key_value_pairs:
            self.store[k] = v

    @override
    def mdelete(self, keys: Sequence[str]) -> None:
        self.mdelete_calls += 1
        for k in keys:
            self.store.pop(k, None)

    @override
    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        self.yield_calls += 1
        for k in self.store:
            if prefix is None or k.startswith(prefix):
                yield k


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


def test_base_store_is_abstract() -> None:
    """`BaseStore` cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseStore()  # type: ignore[abstract]


def test_incomplete_subclass_remains_abstract() -> None:
    """Subclasses missing required abstract methods stay abstract."""

    class Missing(BaseStore[str, int]):
        @override
        def mget(self, keys: Sequence[str]) -> list[int | None]:
            return [None for _ in keys]

        # Intentionally missing `mset`, `mdelete`, and `yield_keys`.

    with pytest.raises(TypeError):
        Missing()  # type: ignore[abstract]


def test_byte_store_alias() -> None:
    """`ByteStore` is an alias for `BaseStore[str, bytes]`."""
    assert ByteStore.__origin__ is BaseStore  # type: ignore[attr-defined]
    assert ByteStore.__args__ == (str, bytes)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Default async fallbacks via `run_in_executor`
# ---------------------------------------------------------------------------


async def test_default_amget_uses_sync_implementation() -> None:
    store = SyncOnlyStore()
    store.mset([("a", 1), ("b", 2)])

    result = await store.amget(["a", "b", "missing"])

    assert result == [1, 2, None]
    assert store.mget_calls >= 1


async def test_default_amset_uses_sync_implementation() -> None:
    store = SyncOnlyStore()

    await store.amset([("k", 99)])

    assert store.store == {"k": 99}
    assert store.mset_calls == 1


async def test_default_amdelete_uses_sync_implementation() -> None:
    store = SyncOnlyStore()
    await store.amset([("k", 1), ("kept", 2)])

    await store.amdelete(["k"])

    assert store.store == {"kept": 2}
    assert store.mdelete_calls == 1


async def test_default_amdelete_missing_key_is_noop() -> None:
    store = SyncOnlyStore()
    # Should not raise.
    await store.amdelete(["nope"])
    assert store.store == {}


async def test_default_ayield_keys_no_prefix() -> None:
    store = SyncOnlyStore()
    await store.amset([("a", 1), ("b", 2), ("c", 3)])

    keys = [k async for k in store.ayield_keys()]

    assert set(keys) == {"a", "b", "c"}


async def test_default_ayield_keys_with_prefix() -> None:
    store = SyncOnlyStore()
    await store.amset([("ab", 1), ("ac", 2), ("bc", 3)])

    keys = [k async for k in store.ayield_keys(prefix="a")]

    assert set(keys) == {"ab", "ac"}


async def test_default_ayield_keys_terminates_on_empty_store() -> None:
    """Async iterator exits cleanly when the store is empty."""
    store = SyncOnlyStore()
    keys = [k async for k in store.ayield_keys()]
    assert keys == []


# ---------------------------------------------------------------------------
# `InMemoryByteStore`
# ---------------------------------------------------------------------------


def test_in_memory_byte_store_round_trip() -> None:
    store = InMemoryByteStore()
    pairs: list[tuple[str, bytes]] = [("k1", b"v1"), ("k2", b"v2")]
    store.mset(pairs)

    assert store.mget(["k1", "k2", "missing"]) == [b"v1", b"v2", None]

    store.mdelete(["k1"])
    assert store.mget(["k1", "k2"]) == [None, b"v2"]

    assert set(store.yield_keys()) == {"k2"}
    assert set(store.yield_keys(prefix="k")) == {"k2"}
    assert list(store.yield_keys(prefix="x")) == []


async def test_in_memory_byte_store_async_round_trip() -> None:
    store = InMemoryByteStore()
    await store.amset([("k", b"v")])
    assert await store.amget(["k", "missing"]) == [b"v", None]
    await store.amdelete(["k"])
    keys = [k async for k in store.ayield_keys()]
    assert keys == []


def test_in_memory_byte_store_is_byte_store() -> None:
    """`InMemoryByteStore` should accept `bytes` values."""
    store: InMemoryByteStore = InMemoryByteStore()
    store.mset([("k", b"data")])
    assert store.mget(["k"]) == [b"data"]


def test_in_memory_store_starts_empty() -> None:
    store = InMemoryStore()
    assert store.store == {}
    assert list(store.yield_keys()) == []


def test_in_memory_store_overwrite_existing_key() -> None:
    store = InMemoryStore()
    store.mset([("k", "v1")])
    store.mset([("k", "v2")])
    assert store.mget(["k"]) == ["v2"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def test_invalid_key_exception_inherits_from_langchain_exception() -> None:
    assert issubclass(InvalidKeyException, LangChainException)
    assert issubclass(InvalidKeyException, Exception)


def test_invalid_key_exception_can_be_raised_and_caught() -> None:
    msg = "bad key"
    with pytest.raises(InvalidKeyException, match=msg):
        raise InvalidKeyException(msg)

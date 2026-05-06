"""Tests for `langchain_core.retrievers`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from typing_extensions import override

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import (
    BaseRetriever,
    LangSmithRetrieverParams,
    RetrieverInput,
    RetrieverOutput,
)

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.runnables import RunnableConfig

# ---------------------------------------------------------------------------
# Fixtures: minimal concrete retrievers
# ---------------------------------------------------------------------------


class HardCodedSyncRetriever(BaseRetriever):
    """Retriever that returns a fixed list of documents from sync only."""

    documents: list[Document]

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # Echo the query as metadata to verify input plumbing.
        return [
            Document(page_content=d.page_content, metadata={**d.metadata, "q": query})
            for d in self.documents
        ]


class NativeAsyncRetriever(BaseRetriever):
    """Retriever overriding `_aget_relevant_documents` with a native impl."""

    documents: list[Document]
    sync_calls: int = 0
    async_calls: int = 0

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        self.sync_calls += 1
        return self.documents

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        self.async_calls += 1
        return self.documents


class LegacyRetriever(BaseRetriever):
    """Old-style retriever without a `run_manager` parameter."""

    documents: list[Document]

    @override
    def _get_relevant_documents(self, query: str) -> list[Document]:  # type: ignore[override]
        return [
            Document(page_content=f"{d.page_content}:{query}") for d in self.documents
        ]


class V1ExtraArgRetriever(BaseRetriever):
    """Retriever that accepts additional keyword arguments."""

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        score_threshold: float = 0.0,
    ) -> list[Document]:
        return [
            Document(
                page_content=query,
                metadata={"score_threshold": score_threshold},
            )
        ]


class _FailingRetriever(BaseRetriever):
    """Always raises to exercise the error path."""

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        msg = f"sync boom: {query}"
        raise RuntimeError(msg)

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        msg = f"async boom: {query}"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Type aliases / `TypedDict`
# ---------------------------------------------------------------------------


def test_module_type_aliases() -> None:
    assert RetrieverInput is str
    # `RetrieverOutput` is a `list[Document]` parameterized alias; just check
    # that its origin is `list`.
    assert getattr(RetrieverOutput, "__origin__", None) is list


def test_langsmith_params_typeddict() -> None:
    """`LangSmithRetrieverParams` should be a `TypedDict` with optional keys."""
    assert LangSmithRetrieverParams.__total__ is False
    assert LangSmithRetrieverParams.__optional_keys__ == frozenset(
        {
            "ls_retriever_name",
            "ls_vector_store_provider",
            "ls_embedding_provider",
            "ls_embedding_model",
        }
    )


# ---------------------------------------------------------------------------
# Subclass detection in `__init_subclass__`
# ---------------------------------------------------------------------------


def test_subclass_with_run_manager_marked_supported() -> None:
    assert HardCodedSyncRetriever._new_arg_supported is True
    assert HardCodedSyncRetriever._expects_other_args is False


def test_legacy_subclass_marked_unsupported() -> None:
    assert LegacyRetriever._new_arg_supported is False
    assert LegacyRetriever._expects_other_args is False


def test_v1_extra_args_detected() -> None:
    assert V1ExtraArgRetriever._new_arg_supported is True
    assert V1ExtraArgRetriever._expects_other_args is True


def test_base_retriever_is_abstract() -> None:
    """`BaseRetriever` cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseRetriever()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# `_get_ls_params` name normalization
# ---------------------------------------------------------------------------


def test_get_ls_params_strips_retriever_suffix() -> None:
    class FooRetriever(HardCodedSyncRetriever):
        pass

    params = FooRetriever(documents=[])._get_ls_params()
    assert params == {"ls_retriever_name": "foo"}


def test_get_ls_params_strips_retriever_prefix() -> None:
    class RetrieverFoo(HardCodedSyncRetriever):
        pass

    params = RetrieverFoo(documents=[])._get_ls_params()
    assert params == {"ls_retriever_name": "foo"}


def test_get_ls_params_falls_back_to_lowercase_name() -> None:
    class CustomThing(HardCodedSyncRetriever):
        pass

    params = CustomThing(documents=[])._get_ls_params()
    assert params == {"ls_retriever_name": "customthing"}


# ---------------------------------------------------------------------------
# `invoke` / `ainvoke`
# ---------------------------------------------------------------------------


def test_invoke_returns_documents() -> None:
    retriever = HardCodedSyncRetriever(documents=[Document(page_content="hello")])

    result = retriever.invoke("query!")

    assert len(result) == 1
    assert result[0].page_content == "hello"
    assert result[0].metadata == {"q": "query!"}


async def test_ainvoke_returns_documents_via_executor_for_sync_only() -> None:
    """When only sync impl is provided, async path is auto-derived."""
    retriever = HardCodedSyncRetriever(documents=[Document(page_content="x")])

    result = await retriever.ainvoke("hi")

    assert [d.page_content for d in result] == ["x"]
    assert result[0].metadata["q"] == "hi"


async def test_native_async_implementation_is_used() -> None:
    retriever = NativeAsyncRetriever(documents=[Document(page_content="y")])

    result = await retriever.ainvoke("q")

    assert result == retriever.documents
    assert retriever.async_calls == 1
    assert retriever.sync_calls == 0


def test_invoke_with_empty_documents_returns_empty_list() -> None:
    retriever = HardCodedSyncRetriever(documents=[])
    assert retriever.invoke("anything") == []


def test_invoke_propagates_extra_kwargs_when_supported() -> None:
    retriever = V1ExtraArgRetriever()
    result = retriever.invoke("hello", score_threshold=0.7)
    assert result[0].metadata == {"score_threshold": 0.7}


def test_invoke_drops_extra_kwargs_when_not_supported() -> None:
    """When the subclass doesn't accept extras, kwargs should not be forwarded."""
    retriever = HardCodedSyncRetriever(documents=[Document(page_content="z")])
    # Should not raise even though `score_threshold` isn't accepted.
    result = retriever.invoke("q", score_threshold=0.9)
    assert len(result) == 1


def test_invoke_uses_legacy_signature() -> None:
    retriever = LegacyRetriever(documents=[Document(page_content="abc")])
    result = retriever.invoke("Q")
    assert result[0].page_content == "abc:Q"


def test_invoke_propagates_exceptions() -> None:
    retriever = _FailingRetriever()
    with pytest.raises(RuntimeError, match="sync boom: ping"):
        retriever.invoke("ping")


async def test_ainvoke_propagates_exceptions() -> None:
    retriever = _FailingRetriever()
    with pytest.raises(RuntimeError, match="async boom: ping"):
        await retriever.ainvoke("ping")


def test_invoke_honors_run_name_in_config() -> None:
    """A custom `run_name` should not break invocation."""
    retriever = HardCodedSyncRetriever(documents=[Document(page_content="ok")])
    config: RunnableConfig = {"run_name": "my_run"}
    assert retriever.invoke("q", config=config)[0].page_content == "ok"


def test_invoke_collects_callback_events() -> None:
    """The `on_retriever_start` / `on_retriever_end` callbacks should fire."""

    class TrackingHandler(BaseCallbackHandler):
        def __init__(self) -> None:
            self.starts: list[str] = []
            self.ends: list[list[Document]] = []
            self.errors: list[BaseException] = []

        @override
        def on_retriever_start(
            self,
            serialized: dict[str, Any] | None,
            query: str,
            **_kwargs: Any,
        ) -> None:
            self.starts.append(query)

        @override
        def on_retriever_end(
            self,
            documents: list[Document] | Any,
            **_kwargs: Any,
        ) -> None:
            self.ends.append(list(documents))

        @override
        def on_retriever_error(self, error: BaseException, **_kwargs: Any) -> None:
            self.errors.append(error)

    handler = TrackingHandler()
    retriever = HardCodedSyncRetriever(documents=[Document(page_content="d")])

    retriever.invoke("hi", config={"callbacks": [handler]})

    assert handler.starts == ["hi"]
    assert len(handler.ends) == 1
    assert handler.ends[0][0].page_content == "d"
    assert handler.errors == []


def test_invoke_callback_error_event_on_failure() -> None:
    class ErrorTracker(BaseCallbackHandler):
        def __init__(self) -> None:
            self.errors: list[BaseException] = []

        @override
        def on_retriever_error(self, error: BaseException, **_kwargs: Any) -> None:
            self.errors.append(error)

    handler = ErrorTracker()
    retriever = _FailingRetriever()

    with pytest.raises(RuntimeError):
        retriever.invoke("nope", config={"callbacks": [handler]})

    assert len(handler.errors) == 1
    assert isinstance(handler.errors[0], RuntimeError)


def test_retriever_tags_and_metadata_attrs() -> None:
    retriever = HardCodedSyncRetriever(
        documents=[],
        tags=["a", "b"],
        metadata={"hello": "world"},
    )
    assert retriever.tags == ["a", "b"]
    assert retriever.metadata == {"hello": "world"}


def test_default_tags_and_metadata_are_none() -> None:
    retriever = HardCodedSyncRetriever(documents=[])
    assert retriever.tags is None
    assert retriever.metadata is None

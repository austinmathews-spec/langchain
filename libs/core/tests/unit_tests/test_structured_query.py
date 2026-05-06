"""Tests for `langchain_core.structured_query`."""

from __future__ import annotations

from typing import Any

import pytest
from typing_extensions import override

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Expr,
    FilterDirective,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
    _to_snake_case,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingVisitor(Visitor):
    """Visitor that records each call and returns identifying tuples."""

    @override
    def visit_operation(self, operation: Operation) -> tuple[str, Operation]:
        return ("operation", operation)

    @override
    def visit_comparison(self, comparison: Comparison) -> tuple[str, Comparison]:
        return ("comparison", comparison)

    @override
    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> tuple[str, StructuredQuery]:
        return ("structured_query", structured_query)


# ---------------------------------------------------------------------------
# `_to_snake_case`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Comparison", "comparison"),
        ("Operation", "operation"),
        ("StructuredQuery", "structured_query"),
        ("Already_snake", "already_snake"),
        ("ABC", "a_b_c"),
        ("", ""),
    ],
)
def test_to_snake_case(name: str, expected: str) -> None:
    assert _to_snake_case(name) == expected


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_operator_values() -> None:
    assert Operator.AND.value == "and"
    assert Operator.OR.value == "or"
    assert Operator.NOT.value == "not"
    # `Operator` is a `str` enum so values compare as strings.
    assert str(Operator.AND.value) == "and"


def test_comparator_values() -> None:
    expected = {
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "contain",
        "like",
        "in",
        "nin",
    }
    assert {c.value for c in Comparator} == expected
    assert str(Comparator.EQ.value) == "eq"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_filter_directive_subclass_relationships() -> None:
    """`Comparison` and `Operation` are both `FilterDirective` subclasses."""
    assert issubclass(Comparison, FilterDirective)
    assert issubclass(Operation, FilterDirective)
    assert issubclass(FilterDirective, Expr)


def test_comparison_construction_and_fields() -> None:
    cmp = Comparison(comparator=Comparator.EQ, attribute="age", value=42)
    assert cmp.comparator == Comparator.EQ
    assert cmp.attribute == "age"
    assert cmp.value == 42
    assert isinstance(cmp, FilterDirective)
    assert isinstance(cmp, Expr)


def test_comparison_positional_construction() -> None:
    cmp = Comparison(Comparator.LT, "age", 18)
    assert cmp.comparator == Comparator.LT
    assert cmp.attribute == "age"
    assert cmp.value == 18


def test_operation_construction_and_fields() -> None:
    inner = Comparison(Comparator.GT, "score", 0)
    op = Operation(operator=Operator.NOT, arguments=[inner])
    assert op.operator == Operator.NOT
    assert op.arguments == [inner]
    assert isinstance(op, FilterDirective)


def test_operation_positional_construction_with_multiple_arguments() -> None:
    a = Comparison(Comparator.EQ, "x", 1)
    b = Comparison(Comparator.EQ, "y", 2)
    op = Operation(Operator.AND, [a, b])
    assert op.operator == Operator.AND
    assert op.arguments == [a, b]


def test_structured_query_default_limit_is_none() -> None:
    sq = StructuredQuery(query="hello", filter=None)
    assert sq.query == "hello"
    assert sq.filter is None
    assert sq.limit is None


def test_structured_query_with_filter_and_limit() -> None:
    flt = Comparison(Comparator.EQ, "color", "red")
    sq = StructuredQuery(query="q", filter=flt, limit=10)
    assert sq.query == "q"
    assert sq.filter == flt
    assert sq.limit == 10


def test_structured_query_with_nested_operation() -> None:
    flt = Operation(
        Operator.AND,
        [
            Comparison(Comparator.EQ, "a", 1),
            Operation(
                Operator.OR,
                [
                    Comparison(Comparator.GT, "b", 0),
                    Comparison(Comparator.LIKE, "c", "%foo%"),
                ],
            ),
        ],
    )
    sq = StructuredQuery(query="nested", filter=flt)
    assert isinstance(sq.filter, Operation)
    assert sq.filter.operator == Operator.AND
    assert len(sq.filter.arguments) == 2


# ---------------------------------------------------------------------------
# Visitor pattern
# ---------------------------------------------------------------------------


def test_expr_accept_dispatches_by_class_name() -> None:
    visitor = _RecordingVisitor()
    cmp = Comparison(Comparator.EQ, "k", "v")
    op = Operation(Operator.AND, [cmp])
    sq = StructuredQuery(query="q", filter=cmp)

    assert cmp.accept(visitor) == ("comparison", cmp)
    assert op.accept(visitor) == ("operation", op)
    assert sq.accept(visitor) == ("structured_query", sq)


def test_visitor_is_abstract() -> None:
    """`Visitor` requires all three visit methods to be implemented."""
    with pytest.raises(TypeError):
        Visitor()  # type: ignore[abstract]


def test_visitor_partial_subclass_still_abstract() -> None:
    class Partial(Visitor):
        @override
        def visit_operation(self, operation: Operation) -> Any:
            return None

        # Missing `visit_comparison` and `visit_structured_query`.

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_validate_func_allows_when_no_restrictions() -> None:
    visitor = _RecordingVisitor()
    # Defaults are `None`, so any operator/comparator should be allowed.
    visitor._validate_func(Operator.AND)
    visitor._validate_func(Comparator.EQ)


def test_validate_func_rejects_disallowed_operator() -> None:
    class Restricted(_RecordingVisitor):
        allowed_operators = [Operator.AND]  # noqa: RUF012

    visitor = Restricted()
    visitor._validate_func(Operator.AND)
    with pytest.raises(ValueError, match="disallowed operator"):
        visitor._validate_func(Operator.OR)


def test_validate_func_rejects_disallowed_comparator() -> None:
    class Restricted(_RecordingVisitor):
        allowed_comparators = [Comparator.EQ, Comparator.NE]  # noqa: RUF012

    visitor = Restricted()
    visitor._validate_func(Comparator.EQ)
    with pytest.raises(ValueError, match="disallowed comparator"):
        visitor._validate_func(Comparator.GT)


def test_validate_func_with_unknown_func_is_noop() -> None:
    """Inputs that are neither `Operator` nor `Comparator` are ignored."""
    visitor = _RecordingVisitor()
    # Should not raise.
    visitor._validate_func("not-an-enum")  # type: ignore[arg-type]

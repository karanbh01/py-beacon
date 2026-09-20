# tests/test_look_ahead.py
"""BN-208: a backtest must never read an observation dated after the day it
is valuing, and that must be structurally true rather than remembered.

Two look-aheads shipped in FX conversion (BN-204) and neither was noticed by
anybody reading the code, because both *had a guard* and both guards looked
careful. The mechanism is a pandas trap rather than carelessness:

    position = index.searchsorted(date, side="right") - 1

returns **-1** when every observation is dated after `date`. In pandas -1 is
a legal index meaning the last element, so the honest value for "nothing to
find" is silently a lookup returning the newest observation in the series —
the largest look-ahead available. Both sites clamped it:

    series.iloc[max(position, 0)]      # the FIRST rate, still the future
    if position < 0: series.iloc[0]    # the same, written out

Which swaps a large look-ahead for a small one. `as_of_position` returns None
instead, because None is the only answer a caller cannot accidentally index
with.

This file is the backstop for the rule, not for the instance.
"""
import ast
from pathlib import Path

import pandas as pd
import pytest

from beacon.data.base import as_of_position

SOURCE_ROOT = Path(__file__).resolve().parent.parent / "src" / "beacon"

# The one function allowed to perform the raw search. Everything else asks it.
PRIMITIVE_FILE = SOURCE_ROOT / "data" / "base.py"


def as_of_searches(tree: ast.AST) -> list[int]:
    """Line numbers of `<index>.searchsorted(..., side="right")` calls.

    That signature is the as-of search specifically: `side="right"` asks for
    the insertion point *after* equal entries, which is what "at or before"
    needs and what the -1 sentinel comes out of. A bare `searchsorted` is a
    different question — `data/adjustment.py` uses one to find where an
    ex-date sits, which is not a point-in-time read — so this does not flag it.
    """
    found = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function = node.func

        if not isinstance(function, ast.Attribute):
            continue

        if function.attr != "searchsorted":
            continue

        if any(keyword.arg == "side"
               and isinstance(keyword.value, ast.Constant)
               and keyword.value.value == "right"
               for keyword in node.keywords):
            found.append(node.lineno)

    return found


class TestNothingReadsAheadOfItsDate:

    def test_only_the_primitive_performs_the_as_of_search(self):
        """The rule, enforced rather than documented.

        Deliberately without an allowlist. A ratchet with exceptions would
        have accepted both of the sites that shipped the bug — each was added
        as a reasonable-looking exception at the time.
        """
        offenders: list[str] = []

        for path in SOURCE_ROOT.rglob("*.py"):
            if path == PRIMITIVE_FILE:
                continue

            tree = ast.parse(path.read_text(encoding="utf-8"))

            offenders += [
                f"{path.relative_to(SOURCE_ROOT)}:{line}"
                for line in as_of_searches(tree)]

        assert offenders == [], (
            "these perform their own as-of search instead of calling "
            "`as_of_position`, which is how BN-204 shipped twice:\n  "
            + "\n  ".join(offenders))

    def test_the_primitive_refuses_rather_than_clamping(self):
        """A date before everything has no answer, not the earliest one."""
        index = pd.to_datetime(["2025-03-03", "2025-04-01", "2025-05-01"])

        assert as_of_position(index, pd.Timestamp("2025-01-15")) is None

    def test_it_never_returns_a_negative_position(self):
        """The whole defect in one assertion. -1 is a valid pandas index, so
        a sentinel that can be indexed with is not a sentinel."""
        index = pd.to_datetime(["2025-03-03", "2025-04-01"])

        for day in ("2024-01-01", "2025-01-15", "2025-03-02"):
            position = as_of_position(index, day)

            assert position is None or position >= 0

    def test_a_date_on_an_observation_takes_that_observation(self):
        index = pd.to_datetime(["2025-03-03", "2025-04-01", "2025-05-01"])

        assert as_of_position(index, pd.Timestamp("2025-04-01")) == 1

    def test_a_date_between_observations_takes_the_earlier(self):
        """The carry, which is the behaviour that must survive the fix."""
        index = pd.to_datetime(["2025-03-03", "2025-04-01", "2025-05-01"])

        assert as_of_position(index, pd.Timestamp("2025-04-15")) == 1

    def test_a_date_after_everything_takes_the_last(self):
        index = pd.to_datetime(["2025-03-03", "2025-04-01", "2025-05-01"])

        assert as_of_position(index, pd.Timestamp("2026-01-01")) == 2

    @pytest.mark.parametrize("day", ["2025-01-15", "2025-04-15", "2026-01-01"])
    def test_the_answer_is_never_dated_after_the_question(self,
                                                          day):
        """The invariant itself, rather than a case of it."""
        index = pd.to_datetime(["2025-03-03", "2025-04-01", "2025-05-01"])
        position = as_of_position(index, day)

        if position is not None:
            assert index[position] <= pd.Timestamp(day)

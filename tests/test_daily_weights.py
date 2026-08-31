# tests/test_daily_weights.py
"""BN-153: the calculator records what the index held on every day.

The rebalance snapshots say what a rebalance *decided*. They cannot say what
the index held on a Tuesday in the middle of a month, and the difference is
not cosmetic:

* prices move, so the realised weights drift away from the decided ones;
* a delisting removes a name and renormalises the rest, on a date nobody
  chose in advance.

So the daily panel is *recorded* as the calculator walks, and the tests below
are about the two claims that matter — that the record is arithmetically the
index (the divisor identity), and that it shows the events a forward-fill of
the snapshots would miss (the deletion).

The data is hand-built rather than generated: prices follow closed-form
geometric paths with a different drift per name, so every weight in the panel
is a number this file can reproduce, and a failure is a defect rather than a
seed.
"""

import numpy as np
import pandas as pd
import pytest

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.result import IndexResult, daily_weights_frame

START = "2021-01-04"
END = "2021-06-30"

# Base price, daily drift, shares outstanding. Drifts differ so the weights
# have something to drift *with*: over a month the leader gains about 4% on
# the laggard, which is far outside any tolerance below.
NAMES = {
    "AAA": (100.0, 0.0012, 1_000_000.0),
    "BBB": (50.0, 0.0004, 3_000_000.0),
    "CCC": (250.0, -0.0006, 400_000.0),
    "DDD": (75.0, 0.0000, 2_000_000.0),
    "EEE": (120.0, 0.0008, 800_000.0),
}

# Mid-month, and deliberately not a rebalance date: the first business day of
# March is, and this is not near it.
DELIST_ON = "2021-02-15"
FIRST_MISSING = pd.Timestamp("2021-02-16")


def build_fetcher(delist_on: str | None = None) -> DataFetcher:
    """Five names on closed-form price paths; one may stop being listed."""
    dates = pd.bdate_range(START, END)
    leaver = pd.Timestamp(delist_on) if delist_on else None

    rows = []
    for identifier, (base, drift, shares) in NAMES.items():
        for step, date in enumerate(dates):
            if identifier == "EEE" and leaver is not None and date > leaver:
                continue

            price = base * (1.0 + drift) ** step
            rows.append({"IDENTIFIER": identifier, "DATE": date,
                         "OPEN": price, "HIGH": price, "LOW": price,
                         "CLOSE": price, "VOLUME": 1_000.0,
                         "SHARES_OUTSTANDING": shares, "FREE_FLOAT": 1.0})

    reference = pd.DataFrame([
        {"IDENTIFIER": identifier, "DATE_FROM": START,
         "DATE_TO": leaver if identifier == "EEE" else pd.NaT,
         "NAME": identifier, "CURRENCY": "USD", "EXCHANGE": "XNAS"}
        for identifier in NAMES
    ])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def run_index(fetcher: DataFetcher) -> IndexResult:
    """An equal-weighted, monthly-rebalanced index over the five names."""
    definition = IndexDefinition(
        index_id="DW", index_name="DW", base_date=START, base_value=1000.0,
        currency="USD", eligibility_rules=[],
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="MONTHLY",
        universe_identifiers=list(NAMES))

    return IndexCalculator(definition, fetcher).run(start_date=START,
                                                    end_date=END)


def price_lookup(fetcher: DataFetcher) -> dict[tuple[str, pd.Timestamp], float]:
    """Every close, read back through the same interface the calculator uses.

    Read from the fetcher rather than recomputed from :data:`NAMES`, so the
    identity test compares the recorded panel against the prices the
    calculator actually saw rather than against this file's own arithmetic.
    """
    prices: dict[tuple[str, pd.Timestamp], float] = {}

    for identifier in NAMES:
        frame = fetcher.fetch_market_data(identifier, START, END)

        for date, close in frame["CLOSE"].items():
            prices[(identifier, date)] = float(close)

    return prices


@pytest.fixture(scope="module")
def intact() -> IndexResult:
    """The run where nothing ever leaves."""
    return run_index(build_fetcher())


@pytest.fixture(scope="module")
def with_deletion() -> IndexResult:
    """The run where EEE stops being listed mid-February."""
    return run_index(build_fetcher(delist_on=DELIST_ON))


class TestThePanel:
    """Shape, coverage, and the dtypes it is stored in."""

    def test_it_covers_every_calculation_day(self,
                                             intact):
        panel = intact.daily_weights

        assert set(panel["DATE"]) == set(intact.index_levels.index)

    def test_it_names_every_constituent_every_day(self,
                                                  intact):
        counts = intact.daily_weights.groupby("DATE", observed=True).size()

        assert (counts == len(NAMES)).all()

    def test_the_columns_are_the_long_form_four(self,
                                                intact):
        assert list(intact.daily_weights.columns) == [
            "DATE", "IDENTIFIER", "AMOUNT", "WEIGHT"]

    def test_the_weights_sum_to_one_on_every_day(self,
                                                 intact):
        """float32 storage, ~5 names: a relative error near 1e-7 a value."""
        totals = intact.daily_weights.groupby(
            "DATE", observed=True)["WEIGHT"].sum()

        assert np.allclose(totals.to_numpy(), 1.0, rtol=1e-5), (
            f"weights sum between {totals.min():.8f} and {totals.max():.8f}")

    def test_the_amounts_hold_still_between_rebalances(self,
                                                       intact):
        """AMOUNT is units held, not a weight: fixed until the next rebalance.

        This is what makes the weights drift rather than being silently reset
        every day, so it is worth pinning separately from the weights.
        """
        rebalances = sorted(intact.weight_snapshots)
        panel = intact.daily_weights
        month = panel[(panel["DATE"] >= rebalances[1])
                      & (panel["DATE"] < rebalances[2])]

        spread = month.groupby("IDENTIFIER", observed=True)["AMOUNT"].nunique()

        assert (spread == 1).all(), (
            f"amounts moved between rebalances: {spread.to_dict()}")

    def test_it_is_stored_compactly(self,
                                    intact):
        """The dtypes are the storage decision, so they are asserted."""
        dtypes = intact.daily_weights.dtypes

        assert str(dtypes["DATE"]) == "datetime64[ns]"
        assert str(dtypes["IDENTIFIER"]) == "category"
        assert str(dtypes["AMOUNT"]) == "float32"
        assert str(dtypes["WEIGHT"]) == "float32"


class TestTheDivisorIdentity:
    """Σ (price × amount) ÷ divisor == the stored level, every day.

    The index book's pinned invariant. A missed deletion, a divisor applied
    twice, or a panel written from stale units fails here on the exact date it
    happened — which is the whole reason the panel is worth storing.
    """

    def check(self,
              result: IndexResult,
              fetcher: DataFetcher) -> None:
        prices = price_lookup(fetcher)
        panel = result.daily_weights

        rebuilt = {}
        for row in panel.itertuples():
            rebuilt[row.DATE] = rebuilt.get(row.DATE, 0.0) + (
                prices[(str(row.IDENTIFIER), row.DATE)] * float(row.AMOUNT))

        dates = sorted(rebuilt)
        levels = np.array([rebuilt[date] / float(result.divisor_history.loc[date])
                           for date in dates])
        stored = result.index_levels.loc[dates].to_numpy()

        # rtol=1e-4 rather than machine precision because AMOUNT is stored as
        # float32 — about seven significant digits, so each product carries a
        # relative error near 1e-7 and the sum of five of them near 1e-6. The
        # tolerance is two orders of magnitude above that measured floor: wide
        # enough that float32 alone can never fail it, and far too tight to
        # let a wrong holding, a missed deletion or a stale divisor through
        # (the smallest of those is a 20% move in this index).
        assert np.allclose(levels, stored, rtol=1e-4), (
            f"worst relative error "
            f"{np.max(np.abs(levels - stored) / stored):.2e}")

    def test_it_holds_for_a_plain_run(self,
                                      intact):
        self.check(intact, build_fetcher())

    def test_it_holds_across_a_deletion(self,
                                        with_deletion):
        self.check(with_deletion, build_fetcher(delist_on=DELIST_ON))


class TestADeletionShows:
    """The case that distinguishes a record from a forward-fill."""

    def test_the_leaver_stops_appearing_on_a_day_that_is_no_rebalance(
            self,
            with_deletion):
        """The claim in one test.

        A panel forward-filled from the rebalance snapshots would carry EEE
        all the way to the March rebalance, because that is the next date the
        snapshots have anything to say. The recorded panel drops it the day
        after its listing ended — and that day is deliberately asserted *not*
        to be a rebalance, since on a rebalance date the two would agree and
        the test would prove nothing.
        """
        panel = with_deletion.daily_weights
        held_on = panel.loc[panel["IDENTIFIER"] == "EEE", "DATE"]

        assert FIRST_MISSING not in set(with_deletion.weight_snapshots), (
            "the deletion date is a rebalance, so this proves nothing")
        assert held_on.max() == FIRST_MISSING - pd.offsets.BDay(1)
        assert (held_on < FIRST_MISSING).all()

    def test_the_snapshot_still_carries_it(self,
                                           with_deletion):
        """The rebalance snapshots are untouched by any of this: February's
        composition included EEE, and that remains true afterwards."""
        february = min(date for date in with_deletion.weight_snapshots
                       if date >= pd.Timestamp("2021-02-01"))

        assert "EEE" in with_deletion.weight_snapshots[february]

    def test_the_survivors_renormalise_that_day(self,
                                                with_deletion):
        """Deleting a name lifts everyone else pro rata — it does not leave a
        hole where its weight used to be."""
        before = with_deletion.weights_on(FIRST_MISSING - pd.offsets.BDay(1))
        after = with_deletion.weights_on(FIRST_MISSING)

        assert sum(after.values()) == pytest.approx(1.0, rel=1e-5)
        assert set(after) == set(before) - {"EEE"}
        assert all(after[name] > before[name] for name in after)

    def test_the_amounts_of_the_survivors_do_not_move(self,
                                                      with_deletion):
        """A deletion is a divisor adjustment, not a trade: the survivors hold
        exactly what they held the day before, and only their *share* moves.
        """
        panel = with_deletion.daily_weights
        pair = panel[panel["DATE"].isin([FIRST_MISSING,
                                         FIRST_MISSING - pd.offsets.BDay(1)])]
        survivors = pair[pair["IDENTIFIER"] != "EEE"]

        spread = survivors.groupby("IDENTIFIER", observed=True)["AMOUNT"].nunique()

        assert (spread == 1).all()


class TestTheWeightsDrift:
    """A mid-period day is not the rebalance that preceded it."""

    def test_a_day_between_rebalances_differs_from_the_snapshot(self,
                                                               intact):
        """Equal weighting sets five weights to 0.2 each; three weeks of
        different drifts pulls them apart. The recorded panel shows that, the
        snapshot cannot.
        """
        rebalances = sorted(intact.weight_snapshots)
        drifted = rebalances[1] + pd.offsets.BDay(15)

        recorded = intact.weights_on(drifted)
        decided = intact.get_weights_on_date(drifted)

        assert drifted not in set(rebalances), (
            "the drifted date is itself a rebalance, so this proves nothing")
        assert set(recorded) == set(decided)
        moves = {name: abs(recorded[name] - decided[name]) for name in recorded}

        assert max(moves.values()) > 1e-3, (
            f"weights barely moved in twenty sessions: {moves}")

    def test_they_agree_on_the_rebalance_date_itself(self,
                                                     intact):
        """The other half of the claim. If the panel disagreed with the
        snapshot on the rebalance date too, the drift above would be a bug
        rather than the market.
        """
        rebalance = sorted(intact.weight_snapshots)[1]

        recorded = intact.weights_on(rebalance)
        decided = intact.get_weights_on_date(rebalance)

        for name, weight in decided.items():
            assert recorded[name] == pytest.approx(weight, rel=1e-5)


class TestWeightsOn:
    """The accessor, including the cases with no panel behind it."""

    def test_it_falls_back_to_the_latest_recorded_date(self,
                                                       intact):
        """A Saturday has no record; the answer is Friday's."""
        friday = pd.Timestamp("2021-03-05")
        saturday = pd.Timestamp("2021-03-06")

        assert intact.weights_on(saturday) == intact.weights_on(friday)

    def test_it_is_empty_before_the_first_record(self,
                                                 intact):
        assert intact.weights_on(pd.Timestamp("2020-12-31")) == {}

    def test_it_is_empty_when_nothing_recorded_a_panel(self):
        """Every existing caller builds an IndexResult without one."""
        result = IndexResult(
            index_id="X",
            index_levels=pd.Series([100.0], index=[pd.Timestamp("2021-01-04")]),
            divisor_history=pd.Series([1.0], index=[pd.Timestamp("2021-01-04")]),
            constituent_snapshots={},
            weight_snapshots={})

        assert result.daily_weights.empty
        assert list(result.daily_weights.columns) == [
            "DATE", "IDENTIFIER", "AMOUNT", "WEIGHT"]
        assert result.weights_on(pd.Timestamp("2021-01-04")) == {}

    def test_it_returns_plain_python(self,
                                     intact):
        """A dict of str to float, so a caller needs no pandas to read it."""
        weights = intact.weights_on(pd.Timestamp("2021-03-05"))

        assert all(type(name) is str for name in weights)
        assert all(type(weight) is float for weight in weights.values())


class TestWhatItCosts:
    """Storage, measured rather than assumed.

    Built from records directly rather than from a run: the size question is
    about the frame, and a thousand-name index calculation would cost minutes
    to answer it. A thousand names is enough to put the identifier codes in
    the same int16 regime a six-thousand-name panel would use, so the
    per-row figure extrapolates honestly.
    """

    ROWS_PER_YEAR = 252

    def measured(self) -> float:
        names = [f"N{i:04d}" for i in range(1_000)]
        dates = pd.bdate_range("2021-01-04", periods=60)

        records = [{"DATE": date, "IDENTIFIER": name,
                    "AMOUNT": 1_234.5, "WEIGHT": 0.001}
                   for date in dates for name in names]

        frame = daily_weights_frame(records)

        return float(frame.memory_usage(deep=True).sum()) / len(frame)

    def test_a_row_costs_under_twenty_bytes(self):
        """Measured at 18.9 bytes a row: 8 for the date, 2 for the categorical
        identifier code, 4 each for the two float32 columns, plus the category
        strings amortised across 60,000 rows (0.9 a row here, and ten times
        less at the scale below, since the same 6,000 strings are shared by
        far more rows).

        Extrapolating to the shape that matters — 6,000 names over ten years,
        15.1M rows — that is ~286 MB, of which ~60 MB is each float32 column.
        The same panel in pandas' default dtypes measures 78.0 bytes a row,
        or ~1.18 GB: 54 bytes of it is the identifier stored as a string.
        That factor of four is what the dtypes in result.py buy.
        """
        bytes_per_row = self.measured()

        assert bytes_per_row < 20.0, (
            f"a panel row costs {bytes_per_row:.1f} bytes")

    def test_the_extrapolation_is_stated_in_bytes(self):
        """The number the docstring above quotes, computed rather than typed,
        so it cannot rot silently."""
        total_gb = (self.measured() * 6_000 * 10 * self.ROWS_PER_YEAR) / 1e9

        assert total_gb < 0.35, (
            f"6,000 names over ten years would cost {total_gb:.2f} GB")

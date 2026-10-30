# tests/test_fast_reads.py
"""BN-411: the faster paths of the factor analytics and the books give what
the paths they replaced gave.

Each name's last market value in a window from the session index; features
read once over every rebalance; the factor returns fitted a stretch at a
time; the fund's daily weights put straight into a grid; the positions'
dates converted once a day. Each against the old calculation, on data built
to reach its edges: gaps, missing values, names with no rows, repeated rows.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.risk.factors import _cross_sections
from beacon.analysis.risk.read_once import FeaturesReadOnce
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.expressions import resolve
from beacon.expressions.core import Field
from beacon.portfolio.base import _pivoted

DAYS = pd.bdate_range("2024-01-02", periods=30)
NAMES = ["A", "B", "C", "D"]


def market() -> MarketData:
    """Four names over thirty sessions: B misses a week, C has a missing
    close on some days, D stops early, and E is asked about but never
    printed."""
    random = np.random.default_rng(3)
    rows = []

    for day_number, day in enumerate(DAYS):
        for name in NAMES:
            if name == "B" and 10 <= day_number < 15:
                continue
            if name == "D" and day_number > 20:
                continue

            close = float(np.round(50 + random.random() * 10, 2))
            rows.append({"IDENTIFIER": name, "DATE": day,
                         "CLOSE": np.nan if name == "C" and day_number % 4 == 0 else close,
                         "SHARES_OUTSTANDING": 1e6 + day_number, "VOLUME": 1e5})

    return MarketData.from_dataframe(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def fetcher() -> DataFetcher:
    reference = pd.DataFrame([{"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
                               "CURRENCY": "USD", "EXCHANGE": "XNYS"} for name in NAMES])

    return DataFetcher(market(), ReferenceData.from_dataframe(reference))


class WindowRead:
    """The fetcher without its session-index read, so the window is read."""

    def __init__(self,
                 inner: DataFetcher):
        self.inner = inner

    def __getattr__(self,
                    name: str):
        if name == "last_market_values":
            raise AttributeError(name)

        return getattr(self.inner, name)


class TestLastMarketValues:

    @pytest.mark.parametrize("column", ["CLOSE", "SHARES_OUTSTANDING", "VOLUME", "MARKET_CAP"])
    def test_each_name_s_last_value_is_the_window_read_s(self,
                                                          fetcher,
                                                          column):
        field = Field("market", column)
        asked = [*NAMES, "E"]

        for day in DAYS[5:]:
            assert (resolve._market_numbers(field, asked, day, fetcher, "USD")
                    == resolve._market_numbers(field, asked, day, WindowRead(fetcher), "USD")), day

    def test_a_gap_and_a_missing_close_take_the_last_value_before(self,
                                                                  fetcher):
        values = fetcher.last_market_values(["B", "C"], DAYS[8], DAYS[13], "CLOSE")
        frame = fetcher.fetch_market_data(["B", "C"], str(DAYS[8].date()), str(DAYS[13].date()))
        expected = frame["CLOSE"].groupby(level=0).last()

        np.testing.assert_array_equal(values, expected.reindex(["B", "C"]).to_numpy())

    def test_a_name_with_nothing_in_the_window_is_missing(self,
                                                          fetcher):
        values = fetcher.last_market_values(["D", "E"], DAYS[25], DAYS[29], "CLOSE")

        assert np.isnan(values).all()

    def test_a_column_it_cannot_read_says_so(self,
                                             fetcher):
        assert fetcher.last_market_values(["A"], DAYS[0], DAYS[5], "NOPE") is None


class TestFeaturesReadOnce:

    class Recording:
        def __init__(self):
            self.calls = []

        def table_as_of(self, identifiers, field, dates, feature_type=None, max_age_days=730):
            self.calls.append((list(identifiers), list(dates)))
            index = pd.DatetimeIndex(dates)

            return pd.DataFrame({name: [hash((name, day)) % 97 for day in index]
                                 for name in identifiers}, index=index, dtype=float)

    def test_a_rebalance_s_read_is_cut_from_one_read(self):
        source = self.Recording()
        features = FeaturesReadOnce(source, NAMES, list(DAYS[:3]))

        first = features.table_as_of(["B", "A"], "PB", [DAYS[1]])
        second = features.table_as_of(["C"], "PB", [DAYS[2]])

        assert len(source.calls) == 1
        pd.testing.assert_frame_equal(first, source.table_as_of(["B", "A"], "PB", [DAYS[1]]))
        pd.testing.assert_frame_equal(second, source.table_as_of(["C"], "PB", [DAYS[2]]))

    def test_a_read_it_does_not_cover_goes_to_the_source(self):
        source = self.Recording()
        features = FeaturesReadOnce(source, NAMES, list(DAYS[:3]))

        features.table_as_of(["Z"], "PB", [DAYS[1]])
        features.table_as_of(["A"], "PB", [DAYS[9]])

        assert [call[0] for call in source.calls] == [["Z"], ["A"]]


def one_day(returns: np.ndarray,
            exposures: np.ndarray,
            caps: np.ndarray) -> list[float] | None:
    """The fit a day at a time, as BN-387 made it."""
    known = ~np.isnan(returns) & (caps > 0.0)

    if known.sum() <= exposures.shape[1] + 1:
        return None

    design = np.column_stack([np.ones(int(known.sum())), exposures[known]])
    root = np.sqrt(caps[known] / caps[known].sum())
    solved, *_ = np.linalg.lstsq(design * root[:, None], returns[known] * root, rcond=None)

    return [float(value) for value in solved]


class TestFactorReturnsByStretch:

    def test_each_day_is_its_fit_on_its_own(self):
        random = np.random.default_rng(7)
        returns = random.normal(0.0, 0.01, (40, 30))
        returns[5, 3] = returns[12, [3, 4]] = np.nan
        returns[20, :26] = np.nan
        exposures = random.normal(0.0, 1.0, (30, 3))
        caps = random.random(30) * 1e9
        caps[7] = 0.0

        fitted = _cross_sections(returns, exposures, caps)

        for day in range(len(returns)):
            alone = one_day(returns[day], exposures, caps)

            if alone is None:
                assert day not in fitted
            else:
                np.testing.assert_allclose(fitted[day], alone, rtol=1e-12, atol=1e-15)

        assert 20 not in fitted


class TestTheFundsWeights:

    def positions(self,
                  rows: list[tuple[str, str, float]]) -> pd.DataFrame:
        return pd.DataFrame({"DATE": pd.to_datetime([row[0] for row in rows]),
                             "ASSET_ID": pd.array([row[1] for row in rows], dtype="str"),
                             "WEIGHT": [row[2] for row in rows]})

    def test_they_are_the_pivot_s_exactly_and_laid_out_as_it(self):
        positions = self.positions([("2024-01-03", "B", 0.6), ("2024-01-02", "A", 0.5),
                                    ("2024-01-02", "B", 0.5), ("2024-01-03", "C", 0.4)])
        pivot = positions.pivot_table(index="DATE", columns="ASSET_ID", values="WEIGHT",
                                      observed=True)

        found = _pivoted(positions)

        pd.testing.assert_frame_equal(found, pivot, check_exact=True)
        assert found.to_numpy().flags.c_contiguous == pivot.to_numpy().flags.c_contiguous

    @pytest.mark.parametrize("rows", [
        [("2024-01-02", "A", 0.2), ("2024-01-02", "A", 0.4), ("2024-01-02", "B", 0.4)],
        [("2024-01-02", "A", float("nan")), ("2024-01-02", "B", 1.0)],
    ])
    def test_repeated_or_missing_weights_are_left_to_the_pivot(self,
                                                              rows):
        positions = self.positions(rows)

        pd.testing.assert_frame_equal(_pivoted(positions), positions.pivot_table(
            index="DATE", columns="ASSET_ID", values="WEIGHT", observed=True))


def test_the_positions_dates_are_each_snapshot_s():
    from test_strategy_futures import index, run

    history = run(index(), transaction_cost_bps=5.0).portfolio._history
    snapshots = [(day, history._positions[day]) for day in sorted(history._positions)]
    old = pd.to_datetime([day for day, snapshot in snapshots for _ in snapshot.asset_ids])

    pd.testing.assert_series_equal(history._build_positions()["DATE"],
                                   pd.Series(old, name="DATE"), check_exact=True)

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


class TestBN416:
    """BN-416: attribution, rolling betas and momentum against the code they
    replaced."""

    @pytest.fixture(scope="class")
    def factors(self):
        from beacon.analysis.data import RunData
        from beacon.analysis.risk.factors import FactorAnalytics
        from beacon.backtest import Backtest
        from test_strategy_futures import DATA, END, START, index

        result = Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                          transaction_cost_bps=5.0).run(index(), start=START, end=END)

        return FactorAnalytics(RunData.from_result(result), DATA)

    @staticmethod
    def parts_day_by_day(factors, lagged, name_returns):
        """BN-395's holdings-based parts, a day at a time."""
        from beacon.analysis.attribution.model import SPECIFIC
        from beacon.data.base import as_of_position
        from beacon.risk.model import MARKET_FACTOR

        found = factors.factor_returns()
        rebalances = pd.DatetimeIndex(factors.dates)
        rows = {}

        for day in lagged.index.intersection(found.index):
            at = as_of_position(rebalances, day - pd.Timedelta(days=1))

            if at is None:
                continue

            held = lagged.loc[day]
            held = held[held != 0.0]
            z = factors.exposures(rebalances[at]).reindex(held.index).fillna(0.0)
            earned = name_returns.loc[day].reindex(held.index).fillna(0.0)
            paid = found.loc[day]
            own = earned - (paid[MARKET_FACTOR] + z @ paid[factors.names])
            exposure = z.T @ held
            rows[day] = {MARKET_FACTOR: float(held.sum() * paid[MARKET_FACTOR]),
                         **{name: float(exposure[name] * paid[name]) for name in factors.names},
                         SPECIFIC: float((held * own).sum())}

        return pd.DataFrame.from_dict(rows, orient="index")

    def test_the_holdings_parts_by_stretch_are_the_day_by_day_ones(self,
                                                                   factors):
        from beacon.analysis.attribution.model import holdings_parts
        from beacon.analysis.panels import FUND_WEIGHTS, NAME_RETURNS

        lagged = factors.run.tables[FUND_WEIGHTS].fillna(0.0).shift(1).fillna(0.0)
        lagged.columns = lagged.columns.astype(str)
        name_returns = factors.run.tables[NAME_RETURNS]

        ours = holdings_parts(factors, lagged, name_returns)
        theirs = self.parts_day_by_day(factors, lagged, name_returns)

        assert list(ours.columns) == list(theirs.columns) and ours.index.equals(theirs.index)
        np.testing.assert_allclose(ours.to_numpy(), theirs.to_numpy(), rtol=1e-12, atol=1e-16)

    def test_the_rolling_betas_are_each_window_s_own_fit(self,
                                                         factors):
        from beacon.analysis.risk.factors import regressed

        target, paid = factors._paired("fund")
        rolled = factors.rolling_betas(window=63)

        for end in range(63, len(target) + 1, 37):
            fit = regressed(target.iloc[end - 63:end], paid.iloc[end - 63:end], 252)
            np.testing.assert_allclose(rolled.loc[target.index[end - 1]].to_numpy(),
                                       fit.betas.to_numpy(), rtol=1e-12, atol=1e-14)

    def test_a_near_singular_window_is_fitted_by_least_squares(self):
        from beacon.analysis.risk.factors import regressed, rolling_fits

        random = np.random.default_rng(5)
        days = pd.bdate_range("2024-01-02", periods=80)
        paid = pd.DataFrame(random.normal(0.0, 0.01, (80, 2)), index=days, columns=["a", "b"])
        paid["c"] = paid["a"] * 2.0
        target = pd.Series(random.normal(0.0, 0.01, 80), index=days)

        rolled = rolling_fits(target, paid, 40)

        np.testing.assert_allclose(rolled.iloc[-1].to_numpy(),
                                   regressed(target.iloc[-40:], paid.iloc[-40:], 252)
                                   .betas.to_numpy(), rtol=1e-9, atol=1e-12)

    def test_contributions_are_each_name_s_own(self):
        from beacon.analysis.attribution.contribution import attribute

        random = np.random.default_rng(9)
        days = pd.bdate_range("2024-01-02", periods=60)
        names = [f"N{number}" for number in range(40)]
        weights = pd.DataFrame(random.random((60, 40)), index=days, columns=names)
        weights = weights.div(weights.sum(axis=1), axis=0)
        weights.iloc[10:20, :5] = 0.0
        returns = pd.DataFrame(random.normal(0.0, 0.01, (60, 40)), index=days, columns=names)
        returns.iloc[::9, 3] = np.nan
        total = (weights.shift(1) * returns).sum(axis=1)

        result = attribute(total, weights, returns)
        lagged = weights.shift(1)
        kept = (lagged * returns).dropna(how="all").index

        for item in result.contributions:
            assert item.average_weight == float(lagged[item.asset_id].mean())
            assert item.total_return == float(
                (1.0 + returns[item.asset_id].reindex(kept).fillna(0.0)).prod() - 1.0)

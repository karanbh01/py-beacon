# tests/test_futures_pricing.py
"""BN-409: what index futures are priced at, beyond fair value.

Runs reuse test_strategy_futures' eight-name market-cap index through 2024,
held through fractional futures with no dividend yield and no cash rate, so
with no mispricing the price is the index level itself. The data adds two
series: FUT, quoted 0.2% above that level on every day but the first ten,
and FLAT, an index that never moves.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.assumptions import ModellingAssumptions
from beacon.backtest import Backtest
from beacon.data.base import MarketData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.strategy.futures import IndexFutures
from test_strategy_futures import DATA, DAYS, END, NO_YIELD, START, futures, run


def with_series(levels: pd.Series) -> DataFetcher:
    """DATA with FUT quoted 0.2% above *levels* after its first ten days, and
    FLAT at 1,000 throughout."""
    rows = [*({"IDENTIFIER": "FUT", "DATE": day, "CLOSE": level * 1.002}
              for day, level in levels.iloc[10:].items()),
            *({"IDENTIFIER": "FLAT", "DATE": day, "CLOSE": 1000.0} for day in DAYS)]
    market = pd.concat([DATA.market.data.reset_index(), pd.DataFrame(rows)])

    return DataFetcher(MarketData.from_dataframe(market), DATA.reference, DATA.corporate_actions)


def run_on(data: DataFetcher,
           strategy):
    return Backtest(initial_capital=1e7, data_provider=data, cache=False,
                    modelling_assumptions=ModellingAssumptions(cash_rate=0.0)).run(
        strategy, start=START, end=END)


@pytest.fixture(scope="module")
def plain():
    return run(futures(**NO_YIELD))


@pytest.fixture(scope="module")
def quoted(plain):
    return with_series(plain.futures_days["level"])


class TestAnAssumedMispricing:

    @pytest.fixture(scope="class")
    def mispriced(self):
        return run(futures(mispricing_bps=20.0, mispricing_seed=3, **NO_YIELD))

    def test_none_by_default(self,
                             plain):
        days = plain.futures_days

        assert (days["mispricing"] == 0.0).all()
        assert days["price"].to_numpy() == pytest.approx(days["fair_value"].to_numpy())

    def test_the_price_is_fair_value_moved_by_it(self,
                                                 mispriced):
        days = mispriced.futures_days

        assert days["price"].to_numpy() == pytest.approx(
            (days["fair_value"] * (1.0 + days["mispricing"])).to_numpy())

    def test_it_has_its_typical_size_and_reverts(self,
                                                 mispriced):
        moved = mispriced.futures_days["mispricing"]

        # About 18 independent draws in a year at a ten-session half-life,
        # so the measured spread is loose around 20 bp.
        assert 0.5 * 20e-4 < moved.std() < 1.6 * 20e-4
        assert moved.autocorr() > 0.8

    def test_the_same_seed_repeats_and_another_does_not(self,
                                                        mispriced):
        again = run(futures(mispricing_bps=20.0, mispricing_seed=3, **NO_YIELD))
        other = run(futures(mispricing_bps=20.0, mispricing_seed=4, **NO_YIELD))

        pd.testing.assert_series_equal(again.futures_days["mispricing"],
                                       mispriced.futures_days["mispricing"])
        assert not np.allclose(other.futures_days["mispricing"],
                               mispriced.futures_days["mispricing"])

    @pytest.mark.parametrize(("terms", "message"), [
        ({"mispricing_bps": -1.0}, "mispricing_bps"),
        ({"mispricing_half_life": 0.0}, "mispricing_half_life"),
        ({"mispricing_seed": 1.5}, "mispricing_seed"),
        ({"underlying": 3}, "underlying"),
    ])
    def test_settings_out_of_range_are_refused(self,
                                               terms,
                                               message):
        with pytest.raises(ValueError, match=message):
            IndexFutures(**terms)


class TestPricesInTheData:

    def test_the_data_s_price_is_used_where_it_has_one(self,
                                                       quoted):
        days = run_on(quoted, futures(prices="FUT", **NO_YIELD)).futures_days

        assert days["price"].iloc[10:].to_numpy() == pytest.approx(
            (days["level"].iloc[10:] * 1.002).to_numpy())
        assert days["price"].iloc[:10].to_numpy() == pytest.approx(
            days["fair_value"].iloc[:10].to_numpy())


class TestTheirOwnUnderlying:

    def test_futures_on_a_market_series(self,
                                        quoted):
        result = run_on(quoted, futures(underlying="FLAT", **NO_YIELD))
        days = result.futures_days

        assert (days["level"] == 1000.0).all()
        assert not (days["index_level"] == 1000.0).all()
        # The futures carry FLAT's return, nothing, while the index moves.
        assert result.nav_per_unit.iloc[-1] == pytest.approx(result.nav_per_unit.iloc[0])

    def test_futures_on_an_index_built_in_beacon(self,
                                                 plain):
        own = run(futures(**NO_YIELD))
        levels = own.index.target.levels * 0.5
        result = run(futures(underlying=levels, **NO_YIELD))

        assert result.futures_days["level"].to_numpy() == pytest.approx(
            (result.futures_days["index_level"] * 0.5).to_numpy())
        assert result.nav_per_unit.to_numpy() == pytest.approx(
            plain.nav_per_unit.to_numpy(), rel=1e-6)

    def test_an_underlying_with_no_levels_is_refused(self):
        with pytest.raises(CalculationError, match="NOWHERE"):
            run(futures(underlying="NOWHERE", **NO_YIELD))

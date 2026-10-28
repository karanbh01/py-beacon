# tests/test_currency_factor_attribution.py
"""BN-395: currency and factor attribution, each reconciling to the return.

Currency: test_book_currency's run, a dollar book holding one name quoted in
pounds at a flat 2.00 while the pound rises from 1.25 to 1.30 dollars. Its
local return is nothing, so all the fund earns is the currency's.

Factors: test_factor_analytics' thirty names, whose returns are the market's
plus two factors' with no noise, so a book's factor parts are its whole
return, and with noise the specific part takes up the rest exactly.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.attribution import Attribution
from beacon.analysis.attribution import model as factor_model
from beacon.analysis.attribution.currency import CROSS, CURRENCY, LOCAL
from beacon.analysis.data import RunData
from beacon.analysis.panels import NAME_RETURNS
from beacon.analysis.risk.factors import FactorAnalytics
from test_factor_analytics import DAYS, FUND, NAME_RETURNS_TABLE, SOURCE, factors, run_data


class TestCurrency:

    @pytest.fixture(scope="class")
    def found(self):
        from test_book_currency import run

        return Attribution(RunData.from_result(run(currency="USD"))).currency(by_currency=True)

    def test_a_flat_name_earns_only_its_currency(self,
                                                 found):
        assert found.effects[LOCAL] == pytest.approx(0.0, abs=1e-12)
        assert found.effects[CROSS] == pytest.approx(0.0, abs=1e-12)
        assert found.effects[CURRENCY] == pytest.approx(found.total_return, rel=0.01)

    def test_it_reconciles_and_is_by_currency(self,
                                              found):
        assert found.explained + found.residual == pytest.approx(found.total_return)
        assert list(found.table.index) == ["GBP"]
        assert found.table.loc["GBP", CURRENCY] == pytest.approx(found.effects[CURRENCY])

    def test_the_parts_multiply_back_to_the_book_return(self):
        from beacon.analysis.attribution.currency import currency_parts

        lagged = pd.DataFrame({"X": [1.0, 1.0]}, index=DAYS[:2])
        book = pd.DataFrame({"X": [0.0, (1.02 * 1.03) - 1.0]}, index=DAYS[:2])
        local = pd.DataFrame({"X": [0.0, 0.02]}, index=DAYS[:2])

        parts = currency_parts(lagged, book, local)

        assert parts[LOCAL].iloc[1, 0] == pytest.approx(0.02)
        assert parts[CURRENCY].iloc[1, 0] == pytest.approx(0.03)
        assert parts[CROSS].iloc[1, 0] == pytest.approx(0.02 * 0.03)


@pytest.fixture(scope="module")
def analytics():
    return FactorAnalytics(run_data(), SOURCE, factors=factors())


class TestFactorsByHoldings:

    def test_with_no_noise_the_factors_are_the_whole_return(self,
                                                            analytics):
        found = Attribution(analytics.run).factors(analytics)

        assert found.effects[factor_model.SPECIFIC] == pytest.approx(0.0, abs=1e-12)
        assert found.reconciles(1e-9)
        assert found.effects["market"] > 0.0

    def test_each_factor_is_its_exposure_times_its_return(self,
                                                          analytics):
        lagged = pd.DataFrame([FUND] * 3, index=DAYS[60:63])
        parts = factor_model.holdings_parts(analytics, lagged, analytics.run.tables[NAME_RETURNS])
        exposure = analytics.exposures(DAYS[60]).T @ FUND
        returns = analytics.factor_returns().loc[DAYS[61]]

        assert parts.loc[DAYS[61], "first"] == pytest.approx(exposure["first"] * returns["first"])

    def test_against_the_index_the_market_drops_out(self,
                                                    analytics):
        found = Attribution(analytics.run).factors(analytics, against="index")

        assert found.effects["market"] == pytest.approx(0.0, abs=1e-12)
        assert found.reconciles(1e-9)

    def test_with_noise_the_specific_part_takes_the_rest(self):
        data = run_data()
        tables = dict(data.tables)
        noise = np.random.default_rng(8).normal(0.0, 0.003, NAME_RETURNS_TABLE.shape)
        tables[NAME_RETURNS] = NAME_RETURNS_TABLE + noise
        fund = (tables[NAME_RETURNS] @ FUND).where(lambda found: found.index != DAYS[0], 0.0)
        noisy = RunData(currency="USD", returns=fund, nav=data.nav, units=data.units,
                        initial_capital=1e6, periods_per_year=252, risk_free_rate=0.0,
                        tables=tables, comparators=data.comparators)
        found = Attribution(noisy).factors(FactorAnalytics(noisy, SOURCE, factors=factors()))

        assert abs(found.effects[factor_model.SPECIFIC]) > 1e-4
        assert found.reconciles(1e-9)


class TestFactorsByReturns:

    def test_with_no_noise_the_betas_explain_it_all(self,
                                                    analytics):
        found = Attribution(analytics.run).factors(analytics, method="returns")

        assert found.effects[factor_model.RESIDUAL] == pytest.approx(0.0, abs=1e-9)
        assert found.reconciles(1e-9)

    def test_an_unknown_method(self,
                               analytics):
        with pytest.raises(ValueError, match="pca"):
            Attribution(analytics.run).factors(analytics, method="pca")

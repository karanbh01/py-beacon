# tests/test_currency_conversion.py
"""BN-253: every read that compares or adds money across names converts it.

Two names, flat in their own currencies: AAA at 10 USD and VOD at 2 GBP, with
GBPUSD rising from 1.25 to 1.30. In dollars, VOD's only return is the exchange
rate's, so every surface can be checked against it exactly.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.asset.equity import Equity
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.expressions import data
from beacon.index.context import IndexContext
from beacon.index.expression_rules import ExpressionRule
from beacon.index.methodology import LiquidityRule
from beacon.server import ServerConfig, create_app
from beacon.server.optimisation import constituent_prices
from beacon.server.risk import constituent_returns
from beacon.server.weights import prices_for

DAYS = pd.bdate_range("2024-01-02", periods=40)
RATES = np.linspace(1.25, 1.30, len(DAYS))
TOKEN = "fx-token"


def build_fetcher(with_rate: bool = True) -> DataFetcher:
    rows = []

    for day, rate in zip(DAYS, RATES, strict=True):
        rows.append({"IDENTIFIER": "AAA", "DATE": day, "CLOSE": 10.0,
                     "VOLUME": 1_000_000.0, "SHARES_OUTSTANDING": 1e8})
        rows.append({"IDENTIFIER": "VOD", "DATE": day, "CLOSE": 2.0,
                     "VOLUME": 1_000_000.0, "SHARES_OUTSTANDING": 1e9})
        if with_rate:
            rows.append({"IDENTIFIER": "GBPUSD", "DATE": day, "CLOSE": rate,
                         "RATE": rate})

    reference = pd.DataFrame([
        {"IDENTIFIER": "AAA", "DATE_FROM": "2020-01-01", "NAME": "A",
         "CURRENCY": "USD", "EXCHANGE": "XNYS"},
        {"IDENTIFIER": "VOD", "DATE_FROM": "2020-01-01", "NAME": "V",
         "CURRENCY": "GBP", "EXCHANGE": "XLON"}])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


@pytest.fixture
def fetcher() -> DataFetcher:
    return build_fetcher()


def vod() -> Equity:
    return Equity(asset_id="VOD", ticker="VOD", name="V", currency="GBP",
                  exchange="XLON")


class TestFetchPrices:

    def test_each_name_is_converted_day_by_day(self,
                                               fetcher):
        prices = fetcher.fetch_prices(["AAA", "VOD"], currency="USD")

        assert list(prices.columns) == ["AAA", "VOD"]
        assert (prices["AAA"] == 10.0).all()
        np.testing.assert_allclose(prices["VOD"].to_numpy(), 2.0 * RATES)

    def test_no_currency_leaves_each_in_its_own(self,
                                                fetcher):
        prices = fetcher.fetch_prices(["AAA", "VOD"])

        assert (prices["VOD"] == 2.0).all()

    def test_a_pair_with_no_rate_is_refused_naming_the_names(self):
        with pytest.raises(CalculationError, match="VOD cannot be priced in USD"):
            build_fetcher(with_rate=False).fetch_prices(["AAA", "VOD"],
                                                        currency="USD")

    def test_the_inverse_pair_converts_the_other_way(self,
                                                     fetcher):
        prices = fetcher.fetch_prices(["AAA"], currency="GBP")

        np.testing.assert_allclose(prices["AAA"].to_numpy(), 10.0 / RATES)


class TestLiquidityRule:
    """VOD trades 2 million GBP a day: 2.5 to 2.6 million in dollars."""

    RULE = LiquidityRule(min_avg_daily_value=2_200_000, lookback_days=20)

    def test_traded_value_is_compared_in_the_index_currency(self,
                                                            fetcher):
        assert self.RULE.is_eligible(vod(), DAYS[-1], fetcher,
                                     IndexContext(currency="USD"))
        assert not self.RULE.is_eligible(vod(), DAYS[-1], fetcher,
                                         IndexContext(currency="GBP"))

    def test_a_missing_rate_is_refused(self):
        with pytest.raises(CalculationError, match="GBP/USD"):
            self.RULE.is_eligible(vod(), DAYS[-1], build_fetcher(with_rate=False),
                                  IndexContext(currency="USD"))


class TestExpressions:
    """VOD's cap is 2 billion GBP: 2.6 billion in dollars on the last day."""

    RULE = ExpressionRule.from_expression(data.market.market_cap > 2.2e9)

    def test_market_cap_is_in_the_index_currency(self,
                                                 fetcher):
        assert self.RULE.is_eligible(vod(), DAYS[-1], fetcher,
                                     IndexContext(currency="USD"))
        assert not self.RULE.is_eligible(vod(), DAYS[-1], fetcher,
                                         IndexContext(currency="GBP"))

    def test_outside_an_index_it_is_in_usd(self,
                                           fetcher):
        assert self.RULE.is_eligible(vod(), DAYS[-1], fetcher)


class TestTheEngineViews:

    def test_risk_returns_carry_the_exchange_rate(self,
                                                  fetcher):
        returns = constituent_returns(fetcher, ["AAA", "VOD"], None, None, "USD")
        expected = pd.Series(RATES, index=DAYS).pct_change().dropna()

        np.testing.assert_allclose(returns["VOD"].to_numpy(),
                                   expected.to_numpy())
        assert (returns["AAA"] == 0.0).all()

    def test_optimisation_prices_are_in_the_index_currency(self,
                                                           fetcher):
        prices = constituent_prices(fetcher, ["AAA", "VOD"], currency="USD")

        np.testing.assert_allclose(prices["VOD"].to_numpy(), 2.0 * RATES)

    def test_the_weights_pane_prices_in_the_index_currency(self,
                                                           fetcher):
        prices = prices_for(fetcher, ["AAA", "VOD"], None, None, "USD")

        np.testing.assert_allclose(prices["VOD"].to_numpy(), 2.0 * RATES)

    def test_a_risk_model_says_its_currency(self,
                                            fetcher):
        app = create_app(ServerConfig(auth_token=TOKEN, data_fetcher=fetcher))
        headers = {"Authorization": f"Bearer {TOKEN}"}

        with TestClient(app, raise_server_exceptions=False) as client:
            client.post("/risk-models/fx/estimate", headers=headers,
                        json={"identifiers": ["AAA", "VOD"]})
            client.portal.call(client.app.state.jobs.drain)
            default = client.get("/risk-models/fx", headers=headers).json()

            client.post("/risk-models/gbp/estimate", headers=headers,
                        json={"identifiers": ["AAA", "VOD"], "currency": "gbp"})
            client.portal.call(client.app.state.jobs.drain)
            pounds = client.get("/risk-models/gbp", headers=headers).json()

        assert default["currency"] == "USD"
        assert pounds["currency"] == "GBP"
        assert default["volatilities"]["VOD"] > 0.0
        assert pounds["volatilities"]["AAA"] > 0.0

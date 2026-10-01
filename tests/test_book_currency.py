# tests/test_book_currency.py
"""BN-228: a backtest keeps its books in the index's currency by default.

One name, VOD, flat at 2 GBP, in a GBP index, with GBPUSD rising from 1.25 to
1.30. A book in pounds sees a flat NAV. A book in dollars, which used to be
the default whatever the index, sees the exchange rate's rise.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.backtest.engine import BacktestEngine
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.cache import IndexResultCache
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.result import IndexResult
from beacon.server import ServerConfig, create_app

DAYS = pd.bdate_range("2024-01-02", periods=40)
RATES = np.linspace(1.25, 1.30, len(DAYS))
CAPITAL = 1_000_000.0


def fetcher() -> DataFetcher:
    rows = []

    for day, rate in zip(DAYS, RATES, strict=True):
        rows.append({"IDENTIFIER": "VOD", "DATE": day, "CLOSE": 2.0,
                     "VOLUME": 1e6, "SHARES_OUTSTANDING": 1e9})
        rows.append({"IDENTIFIER": "GBPUSD", "DATE": day, "CLOSE": rate,
                     "RATE": rate})

    reference = pd.DataFrame([{"IDENTIFIER": "VOD", "DATE_FROM": "2020-01-01",
                               "NAME": "V", "CURRENCY": "GBP",
                               "EXCHANGE": "XLON"}])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def pound_index(data: DataFetcher) -> IndexResult:
    definition = IndexDefinition(index_id="gbp", index_name="Pounds",
                                 base_date=str(DAYS[0].date()),
                                 base_value=100.0, currency="GBP",
                                 eligibility_rules=[],
                                 weighting_scheme=EqualWeighted(),
                                 rebalancing_frequency="MONTHLY",
                                 calendar="XLON",
                                 universe_identifiers=["VOD"])

    return IndexCalculator(definition, data).run(end_date=str(DAYS[-1].date()))


def run(currency: str | None = None):
    data = fetcher()
    index = pound_index(data)

    return BacktestEngine(start_date=str(DAYS[0].date()),
                          end_date=str(DAYS[-1].date()),
                          initial_capital=CAPITAL, data_provider=data,
                          index_result=index, currency=currency).run()


class TestTheIndexRecordsItsCurrency:

    def test_a_calculated_index_carries_its_definitions_currency(self):
        assert pound_index(fetcher()).currency == "GBP"

    def test_the_currency_survives_the_cache(self,
                                             tmp_path):
        result = pound_index(fetcher())
        cache = IndexResultCache(tmp_path)
        cache.put("b" * 64, result)

        assert cache.get("b" * 64).currency == "GBP"


class TestTheBookCurrency:

    def test_the_book_defaults_to_the_index_currency(self):
        result = run()

        assert result.currency == "GBP"
        assert result.trading_nav.iloc[-1] == pytest.approx(CAPITAL, rel=1e-9)

    def test_another_currency_can_be_chosen(self):
        """A dollar investor tracking a pound index sees the rate move."""
        result = run("usd")
        nav = result.trading_nav

        assert result.currency == "USD"
        assert nav.iloc[-1] / nav.iloc[0] == pytest.approx(RATES[-1] / RATES[0],
                                                           rel=1e-6)

    def test_an_index_without_a_currency_keeps_a_usd_book(self):
        data = fetcher()
        index = pound_index(data)
        index.currency = None

        result = BacktestEngine(start_date=str(DAYS[0].date()),
                                end_date=str(DAYS[-1].date()),
                                initial_capital=CAPITAL, data_provider=data,
                                index_result=index).run()

        assert result.currency == "USD"


class TestTheEngineSaysItsCurrency:

    def test_a_backtest_run_reports_its_book_currency(self,
                                                      tmp_path):
        from beacon.testing import dataset

        token = "ccy-token"
        headers = {"Authorization": f"Bearer {token}"}
        document = {
            "id": "gbp", "name": "Pound index", "base_date": "2023-01-03",
            "base_value": 100.0, "currency": "GBP", "calendar": "XNYS",
            "rebalancing_frequency": "QUARTERLY",
            "universe": {"universe_id": None,
                         "identifiers": ["AAA", "BBB"]},
            "pipeline": {"selection": [],
                         "weighting": {"id": "w", "scheme": "EqualWeighted",
                                       "params": {}},
                         "treatment": {"corporate_actions": "ADJUST_DIVISOR"}}}
        app = create_app(ServerConfig(auth_token=token,
                                      data_fetcher=dataset.data_fetcher(),
                                      storage_root=tmp_path))

        with TestClient(app, raise_server_exceptions=False) as client:
            client.put("/indices/gbp", json=document, headers=headers)
            client.post("/beacon/gbp/backtest", headers=headers,
                        json={"start": "2023-01-03", "end": "2023-03-31"})
            client.portal.call(client.app.state.jobs.drain)
            run_result = client.app.state.jobs.latest_result("backtest:gbp")

        assert run_result is not None
        assert run_result["currency"] == "GBP"

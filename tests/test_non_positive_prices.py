# tests/test_non_positive_prices.py
"""BN-334: a price at or below zero is read as no price, and synthetic prices
never go below zero.

Found by beacon-ui: a plain index strategy on the default synthetic store
failed with "price cannot be negative". A fat-tailed return below -100% had
turned a name's path negative, and the engine traded it.
"""
import logging

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app
from beacon.synthetic.dataset import SyntheticConfig, generate
from test_server_backtest import BASE_PRICE, DATES, GROWTH, SHARES, TOKEN, auth, index_document

# The day BBB's closes turn negative, and stay so.
BROKEN_FROM = DATES[len(DATES) // 2]


def market(broken_from: pd.Timestamp | None = BROKEN_FROM) -> pd.DataFrame:
    span = len(DATES) - 1
    rows = [{"IDENTIFIER": name, "DATE": date,
             "CLOSE": BASE_PRICE[name] * (growth ** (position / span)),
             "VOLUME": 1_000_000, "SHARES_OUTSTANDING": SHARES}
            for name, growth in GROWTH.items() for position, date in enumerate(DATES)]
    frame = pd.DataFrame(rows)

    if broken_from is not None:
        frame.loc[(frame["IDENTIFIER"] == "BBB") & (frame["DATE"] >= broken_from),
                  "CLOSE"] *= -1

    return frame


class TestLoadingMarketData:

    def test_a_non_positive_price_is_read_as_missing(self):
        data = DataFetcher(MarketData.from_dataframe(market())).market.data
        broken = data.loc["BBB"].loc[BROKEN_FROM:]

        assert broken["CLOSE"].isna().all()
        assert (broken["VOLUME"] == 1_000_000).all()
        assert data.loc["AAA", "CLOSE"].notna().all()

    def test_one_warning_names_the_identifier_and_first_date(self,
                                                              caplog):
        with caplog.at_level(logging.WARNING, logger="beacon.data.base"):
            DataFetcher(MarketData.from_dataframe(market()))

        warnings = [record.getMessage() for record in caplog.records]

        assert len(warnings) == 1
        assert f"BBB from {BROKEN_FROM:%Y-%m-%d}" in warnings[0]

    def test_clean_data_is_left_alone_and_unremarked(self,
                                                     caplog):
        with caplog.at_level(logging.WARNING, logger="beacon.data.base"):
            data = DataFetcher(MarketData.from_dataframe(market(None))).market.data

        assert data["CLOSE"].notna().all()
        assert caplog.records == []


# Three bad closes inside a quarter, clear of any rebalance.
BLIP = pd.bdate_range("2023-11-14", "2023-11-16")


def backtest(tmp_path,
             frame: pd.DataFrame,
             selection: list[dict] | None = None,
             body: dict | None = None) -> dict:
    """Run the BT strategy over *frame* through the API, and return its job."""
    reference = pd.DataFrame([{"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
                               "CURRENCY": "USD", "EXCHANGE": "NYSE"} for name in GROWTH])
    fetcher = DataFetcher(MarketData.from_dataframe(frame), ReferenceData.from_dataframe(reference))
    document = index_document()
    document["pipeline"]["selection"] = selection or []
    config = ServerConfig(auth_token=TOKEN, data_fetcher=fetcher, storage_root=tmp_path)

    with TestClient(create_app(config), raise_server_exceptions=False) as client:
        assert client.post("/indices", json=document, headers=auth()).status_code == 200
        assert client.put("/strategies/BT", json={"id": "BT", "name": "BT", "index_id": "BT"},
                          headers=auth()).status_code == 200
        submitted = client.post("/strategies/BT/backtest", json=body or {}, headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        job: dict = client.get(f"/jobs/{submitted.json()['job_id']}", headers=auth()).json()
        return job


class TestABacktest:

    def test_carries_a_name_through_a_few_bad_closes(self,
                                                     tmp_path):
        frame = market(None)
        frame.loc[(frame["IDENTIFIER"] == "BBB") & frame["DATE"].isin(BLIP), "CLOSE"] *= -1

        job = backtest(tmp_path, frame)
        gaps = [gap["date"] for gap in job["result"]["price_gaps"] if gap["asset_id"] == "BBB"]

        assert job["status"] == "succeeded", job["error"]
        assert gaps == [f"{day:%Y-%m-%d}" for day in BLIP]

    def test_runs_past_a_name_whose_prices_went_negative_for_good(self,
                                                                  tmp_path):
        """An old synthetic store's case: the path stays negative, and a rule
        that values the name drops it once it has no price."""
        job = backtest(tmp_path, market(),
                       [{"id": "cap", "type": "MarketCapRule", "params": {"min_market_cap": 1.0}}])

        assert job["status"] == "succeeded", job["error"]


class TestSyntheticPrices:

    def test_stay_above_zero_through_the_fattest_tail(self):
        """600 names over ten years drew one return of -104% before log
        returns, and 2,194 of its prices were negative."""
        dataset = generate(SyntheticConfig(assets=600, start="2016-10-21", end="2026-10-21"))
        returns = dataset.returns.to_numpy()
        prices = dataset.market.data[["OPEN", "HIGH", "LOW", "CLOSE"]].to_numpy()

        assert np.nanmin(returns) > -1.0
        assert np.nanmin(prices) > 0.0

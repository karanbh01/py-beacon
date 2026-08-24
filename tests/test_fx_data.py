# tests/test_data_gaps.py
"""BN-144/145: currency pairs, and the coverage row for them.

From beacon-ui BU-100 and BU-101. The report said the pairs were not
addressable; measured at HEAD they always were, and what was actually broken
was that their value sat in a column no price view reads.
"""
import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.synthetic import SyntheticConfig, generate

TOKEN = "gaps-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def panel():
    logging.disable(logging.ERROR)

    try:
        return generate(SyntheticConfig(assets=12, start="2024-01-02",
                                        end="2024-12-31", seed=1))
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def fetcher(panel):
    return panel.fetcher()


@pytest.fixture
def client(fetcher):
    return TestClient(create_app(ServerConfig(
        auth_token=TOKEN, data_fetcher=fetcher,
        storage_root=Path(tempfile.mkdtemp()))))


def coverage_of(client) -> dict:
    body = client.get("/data/coverage", headers=HEADERS).json()

    return {entry["dataset"]: entry for entry in body["datasets"]}


class TestIdentifyingAPair:
    """What makes a market identifier a currency pair."""

    def test_the_pairs_are_found(self,
                                 fetcher):
        assert fetcher.fx_pairs == ["AUDUSD", "CADUSD", "EURUSD", "GBPUSD",
                                    "HKDUSD", "JPYUSD"]

    def test_an_instrument_is_not_a_pair(self,
                                         fetcher):
        assert "CMPA" not in fetcher.fx_pairs

    def test_instruments_exclude_the_pairs(self,
                                           fetcher):
        """What a universe or a search should offer: a rate series is not
        something anybody holds."""
        assert set(fetcher.instrument_identifiers) & set(fetcher.fx_pairs) == set()
        assert set(fetcher.instrument_identifiers) | set(fetcher.fx_pairs) == set(
            fetcher.identifiers)

    def test_it_is_the_rate_column_that_decides(self,
                                                panel):
        """Not a six-letter name pattern.

        An instrument legitimately called `EURUSD` would be misfiled by a name
        rule, and a store may hold pairs for currencies its reference data
        never mentions. `RATE` is populated on a pair and null on everything
        else, which is a fact about the data rather than about the spelling.
        """
        market = panel.market.data

        assert market.loc["CMPA"]["RATE"].isna().all()
        assert market.loc["EURUSD"]["RATE"].notna().all()


class TestFxCoverage:
    """BU-100: `fx` on the coverage report."""

    def test_it_is_reported_as_a_dataset(self,
                                         client):
        assert coverage_of(client)["fx"]["configured"] is True

    def test_it_counts_the_pairs(self,
                                 client):
        assert coverage_of(client)["fx"]["identifiers"] == 6

    def test_it_carries_the_freshness_fields(self,
                                             client):
        """A client derives its staleness indicator from these, so a row
        without them renders as permanently unknown."""
        entry = coverage_of(client)["fx"]

        assert entry["frequency"] == "daily"
        assert entry["stale_after_seconds"]

    def test_it_does_not_inflate_the_union(self,
                                           client):
        """The pairs are market rows and were already counted. A union that
        grew when the row was added would report more assets than exist."""
        body = client.get("/data/coverage", headers=HEADERS).json()
        market = coverage_of(client)["market"]["identifiers"]

        assert body["identifiers_union"] == market

    def test_it_claims_no_file_of_its_own(self,
                                          client):
        """The rows live in the market file, which `market` already reports.
        Repeating that size here counts it twice and makes the sum of the
        parts exceed the store total — which it did, until a test said so.
        """
        assert coverage_of(client)["fx"]["cache_size_bytes"] is None

    def test_a_store_without_pairs_says_so(self):
        from beacon.testing import dataset

        client = TestClient(create_app(ServerConfig(
            auth_token=TOKEN, data_fetcher=dataset.data_fetcher(),
            storage_root=Path(tempfile.mkdtemp()))))
        entry = coverage_of(client)["fx"]

        assert entry["configured"] is False
        assert entry["identifiers"] == 0


class TestAPairChartsLikeAnythingElse:
    """BU-101. The pairs were always addressable; the series was empty."""

    def test_the_pair_is_addressable(self,
                                     client):
        assert client.get("/data/prices/EURUSD",
                          headers=HEADERS).status_code == 200

    def test_close_is_populated(self,
                                client):
        """The actual defect. A client charting `CLOSE` — which is every price
        view, because that is where every other identifier puts its value —
        got 261 rows of nulls, which reads as "addressable but empty"."""
        prices = client.get("/data/prices/EURUSD", headers=HEADERS).json()
        frame = prices["prices"]
        column = frame["columns"].index("CLOSE")
        close = [row[column] for row in frame["data"]]

        assert len(close) > 200
        assert all(value is not None for value in close)

    def test_close_equals_the_rate(self,
                                   panel):
        """Duplicated rather than moved, so the two must never diverge: `RATE`
        is what every conversion path reads and `CLOSE` is what every chart
        reads."""
        pair = panel.market.data.loc["EURUSD"]

        assert np.allclose(pair["CLOSE"], pair["RATE"])

    def test_the_conversion_paths_still_read_the_same_numbers(self,
                                                              fetcher,
                                                              panel):
        rates = fetcher.fetch_fx_rates("EUR", "USD")
        stored = panel.market.data.loc["EURUSD"]["RATE"]

        assert np.allclose(rates.to_numpy(), stored.to_numpy())

    def test_the_pair_is_enumerated(self,
                                    client):
        """Already true before this change, and worth pinning: the subject
        field is how anything gets loaded."""
        body = client.get("/data/identifiers", headers=HEADERS,
                          params={"limit": 1000}).json()
        names = {entry["identifier"] for entry in body["identifiers"]}

        assert "EURUSD" in names

    def test_an_equity_carries_no_rate(self,
                                       client):
        """The report said market bars carry the instrument's own rate. They
        do not — `RATE` is null on every equity, which is what makes it a
        usable discriminator."""
        prices = client.get("/data/prices/CMPA", headers=HEADERS).json()
        frame = prices["prices"]
        rates = [row[frame["columns"].index("RATE")] for row in frame["data"]]

        assert all(value is None for value in rates)

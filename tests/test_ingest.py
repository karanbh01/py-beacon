# tests/test_ingest.py
"""BN-100: data ingestion and the sync job."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.data.ingest import (
    IngestResult,
    ingest_market_data,
    ingest_reference_data,
    normalise_history,
    normalise_reference,
    yfinance_downloader,
    yfinance_reference_downloader,
)
from beacon.exceptions import DataNotFoundError
from beacon.server import ServerConfig, create_app

TOKEN = "test-token-value"
PERIODS = 4

DELISTED = "BAD"
QUIET = "EMPTY"


def history(periods: int = PERIODS,
            timezone: str | None = "UTC") -> pd.DataFrame:
    """A downloaded OHLCV frame in the source's shape."""
    index = pd.date_range("2024-01-01", periods=periods, tz=timezone)

    return pd.DataFrame({"Open": [1.0] * periods,
                         "High": [2.0] * periods,
                         "Low": [0.5] * periods,
                         "Close": [1.5 + step * 0.1 for step in range(periods)],
                         "Volume": [10] * periods,
                         "Dividends": [0.0] * periods},
                        index=index)


def fake_downloader(identifier: str,
                    start: str | None = None,
                    end: str | None = None) -> pd.DataFrame:
    """Stands in for the network. Two identifiers misbehave on purpose."""
    if identifier == DELISTED:
        raise RuntimeError("delisted")

    if identifier == QUIET:
        return pd.DataFrame()

    return history()


def fake_reference(identifier: str) -> dict[str, object]:
    """Stands in for the source's info mapping."""
    if identifier == DELISTED:
        raise RuntimeError("no profile")

    return {"longName": f"{identifier} Corp", "currency": "USD",
            "exchange": "NYSE", "sector": "Technology",
            "marketCap": 1_000_000_000}


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def fetcher():
    market = MarketData.from_dataframe(pd.DataFrame({
        "IDENTIFIER": ["ZZZ"] * 2,
        "DATE": pd.bdate_range("2024-01-01", periods=2),
        "CLOSE": [99.0, 99.5]}))

    return DataFetcher(market)


@pytest.fixture
def client(fetcher):
    """Entered as a context manager so the test portal exists.

    Without it TestClient has no running event loop to drive the job registry
    from, and `client.portal` is None.
    """
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=fetcher,
                          market_downloader=fake_downloader)

    with TestClient(create_app(config), raise_server_exceptions=False) as started:
        yield started


class TestNormaliseHistory:

    def test_it_produces_the_long_form(self):
        frame = normalise_history("AAA", history())

        assert list(frame.columns) == ["IDENTIFIER", "DATE", "OPEN", "HIGH",
                                       "LOW", "CLOSE", "VOLUME"]
        assert (frame["IDENTIFIER"] == "AAA").all()

    def test_unmapped_columns_are_dropped(self):
        """A column carried through under a foreign name is an invitation to
        read the wrong one."""
        assert "Dividends" not in normalise_history("AAA", history()).columns

    def test_timezones_are_stripped(self):
        """MarketData compares against naive timestamps, and a mixed-awareness
        comparison raises rather than quietly misaligning."""
        frame = normalise_history("AAA", history(timezone="America/New_York"))

        assert frame["DATE"].dt.tz is None

    def test_naive_input_is_accepted_too(self):
        frame = normalise_history("AAA", history(timezone=None))

        assert len(frame) == PERIODS

    def test_dates_are_normalised_to_midnight(self):
        frame = normalise_history("AAA", history())

        assert (frame["DATE"] == frame["DATE"].dt.normalize()).all()

    def test_an_empty_frame_gives_an_empty_result(self):
        result = normalise_history("AAA", pd.DataFrame())

        assert result.empty
        assert "CLOSE" in result.columns

    def test_a_frame_without_a_close_is_refused(self):
        """The calculator and the engine both read CLOSE, so accepting one
        without it would fail much later and somewhere unrelated."""
        frame = pd.DataFrame({"Open": [1.0]}, index=pd.date_range("2024-01-01",
                                                                  periods=1))

        with pytest.raises(DataNotFoundError, match="close price"):
            normalise_history("AAA", frame)

    def test_a_partial_frame_is_accepted(self):
        """No volume is still a usable price series."""
        frame = pd.DataFrame({"Close": [1.0, 2.0]},
                             index=pd.date_range("2024-01-01", periods=2))

        assert len(normalise_history("AAA", frame)) == 2


class TestIngestMarketData:

    def test_it_fetches_what_it_can(self):
        result = ingest_market_data(["AAA", "BBB"], fake_downloader)

        assert result.fetched == ["AAA", "BBB"]
        assert result.rows == 2 * PERIODS
        assert result.succeeded

    def test_one_failure_does_not_stop_the_run(self):
        """A sync over five hundred names must not collapse because one was
        delisted last month."""
        result = ingest_market_data(["AAA", DELISTED, "BBB"], fake_downloader)

        assert result.fetched == ["AAA", "BBB"]
        assert "delisted" in result.failed[DELISTED]

    def test_an_empty_response_counts_as_a_failure(self):
        """Nothing came back, so the caller should know rather than assume."""
        result = ingest_market_data([QUIET], fake_downloader)

        assert result.failed[QUIET] == "no data returned"
        assert not result.succeeded

    def test_progress_is_reported_per_identifier(self):
        """'142 of 500' is a real statement; a bulk request can only say 0
        then 100."""
        seen = []

        ingest_market_data(["AAA", "BBB", "CCC"], fake_downloader,
                           on_progress=lambda done, total, name: seen.append(
                               (done, total, name)))

        assert seen == [(1, 3, "AAA"), (2, 3, "BBB"), (3, 3, "CCC")]

    def test_progress_counts_failures_too(self):
        """Otherwise a run with failures never reaches 100%."""
        seen = []

        ingest_market_data(["AAA", DELISTED], fake_downloader,
                           on_progress=lambda done, total, name: seen.append(done))

        assert seen == [1, 2]

    def test_the_summary_reports_both_sides(self):
        summary = ingest_market_data(["AAA", DELISTED], fake_downloader).summary()

        assert summary["fetched"] == 1
        assert summary["failed"] == 1
        assert DELISTED in summary["errors"]

    def test_nothing_requested_is_an_empty_result(self):
        result = ingest_market_data([], fake_downloader)

        assert result.rows == 0
        assert not result.succeeded


class TestIngestReferenceData:

    def test_it_maps_the_fields_it_knows(self):
        result = ingest_reference_data(["AAA"], fake_reference)
        row = result.reference.iloc[0]

        assert row["NAME"] == "AAA Corp"
        assert row["SECTOR"] == "Technology"

    def test_unmapped_fields_are_left_out(self):
        result = ingest_reference_data(["AAA"], fake_reference)

        assert "marketCap" not in result.reference.columns

    def test_records_carry_a_validity_start(self):
        """Reference data needs a DATE_FROM, and a download carries no history
        of when a field changed."""
        result = ingest_reference_data(["AAA"], fake_reference)

        assert "DATE_FROM" in result.reference.columns

    def test_a_missing_field_is_absent_rather_than_filled(self):
        row = normalise_reference("AAA", {"longName": "Alpha"})

        assert row["NAME"] == "Alpha"
        assert "SECTOR" not in row

    def test_failures_are_isolated(self):
        result = ingest_reference_data(["AAA", DELISTED], fake_reference)

        assert result.fetched == ["AAA"]
        assert DELISTED in result.failed


class TestMerge:

    def test_new_identifiers_become_queryable(self,
                                              fetcher):
        """The acceptance criterion: synced data is readable through the
        fetcher the server is already serving from."""
        result = ingest_market_data(["AAA"], fake_downloader)
        fetcher.merge_market_data(result.market)

        assert "AAA" in fetcher.identifiers
        assert len(fetcher.fetch_market_data("AAA")) == PERIODS

    def test_existing_data_survives(self,
                                    fetcher):
        fetcher.merge_market_data(ingest_market_data(["AAA"], fake_downloader).market)

        assert "ZZZ" in fetcher.identifiers

    def test_a_resync_restates_rather_than_duplicating(self,
                                                       fetcher):
        """A re-sync of a window is a correction — a restated close, a
        backfilled volume — so the newer value wins and nothing is added."""
        result = ingest_market_data(["AAA"], fake_downloader)

        first = fetcher.merge_market_data(result.market)
        second = fetcher.merge_market_data(result.market)

        assert first == PERIODS
        assert second == 0
        assert len(fetcher.fetch_market_data("AAA")) == PERIODS

    def test_a_restated_price_replaces_the_old_one(self,
                                                   fetcher):
        original = ingest_market_data(["AAA"], fake_downloader).market
        fetcher.merge_market_data(original)

        restated = original.copy()
        restated["CLOSE"] = 42.0
        fetcher.merge_market_data(restated)

        assert (fetcher.fetch_market_data("AAA")["CLOSE"] == 42.0).all()

    def test_merging_nothing_changes_nothing(self,
                                             fetcher):
        assert fetcher.merge_market_data(pd.DataFrame()) == 0

    def test_reference_data_merges_into_an_empty_source(self,
                                                        fetcher):
        result = ingest_reference_data(["AAA"], fake_reference)

        assert fetcher.merge_reference_data(result.reference) == 1
        assert fetcher.fetch_classification("AAA") == "Technology"

    def test_reference_data_merges_into_an_existing_source(self):
        market = MarketData.from_dataframe(pd.DataFrame({
            "IDENTIFIER": ["ZZZ"], "DATE": ["2024-01-01"], "CLOSE": [1.0]}))
        reference = ReferenceData.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "ZZZ", "NAME": "Zed", "DATE_FROM": "2020-01-01"}]))
        source = DataFetcher(market, reference)

        added = source.merge_reference_data(
            ingest_reference_data(["AAA"], fake_reference).reference)

        assert added == 1
        assert source.fetch_classification("AAA") == "Technology"


class TestSyncEndpoint:

    def test_it_returns_a_job(self,
                              client):
        response = client.post("/data/coverage/market/sync",
                               json={"identifiers": ["AAA", "BBB"]},
                               headers=auth())

        assert response.status_code == 202
        assert response.json()["kind"] == "sync:market"

    def test_the_job_lands_the_data(self,
                                    client,
                                    fetcher):
        """The acceptance criterion, end to end."""
        job_id = client.post("/data/coverage/market/sync",
                             json={"identifiers": ["AAA", "BBB"]},
                             headers=auth()).json()["job_id"]
        client.portal.call(client.app.state.jobs.drain)

        result = client.get(f"/jobs/{job_id}", headers=auth()).json()

        assert result["status"] == "succeeded"
        assert result["result"]["fetched"] == 2
        assert "AAA" in fetcher.identifiers

    def test_the_job_reports_failures_without_failing(self,
                                                      client):
        """A partial sync succeeded at something and should say so."""
        job_id = client.post("/data/coverage/market/sync",
                             json={"identifiers": ["AAA", DELISTED]},
                             headers=auth()).json()["job_id"]
        client.portal.call(client.app.state.jobs.drain)

        result = client.get(f"/jobs/{job_id}", headers=auth()).json()

        assert result["status"] == "succeeded"
        assert result["result"]["failed"] == 1
        assert DELISTED in result["result"]["errors"]

    def test_an_empty_body_resyncs_what_is_loaded(self,
                                                  client):
        """The common case: refresh what I have."""
        job_id = client.post("/data/coverage/market/sync",
                             headers=auth()).json()["job_id"]
        client.portal.call(client.app.state.jobs.drain)

        result = client.get(f"/jobs/{job_id}", headers=auth()).json()

        assert result["result"]["identifiers"] == ["ZZZ"]

    def test_a_freshness_event_is_published(self,
                                            client):
        """Announced only once the data is queryable, so a client refetching
        on the event cannot beat the merge to it."""
        registry = client.app.state.jobs
        queue = client.portal.call(_subscribe, registry)

        client.post("/data/coverage/market/sync",
                    json={"identifiers": ["AAA"]}, headers=auth())
        client.portal.call(registry.drain)

        events = client.portal.call(_drain_queue, queue)
        freshness = [event for event in events if event["type"] == "data.freshness"]

        assert freshness
        assert freshness[0]["dataset"] == "market"
        assert freshness[0]["detail"]["identifiers"] == 1

    def test_an_unknown_dataset_is_a_404(self,
                                         client):
        response = client.post("/data/coverage/fundamentals/sync", headers=auth())

        assert response.status_code == 404

    def test_it_requires_authentication(self,
                                        client):
        assert client.post("/data/coverage/market/sync").status_code == 401

    def test_a_server_without_data_reports_500(self):
        config = ServerConfig(auth_token=TOKEN, market_downloader=fake_downloader)
        bare = TestClient(create_app(config), raise_server_exceptions=False)

        assert bare.post("/data/coverage/market/sync",
                         headers=auth()).status_code == 500

    def test_nothing_to_sync_is_a_404(self):
        empty = MarketData.from_dataframe(
            pd.DataFrame(columns=["IDENTIFIER", "DATE", "CLOSE"]))
        config = ServerConfig(auth_token=TOKEN,
                              data_fetcher=DataFetcher(empty),
                              market_downloader=fake_downloader)
        client = TestClient(create_app(config), raise_server_exceptions=False)

        assert client.post("/data/coverage/market/sync",
                           headers=auth()).status_code == 404

    def test_coverage_reflects_the_sync(self,
                                        client):
        before = client.get("/data/coverage", headers=auth()).json()
        market_before = next(d for d in before["datasets"] if d["dataset"] == "market")

        client.post("/data/coverage/market/sync",
                    json={"identifiers": ["AAA", "BBB"]}, headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        after = client.get("/data/coverage", headers=auth()).json()
        market_after = next(d for d in after["datasets"] if d["dataset"] == "market")

        assert market_after["identifiers"] > market_before["identifiers"]


class TestMissingExtra:

    def test_the_market_downloader_names_the_extra(self,
                                                   monkeypatch):
        """The acceptance criterion: no yfinance, actionable message."""
        import importlib

        from beacon.exceptions import MissingDependencyError

        def refuse(name):
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", refuse)

        with pytest.raises(MissingDependencyError, match=r"py-beacon\[data\]"):
            yfinance_downloader()

    def test_the_reference_downloader_names_the_extra(self,
                                                      monkeypatch):
        import importlib

        from beacon.exceptions import MissingDependencyError

        def refuse(name):
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", refuse)

        with pytest.raises(MissingDependencyError, match=r"py-beacon\[data\]"):
            yfinance_reference_downloader()

    def test_the_module_imports_without_the_extra(self):
        """Injecting a downloader must not require the optional package."""
        import subprocess
        import sys

        script = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'yfinance':\n"
            "            raise ImportError(name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "from beacon.data.ingest import ingest_market_data\n"
            "import pandas as pd\n"
            "def fake(i, s, e):\n"
            "    return pd.DataFrame({'Close': [1.0]},\n"
            "                        index=pd.date_range('2024-01-01', periods=1))\n"
            "assert ingest_market_data(['AAA'], fake).rows == 1\n"
            "print('ok')\n"
        )

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True, text=True, check=False)

        assert completed.returncode == 0, completed.stderr
        assert "ok" in completed.stdout


class TestIngestResult:

    def test_an_empty_result_reports_nothing(self):
        result = IngestResult(market=pd.DataFrame())

        assert result.rows == 0
        assert not result.succeeded


async def _subscribe(registry):
    """Subscribe from inside the app's event loop."""
    return registry.subscribe()


async def _drain_queue(queue):
    """Read everything currently queued."""
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    return events


class FakeTicker:
    """Stands in for yfinance.Ticker, matching the two attributes used."""

    def __init__(self,
                 symbol: str):
        self.symbol = symbol

    def history(self,
                start=None,
                end=None,
                auto_adjust=False):
        return history()

    @property
    def info(self) -> dict[str, object]:
        return {"longName": f"{self.symbol} Corp", "currency": "USD"}


class FakeYFinance:
    """A stand-in module, so the real downloaders can be exercised offline."""
    Ticker = FakeTicker


class TestYFinanceDownloaders:
    """The adapters themselves, driven against a stand-in for the library.

    These are the only lines that talk to the optional dependency's API, so
    leaving them untested would mean the one place the mapping could be wrong
    is the one place nothing checks.
    """

    @pytest.fixture(autouse=True)
    def _fake_module(self,
                     monkeypatch):
        import importlib

        monkeypatch.setattr(importlib, "import_module",
                            lambda name: FakeYFinance())

    def test_the_market_downloader_returns_a_history(self):
        frame = yfinance_downloader()("AAA", "2024-01-01", "2024-01-05")

        assert len(frame) == PERIODS
        assert "Close" in frame.columns

    def test_its_output_normalises(self):
        """The adapter and the reshaper have to agree on the column names."""
        raw = yfinance_downloader()("AAA", None, None)

        assert len(normalise_history("AAA", raw)) == PERIODS

    def test_the_reference_downloader_returns_info(self):
        info = yfinance_reference_downloader()("AAA")

        assert info["longName"] == "AAA Corp"

    def test_a_full_market_ingest_runs_through_the_adapter(self):
        result = ingest_market_data(["AAA", "BBB"], yfinance_downloader())

        assert result.fetched == ["AAA", "BBB"]

    def test_a_full_reference_ingest_runs_through_the_adapter(self):
        result = ingest_reference_data(["AAA"], yfinance_reference_downloader())

        assert result.reference.iloc[0]["NAME"] == "AAA Corp"

    def test_the_reference_sync_endpoint_runs(self,
                                              fetcher):
        """The reference branch of the sync job, end to end."""
        config = ServerConfig(auth_token=TOKEN, data_fetcher=fetcher)

        with TestClient(create_app(config),
                        raise_server_exceptions=False) as client:
            job_id = client.post("/data/coverage/reference/sync",
                                 json={"identifiers": ["AAA"]},
                                 headers=auth()).json()["job_id"]
            client.portal.call(client.app.state.jobs.drain)
            result = client.get(f"/jobs/{job_id}", headers=auth()).json()

        assert result["status"] == "succeeded"
        assert result["result"]["dataset"] == "reference"
        assert fetcher.fetch_classification("AAA", scheme="NAME") == "AAA Corp"


class TestEmptyReferenceIngest:

    def test_no_identifiers_gives_an_empty_frame(self):
        result = ingest_reference_data([], fake_reference)

        assert result.reference.empty
        assert "IDENTIFIER" in result.reference.columns

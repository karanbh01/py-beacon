# tests/test_server_backtest.py
"""Contract tests for the backtest endpoint and its result consistency."""
import copy
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.backtest.result import BacktestResult, Book, IndexBooks
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.result import IndexResult
from beacon.portfolio.base import Portfolio
from beacon.server import ServerConfig, create_app
from beacon.server.backtests import annual_returns
from beacon.server.jobs import SUCCEEDED
from beacon.server.schemas import MAX_REBALANCES, BacktestResultSummary

TOKEN = "test-token-value"
# Spans a calendar-year boundary so the annual-return identity is exercised,
# but kept short: the calculator walks every business day and fetches prices
# per name per day, so a multi-year window makes this file dominate the suite.
START = "2023-10-02"
END = "2024-03-29"
DATES = pd.bdate_range(START, END)

# Two names on deterministic geometric paths over two calendar years, so the
# result spans a year boundary and the annual-return identity is exercised.
GROWTH = {"AAA": 1.30, "BBB": 0.85}
BASE_PRICE = {"AAA": 100.0, "BBB": 50.0}
SHARES = 1_000


def build_fetcher() -> DataFetcher:
    """Synthetic market and reference data with no randomness."""
    span = len(DATES) - 1
    rows = []
    for name, total_growth in GROWTH.items():
        for index, date in enumerate(DATES):
            fraction = index / span
            rows.append({"IDENTIFIER": name,
                         "DATE": date,
                         "CLOSE": BASE_PRICE[name] * (total_growth ** fraction),
                         "VOLUME": 1_000_000,
                         "SHARES_OUTSTANDING": SHARES})

    reference = pd.DataFrame([
        {"IDENTIFIER": name,
         "DATE_FROM": "2020-01-01",
         "NAME": name,
         "CURRENCY": "USD",
         "EXCHANGE": "NYSE"}
        for name in GROWTH
    ])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def index_document() -> dict:
    """An equal-weighted, quarterly-rebalanced index over the two names."""
    return copy.deepcopy({
        "id": "BT",
        "name": "Backtest Index",
        "base_date": START,
        "base_value": 1000.0,
        "currency": "USD",
        "rebalancing_frequency": "QUARTERLY",
        "description": None,
        "universe": {"universe_id": None, "identifiers": list(GROWTH)},
        "pipeline": {
            "selection": [],
            "weighting": {"id": "weighting",
                          "scheme": "EqualWeighted",
                          "params": {},
                          "max_weight": None},
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    })


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(tmp_path):
    """Entered client holding the index definition and a data source."""
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path)
    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        created = entered.post("/indices", json=index_document(), headers=auth())
        assert created.status_code == 200, created.json()
        yield entered


@pytest.fixture(scope="module")
def module_client(tmp_path_factory):
    """A client shared across the read-only consistency tests.

    Those tests all interrogate the same completed backtest, and a backtest is
    expensive enough that running one per test would dominate the suite.
    """
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("backtest"))
    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        created = entered.post("/indices", json=index_document(), headers=auth())
        assert created.status_code == 200, created.json()
        yield entered


def run_backtest(client,
                 **body) -> dict:
    """Submit a backtest, wait for it, and return the job's result."""
    response = client.post("/beacon/BT/backtest", json=body or {}, headers=auth())
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]

    client.portal.call(client.app.state.jobs.drain)

    job = client.get(f"/jobs/{job_id}", headers=auth()).json()
    assert job["status"] == SUCCEEDED, job

    return job["result"]


@pytest.fixture(scope="module")
def result(module_client) -> dict:
    """One completed backtest, shared by every test that only reads it."""
    return run_backtest(module_client)


class TestSubmission:

    def test_returns_a_job_immediately(self,
                                       client):
        response = client.post("/beacon/BT/backtest", json={}, headers=auth())

        assert response.status_code == 202
        assert response.json()["kind"] == "backtest:BT"
        assert response.json()["job_id"]

    def test_unknown_index_fails_before_a_job_is_created(self,
                                                         client):
        """Better a clean 404 than a job that fails a moment later."""
        response = client.post("/beacon/ABSENT/backtest", json={}, headers=auth())

        assert response.status_code == 404
        assert client.get("/jobs", headers=auth()).json()["jobs"] == []

    def test_without_a_data_source_is_a_configuration_error(self,
                                                            tmp_path):
        config = ServerConfig(auth_token=TOKEN, storage_root=tmp_path)
        with TestClient(create_app(config), raise_server_exceptions=False) as bare:
            bare.post("/indices", json=index_document(), headers=auth())

            response = bare.post("/beacon/BT/backtest", json={}, headers=auth())

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "CONFIGURATION_ERROR"

    def test_requires_authentication(self,
                                     client):
        assert client.post("/beacon/BT/backtest", json={}).status_code == 401

    def test_rejects_non_positive_capital(self,
                                          client):
        response = client.post("/beacon/BT/backtest",
                               json={"initial_capital": 0},
                               headers=auth())

        assert response.status_code == 422


class TestResultConsistency:
    """The acceptance criteria: every series reconciles against the others."""

    def test_level_compounds_exactly_from_the_returns(self,
                                                      result):
        level = result["level"]["data"]
        returns = result["returns"]["data"]

        rebuilt = [level[0]]
        for period_return in returns:
            rebuilt.append(rebuilt[-1] * (1 + period_return))

        assert len(rebuilt) == len(level)
        for expected, actual in zip(level, rebuilt, strict=True):
            assert math.isclose(actual, expected, rel_tol=1e-9)

    def test_drawdown_derives_from_the_level_series(self,
                                                    result):
        level = result["level"]["data"]
        drawdown = result["drawdown"]["data"]

        peak = level[0]
        for value, reported in zip(level, drawdown, strict=True):
            peak = max(peak, value)
            assert math.isclose(reported, value / peak - 1.0, abs_tol=1e-12)

    def test_drawdown_is_never_positive(self,
                                        result):
        assert all(value <= 1e-12 for value in result["drawdown"]["data"])

    def test_annual_returns_compound_to_the_total(self,
                                                  result):
        compounded = 1.0
        for annual in result["annual_returns"].values():
            compounded *= 1 + annual

        assert math.isclose(compounded - 1.0,
                            result["metrics"]["total_return"],
                            rel_tol=1e-9)

    def test_annual_returns_cover_every_year_in_the_series(self,
                                                           result):
        years = {label[:4] for label in result["level"]["index"]}

        assert set(result["annual_returns"]) == years

    def test_metrics_match_the_returned_series(self,
                                               result):
        """Recomputing from the payload must reproduce the reported metrics."""
        level = result["level"]["data"]
        returns = pd.Series(result["returns"]["data"])
        metrics = result["metrics"]

        total = level[-1] / level[0] - 1.0
        assert math.isclose(total, metrics["total_return"], rel_tol=1e-9)

        volatility = float(returns.std() * math.sqrt(252))
        assert math.isclose(volatility, metrics["volatility"], rel_tol=1e-9)

        assert math.isclose(min(result["drawdown"]["data"]),
                            metrics["max_drawdown"],
                            rel_tol=1e-9)

    def test_sharpe_is_consistent_with_the_other_metrics(self,
                                                         result):
        metrics = result["metrics"]

        if metrics["volatility"] > 0:
            expected = metrics["annualised_return"] / metrics["volatility"]
            assert math.isclose(expected, metrics["sharpe_ratio"], rel_tol=1e-9)


class TestSeriesShape:

    def test_level_starts_at_one_hundred(self,
                                         result):
        assert result["level"]["data"][0] == pytest.approx(100.0)

    def test_benchmark_starts_at_one_hundred(self,
                                             result):
        assert result["index_level"]["data"][0] == pytest.approx(100.0)

    def test_returns_is_one_shorter_than_level(self,
                                               result):
        assert len(result["returns"]["data"]) == len(result["level"]["data"]) - 1

    def test_index_is_iso_dates(self,
                                result):
        assert result["level"]["index"][0].startswith("2023-10")

    def test_tracking_metrics_are_present(self,
                                          result):
        """The backtest tracked an index, so tracking figures are meaningful."""
        assert result["metrics"]["tracking_error"] is not None
        assert result["metrics"]["tracking_difference"] is not None

    def test_costs_reduce_the_total_return(self,
                                           module_client):
        """Tightened once BN-85 (#104) landed.

        This previously asserted only that costs *changed* the result: the
        engine dropped a buy entirely when cash fell short, so a cost could
        remove one leg of a rebalance and leave the portfolio concentrated in
        whichever name performed better — making a costlier run look better.
        Orders are now sized down instead, so the intuitive direction holds.
        """
        free = run_backtest(module_client, transaction_cost_bps=0.0)
        costly = run_backtest(module_client, transaction_cost_bps=100.0)

        assert (costly["metrics"]["total_return"]
                < free["metrics"]["total_return"])


class TestProgressOverTheSocket:
    """The acceptance criterion: progress is visible over the WebSocket."""

    def test_progress_frames_reach_a_subscriber(self,
                                                client):
        with client.websocket_connect(f"/ws?token={TOKEN}") as socket:
            client.post("/beacon/BT/backtest", json={}, headers=auth())
            # Let the job finish before reading. The socket subscribed before
            # submission, so every frame was published to its queue and is
            # waiting — this proves delivery without a receive that could
            # block forever if the job never ran.
            client.portal.call(client.app.state.jobs.drain)

            statuses = []
            progresses = []
            for _ in range(40):
                event = socket.receive_json()
                if event["type"] != "job":
                    continue
                statuses.append(event["status"])
                progresses.append(event["progress"])
                # Break on ANY terminal state, not just success: a failed job
                # would otherwise leave this blocking on a frame that never
                # arrives, turning a clear assertion failure into a hang.
                if event["status"] in {"succeeded", "failed", "cancelled"}:
                    break

        assert statuses[-1] == SUCCEEDED
        assert progresses[-1] == 1.0
        assert any(0.0 < value < 1.0 for value in progresses), progresses


class TestAnnualReturnsHelper:
    """The telescoping definition, checked directly."""

    def test_compounds_to_the_total_across_years(self):
        dates = pd.to_datetime(["2022-06-01", "2022-12-30",
                                "2023-06-01", "2023-12-29"])
        level = pd.Series([100.0, 110.0, 120.0, 132.0], index=dates)

        returns = annual_returns(level)
        compounded = math.prod(1 + value for value in returns.values())

        assert set(returns) == {"2022", "2023"}
        assert math.isclose(compounded - 1.0, 132.0 / 100.0 - 1.0, rel_tol=1e-12)

    def test_empty_series_has_no_years(self):
        assert annual_returns(pd.Series(dtype=float)) == {}

    def test_single_year_return_is_the_whole_move(self):
        dates = pd.to_datetime(["2023-01-03", "2023-12-29"])
        level = pd.Series([100.0, 125.0], index=dates)

        assert annual_returns(level)["2023"] == pytest.approx(0.25)


@pytest.fixture(scope="module")
def record(module_client,
           result):
    """The record of the run the module fixture already paid for.

    Depends on `result` so the backtest has definitely completed before the
    record is read.
    """
    response = module_client.get("/beacon/BT/record", headers=auth())
    assert response.status_code == 200, response.text
    return response.json()


class TestTheRecord:
    """BN-158: `GET /beacon/{index_id}/record` — the books, finally served.

    Found by the beacon-ui session after its BN-155 migration: the nested
    `BacktestResultSummary` was defined, exported and tested, but referenced
    by no route — the OpenAPI spec was byte-identical before and after the
    reshape. The record is captured at job completion, because the library
    BacktestResult exists only inside the job.
    """


    def test_the_portfolio_book_arrives(self,
                                        record):
        book = record["portfolio"]

        assert book["portfolio_id"]
        assert book["positions_total"] > 0
        assert len(book["positions"]["data"]) > 0
        assert book["weights_dates_total"] > 0

    def test_nav_opens_with_the_capital(self,
                                        record):
        """The day-zero row survives serialisation: the first NAV value is
        the initial capital, dated before the first trading day."""
        book = record["portfolio"]

        assert book["nav"]["data"][0] == pytest.approx(
            book["initial_capital"])

    def test_the_target_book_is_present_and_the_benchmark_null(self,
                                                               record):
        """This run tracked an index and was given no benchmark. Null and
        present must both survive the wire — a client tells "not measured"
        from "measured and empty" by exactly this. Since BN-164 the index
        arrives as a `{target, optimised}` container, the flat
        `target_index` field gone."""
        assert record["index"]["target"] is not None
        assert len(record["index"]["target"]["levels"]["data"]) > 0
        assert record["index"]["optimised"] is None
        assert record["benchmark"] is None
        assert "target_index" not in record

    def test_a_never_backtested_index_404s_with_the_pointer(self,
                                                            module_client):
        created = module_client.post(
            "/indices", json={**index_document(), "id": "NEVER-RUN",
                              "name": "Never run"}, headers=auth())
        assert created.status_code == 200

        response = module_client.get("/beacon/NEVER-RUN/record",
                                     headers=auth())

        assert response.status_code == 404
        assert "backtest" in response.json()["error"]["message"]

    def test_it_requires_authentication(self,
                                        module_client):
        assert module_client.get("/beacon/BT/record").status_code == 401

    def test_the_route_is_in_the_spec(self,
                                      module_client):
        """The finding that started this: the spec did not change when the
        payload did. Now it must contain the path."""
        paths = module_client.get("/openapi.json").json()["paths"]

        assert "/beacon/{index_id}/record" in paths

    def test_deleting_the_index_deletes_the_record(self,
                                                   module_client):
        """BN-157's cascade extends to the record store, or the delete leaves
        exactly the orphan it exists to prevent.

        A dedicated index rather than the shared BT one: consuming the module
        fixture's index would break sibling tests under randomised ordering.
        The record is seeded through the store the job writes to, which is
        the layer the cascade must clean.
        """
        created = module_client.post(
            "/indices", json={**index_document(), "id": "DOOMED",
                              "name": "Doomed"}, headers=auth())
        assert created.status_code == 200

        # A skeletal record: enough for the store, deliberately not enough
        # for the route's response model -- this test is about the cascade,
        # and the pre-check that guards against a vacuous pass reads the
        # store directly.
        records = module_client.app.state.backtest_record_store
        records.write("DOOMED", {"portfolio": {"portfolio_id": "DOOMED"}})
        assert records.read("DOOMED") is not None

        deleted = module_client.delete("/indices/DOOMED", headers=auth())
        assert deleted.status_code == 200
        assert deleted.json()["deleted"][0]["backtest_record_deleted"] is True

        assert records.read("DOOMED") is None
        response = module_client.get("/beacon/DOOMED/record", headers=auth())

        assert response.status_code == 404


SNAPSHOT_FIELDS = {"date", "announced", "weights", "uncapped_weights",
                   "capped", "cap", "redistributed"}


def built_record(snapshot_dates,
                 announcements=None,
                 cap=None,
                 levels_only_benchmark=False) -> dict:
    """A record built from a hand-made result, not a calculation.

    The only way to reach the snapshot bound: 250 quarterly rebalances is 62
    years of index, and the calculator walks every business day of it.
    """
    days = pd.bdate_range(START, periods=2)
    index_result = IndexResult(
        index_id="HAND",
        index_levels=pd.Series([1000.0, 1010.0], index=days),
        divisor_history=pd.Series(1.0, index=days),
        constituent_snapshots={date: list(GROWTH) for date in snapshot_dates},
        weight_snapshots={date: {"AAA": 0.6, "BBB": 0.4}
                          for date in snapshot_dates},
        announcement_dates=dict(announcements or {}))

    portfolio = Portfolio(portfolio_id="hand", initial_cash=1000.0,
                          inception=days[0] - pd.tseries.offsets.BDay(1))
    for day in days:
        portfolio._history.record(day, {}, 1000.0)

    benchmark = (Book.from_levels(pd.Series([100.0, 101.0], index=days))
                 if levels_only_benchmark else None)
    result = BacktestResult(
        portfolio=portfolio,
        index=IndexBooks(target=Book.from_index(index_result)),
        benchmark=benchmark)

    return BacktestResultSummary.from_result(
        result, cap=cap).model_dump(mode="json")


class TestTheDecidedWeights:
    """BN-173: what each rebalance DECIDED, in the record that survives.

    The daily panel is what the index HELD — drift included — and the two agree
    only on a rebalance date. The decided weights used to reach the wire only
    through the transient run payload, so the question `uncapped_weights` exists
    to answer ("what did capping cost?") died with the job result.

    Bounded like the panel beside it, with its own constant: snapshots are a
    far sparser fact, and MAX_REBALANCES is reached here with a hand-made
    result rather than 60 years of calculation.
    """

    def test_the_target_book_carries_them(self,
                                          record):
        book = record["index"]["target"]

        assert book["rebalances"], "the record published no decided weights"
        assert book["rebalances_total"] == len(book["rebalances"])
        for entry in book["rebalances"]:
            assert set(entry) == SNAPSHOT_FIELDS
            assert sum(entry["weights"].values()) == pytest.approx(1.0)

    def test_they_are_the_rows_the_run_payload_published(self,
                                                         record,
                                                         result):
        """The property that matters most: two representations of one fact,
        from one run, must agree row for row. If they can disagree, a client
        reading the record instead of the job result is reading something
        else."""
        assert record["index"]["target"]["rebalances"] == result["rebalances"]

    def test_the_held_weights_are_a_different_fact(self,
                                                   record):
        """Not two copies of one: by the last day prices have moved the held
        weights off the last decision, which is why resampling the panel
        cannot answer what was decided."""
        book = record["index"]["target"]
        decided = book["rebalances"][-1]["weights"]
        columns = book["weights"]["columns"]
        held = dict(zip(columns, book["weights"]["data"][-1], strict=True))

        assert set(held) == set(decided)
        assert held != pytest.approx(decided)

    def test_every_rebalance_is_served_below_the_bound(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="MS")

        book = built_record(dates)["index"]["target"]

        assert len(book["rebalances"]) == 5
        assert book["rebalances_total"] == 5

    def test_over_the_bound_the_total_is_still_true(self):
        """Never silent truncation: the served slice shrinks, the total does
        not, so a client says "last 250 of 260" instead of believing it saw
        the whole methodology."""
        dates = pd.date_range("2000-01-01", periods=MAX_REBALANCES + 10,
                              freq="MS")

        book = built_record(dates)["index"]["target"]

        assert len(book["rebalances"]) == MAX_REBALANCES
        assert book["rebalances_total"] == MAX_REBALANCES + 10

    def test_the_most_recent_rebalances_are_the_kept_ones(self):
        """The daily panel's stance, for the same reason: the tail is what a
        client is looking at."""
        dates = pd.date_range("2000-01-01", periods=MAX_REBALANCES + 10,
                              freq="MS")

        served = [entry["date"]
                  for entry in built_record(dates)["index"]["target"]["rebalances"]]

        assert served[0] == dates[10].strftime("%Y-%m-%d")
        assert served[-1] == dates[-1].strftime("%Y-%m-%d")

    def test_the_announcement_date_survives(self):
        """Carried rather than dropped: its presence is itself the signal that
        an effective-date lag applies, which is a fact about the index and not
        about the job that happened to publish it."""
        effective = pd.Timestamp("2024-01-02")

        record = built_record([effective],
                             announcements={effective: pd.Timestamp("2023-12-28")})
        entry = record["index"]["target"]["rebalances"][0]

        assert entry["date"] == "2024-01-02"
        assert entry["announced"] == "2023-12-28"

    def test_an_unlagged_index_announces_nothing(self,
                                                 record):
        """Null rather than a repeat of the effective date, so the lag reads
        as absent instead of as zero."""
        assert all(entry["announced"] is None
                   for entry in record["index"]["target"]["rebalances"])

    def test_the_cap_is_stamped_from_the_definition(self):
        """Not read off the cap reports, which the calculator files only where
        the cap actually bound: "a 35% cap applies and nothing reached it" and
        "no cap applies" are different answers."""
        book = built_record([pd.Timestamp("2024-01-02")],
                            cap=0.35)["index"]["target"]

        assert book["rebalances"][0]["cap"] == 0.35
        assert book["rebalances"][0]["capped"] == []

    def test_a_book_of_bare_levels_decided_nothing(self):
        """A comparator given as a level series has no snapshots to publish —
        empty, because there is nothing to truncate."""
        book = built_record([pd.Timestamp("2024-01-02")],
                            levels_only_benchmark=True)["benchmark"]

        assert book["rebalances"] == []
        assert book["rebalances_total"] == 0


class TestTheListing:
    """BN-162: `GET /beacon/backtests` — which indices HAVE a record.

    The record endpoint answers per index, so a client wanting to offer
    backtests as search results had no way to enumerate them. The listing is
    deliberately thin — id and capture time — because everything else is one
    `/record` call away, and names live in the catalogue the client already
    holds.
    """

    def test_an_empty_store_lists_nothing(self,
                                          client):
        response = client.get("/beacon/backtests", headers=auth())

        assert response.status_code == 200
        assert response.json() == []

    def test_a_run_appears_with_a_parseable_utc_stamp(self,
                                                      module_client,
                                                      result):
        """The stamp is the row's whole value over a boolean: "backtest ·
        3 days ago" is a row someone can judge, a flag is not."""
        from datetime import datetime

        rows = module_client.get("/beacon/backtests", headers=auth()).json()
        ours = [row for row in rows if row["index_id"] == "BT"]

        assert len(ours) == 1
        stamp = datetime.fromisoformat(ours[0]["run_at"])
        assert stamp.tzinfo is not None

    def test_newest_first_and_unstamped_last(self,
                                             client):
        """Records written before BN-162 carry no stamp; they sort after
        every dated row rather than lying about when they ran. Seeded
        skeletally in a function-scoped store: ordering is the subject, so
        the rows are arranged, not raced."""
        records = client.app.state.backtest_record_store
        records.write("OLD", {"run_at": "2020-01-01T00:00:00+00:00"})
        records.write("NEW", {"run_at": "2030-01-01T00:00:00+00:00"})
        records.write("UNSTAMPED", {})

        rows = client.get("/beacon/backtests", headers=auth()).json()

        assert [row["index_id"] for row in rows] == ["NEW", "OLD", "UNSTAMPED"]
        assert rows[-1]["run_at"] is None

    def test_a_corrupt_record_is_skipped_not_a_500(self,
                                                   client):
        """One bad file must not hide every good one: the search bar losing
        all its backtest rows to a single truncated write would report the
        store empty when it is merely imperfect."""
        records = client.app.state.backtest_record_store
        records.write("GOOD", {"run_at": "2025-01-01T00:00:00+00:00"})
        (records.directory / "BAD.json").write_text("{not json",
                                                    encoding="utf-8")

        rows = client.get("/beacon/backtests", headers=auth()).json()

        assert [row["index_id"] for row in rows] == ["GOOD"]

    def test_deleting_the_index_removes_the_row(self,
                                                client):
        """The acceptance case: the BN-157 cascade reaches the listing."""
        records = client.app.state.backtest_record_store
        records.write("BT", {"run_at": "2025-01-01T00:00:00+00:00"})

        deleted = client.delete("/indices/BT", headers=auth())
        assert deleted.status_code == 200

        assert client.get("/beacon/backtests", headers=auth()).json() == []

    def test_it_requires_authentication(self,
                                        client):
        assert client.get("/beacon/backtests").status_code == 401

    def test_the_route_is_in_the_spec(self,
                                      client):
        paths = client.get("/openapi.json").json()["paths"]

        assert "/beacon/backtests" in paths

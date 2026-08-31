# tests/test_benchmark.py
"""Tests for relative performance and external benchmark resolution."""
import copy

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.analysis.relative import (
    MINIMUM_ALIGNED_OBSERVATIONS,
    align_on_common_window,
    relative_metrics,
)
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.server import ServerConfig, create_app
from beacon.server.jobs import SUCCEEDED

TOKEN = "test-token-value"
START = "2023-10-02"
END = "2024-03-29"
DATES = pd.bdate_range(START, END)

# Two index constituents plus a separate series usable as an external
# benchmark, so a backtest can be compared against something it never tracked.
GROWTH = {"AAA": 1.30, "BBB": 0.85, "MKT": 1.10}
BASE_PRICE = {"AAA": 100.0, "BBB": 50.0, "MKT": 1000.0}
CONSTITUENTS = ["AAA", "BBB"]


def levels(growth: float,
           periods: int = 60,
           start: str = "2024-01-01") -> pd.Series:
    """A geometric level series, for the pure relative-metrics tests."""
    dates = pd.bdate_range(start, periods=periods)
    fractions = np.arange(periods) / max(periods - 1, 1)

    return pd.Series(100.0 * growth ** fractions, index=dates)


def wobbly_levels(growth: float,
                  periods: int = 60,
                  seed: int = 0,
                  start: str = "2024-01-01") -> pd.Series:
    """A level series with genuine period-to-period return variation.

    Needed wherever a test touches correlation or beta: a smooth geometric
    series has constant returns, so those quantities are undefined and any
    value computed from one is rounding noise.
    """
    dates = pd.bdate_range(start, periods=periods)
    generator = np.random.default_rng(seed)

    drift = growth ** (1.0 / max(periods - 1, 1)) - 1.0
    returns = drift + generator.normal(0.0, 0.01, size=periods)
    returns[0] = 0.0

    return pd.Series(100.0 * (1.0 + returns).cumprod(), index=dates)


def build_fetcher() -> DataFetcher:
    """Market and reference data covering the constituents and the benchmark."""
    span = len(DATES) - 1
    rows = [
        {"IDENTIFIER": name,
         "DATE": date,
         "CLOSE": BASE_PRICE[name] * (GROWTH[name] ** (index / span)),
         "VOLUME": 1_000_000,
         "SHARES_OUTSTANDING": 1_000}
        for name in GROWTH
        for index, date in enumerate(DATES)
    ]
    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "NYSE"}
        for name in GROWTH
    ])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def index_document(index_id: str = "BT",
                   universe: list[str] | None = None) -> dict:
    """An equal-weighted quarterly index over the given universe."""
    return copy.deepcopy({
        "id": index_id,
        "name": f"Index {index_id}",
        "base_date": START,
        "base_value": 1000.0,
        "currency": "USD",
        "rebalancing_frequency": "QUARTERLY",
        "description": None,
        "universe": {"universe_id": None,
                     "identifiers": universe or CONSTITUENTS},
        "pipeline": {
            "selection": [],
            "weighting": {"id": "weighting", "scheme": "EqualWeighted",
                          "params": {}, "max_weight": None},
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    })


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


class TestAlignment:

    def test_identical_indexes_are_unchanged(self):
        series = levels(1.2)

        left, right = align_on_common_window(series, series)

        assert len(left) == len(series)
        assert left.index.equals(right.index)

    def test_trims_to_the_shared_window(self):
        portfolio = levels(1.2, periods=60, start="2024-01-01")
        benchmark = levels(1.1, periods=60, start="2024-02-01")

        left, right = align_on_common_window(portfolio, benchmark)

        assert left.index.equals(right.index)
        assert len(left) < len(portfolio)
        assert left.index[0] == benchmark.index[0]

    def test_result_is_sorted(self):
        portfolio = levels(1.2)
        shuffled = portfolio.sample(frac=1.0, random_state=3)

        left, _ = align_on_common_window(portfolio, shuffled)

        assert left.index.is_monotonic_increasing

    def test_disjoint_windows_raise(self):
        """The acceptance criterion — a mapped error, not a meaningless number."""
        portfolio = levels(1.2, periods=30, start="2024-01-01")
        benchmark = levels(1.1, periods=30, start="2025-01-01")

        with pytest.raises(CalculationError, match="share only 0 date"):
            align_on_common_window(portfolio, benchmark)

    def test_the_error_reports_both_spans(self):
        """So a caller can see *why* they do not overlap."""
        portfolio = levels(1.2, periods=30, start="2024-01-01")
        benchmark = levels(1.1, periods=30, start="2025-01-01")

        with pytest.raises(CalculationError, match="2024-01-01"):
            align_on_common_window(portfolio, benchmark)

    def test_barely_overlapping_windows_raise(self):
        """Two shared dates give one return: not a tracking error."""
        portfolio = levels(1.2, periods=30, start="2024-01-01")
        overlap = portfolio.index[-2:]
        benchmark = pd.Series([100.0, 101.0], index=overlap)

        with pytest.raises(CalculationError, match="at least 3"):
            align_on_common_window(portfolio, benchmark)

    def test_empty_series_raises(self):
        with pytest.raises(CalculationError, match="either series is empty"):
            align_on_common_window(pd.Series(dtype=float), levels(1.1))

    def test_minimum_is_three_observations(self):
        assert MINIMUM_ALIGNED_OBSERVATIONS == 3


class TestRelativeMetrics:

    def test_identical_series_have_no_excess_return(self):
        series = levels(1.25)

        metrics = relative_metrics(series, series)

        assert metrics.excess_return == pytest.approx(0.0, abs=1e-12)
        assert metrics.tracking_error == pytest.approx(0.0, abs=1e-12)

    def test_identical_series_are_perfectly_correlated(self):
        """Needs a varying series: a constant one has nothing to correlate."""
        series = wobbly_levels(1.25, seed=4)

        metrics = relative_metrics(series, series)

        assert metrics.correlation == pytest.approx(1.0)
        assert metrics.beta == pytest.approx(1.0)

    def test_identical_smooth_series_still_have_no_excess_return(self):
        """Excess return and tracking error are defined even without variation."""
        series = levels(1.25)

        metrics = relative_metrics(series, series)

        assert metrics.excess_return == pytest.approx(0.0, abs=1e-12)
        assert metrics.tracking_error == pytest.approx(0.0, abs=1e-12)

    def test_total_returns_are_hand_checkable(self):
        portfolio = levels(1.25)
        benchmark = levels(1.10)

        metrics = relative_metrics(portfolio, benchmark)

        assert metrics.total_return == pytest.approx(0.25)
        assert metrics.benchmark_return == pytest.approx(0.10)

    def test_excess_return_is_the_difference(self):
        metrics = relative_metrics(levels(1.25), levels(1.10))

        assert metrics.excess_return == pytest.approx(
            metrics.total_return - metrics.benchmark_return)

    def test_outperformance_is_positive(self):
        assert relative_metrics(levels(1.30), levels(1.05)).excess_return > 0

    def test_underperformance_is_negative(self):
        assert relative_metrics(levels(1.05), levels(1.30)).excess_return < 0

    def test_metrics_are_scale_invariant(self):
        """Returns do not depend on the level, so rebasing must not matter.

        Uses wobbly series rather than the smooth geometric ones: a perfectly
        smooth series has constant returns, so its beta is a ratio of two
        rounding errors and is not scale-invariant in floating point. That is
        exactly why `_beta` treats negligible variation as no variation.
        """
        portfolio = wobbly_levels(1.25, seed=1)
        benchmark = wobbly_levels(1.10, seed=2)

        original = relative_metrics(portfolio, benchmark)
        rescaled = relative_metrics(portfolio * 7.0, benchmark / 3.0)

        assert rescaled.total_return == pytest.approx(original.total_return)
        assert rescaled.tracking_error == pytest.approx(original.tracking_error)
        assert rescaled.beta == pytest.approx(original.beta)
        assert rescaled.correlation == pytest.approx(original.correlation)

    def test_a_smooth_series_reports_no_beta_rather_than_noise(self):
        """Constant returns carry no information about sensitivity."""
        metrics = relative_metrics(levels(1.25), levels(1.10))

        assert metrics.beta == 0.0
        assert metrics.correlation == 0.0

    def test_a_varying_benchmark_gives_a_real_beta(self):
        """A portfolio moving twice as hard as its benchmark has beta near 2."""
        benchmark = wobbly_levels(1.10, seed=5)
        benchmark_returns = benchmark.pct_change().fillna(0.0)
        doubled = 100.0 * (1.0 + 2.0 * benchmark_returns).cumprod()

        metrics = relative_metrics(doubled, benchmark)

        assert metrics.beta == pytest.approx(2.0, rel=0.05)

    def test_reports_the_aligned_window(self):
        portfolio = levels(1.2, periods=60, start="2024-01-01")
        benchmark = levels(1.1, periods=60, start="2024-02-01")

        metrics = relative_metrics(portfolio, benchmark)

        assert metrics.observations < len(portfolio)
        assert metrics.start == benchmark.index[0].isoformat()

    def test_tracking_error_grows_with_divergence(self):
        base = levels(1.10)
        noise = pd.Series(np.linspace(0, 5, len(base)), index=base.index)

        quiet = relative_metrics(base + noise * 0.1, base).tracking_error
        loud = relative_metrics(base + noise * np.sin(
            np.arange(len(base))) * 2, base).tracking_error

        assert loud > quiet

    def test_a_flat_benchmark_gives_zero_beta_not_an_error(self):
        """No benchmark movement means nothing to be sensitive to."""
        flat = pd.Series(100.0, index=levels(1.2).index)

        metrics = relative_metrics(levels(1.2), flat)

        assert metrics.beta == 0.0
        assert metrics.correlation == 0.0

    def test_a_zero_starting_level_is_rejected(self):
        series = levels(1.2)
        broken = series.copy()
        broken.iloc[0] = 0.0

        with pytest.raises(CalculationError, match="starts at zero"):
            relative_metrics(broken, series)


@pytest.fixture
def client(tmp_path):
    """Client holding an index and a data source covering the benchmark."""
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path)
    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        assert entered.post("/indices", json=index_document(),
                            headers=auth()).status_code == 200
        yield entered


def run_backtest(client,
                 **body) -> dict:
    """Submit a backtest, drain it, and return the job's result."""
    response = client.post("/beacon/BT/backtest", json=body or {}, headers=auth())
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]

    client.portal.call(client.app.state.jobs.drain)
    job = client.get(f"/jobs/{job_id}", headers=auth()).json()

    return job


class TestBenchmarkAgainstAnIdentifier:
    """The acceptance criterion: a loaded market-data identifier works."""

    def test_produces_aligned_series_and_metrics(self,
                                                 client):
        job = run_backtest(client,
                           benchmark={"kind": "identifier", "id": "MKT"})

        assert job["status"] == SUCCEEDED, job
        benchmark = job["result"]["benchmark"]

        assert benchmark["reference"]["id"] == "MKT"
        assert benchmark["observations"] > MINIMUM_ALIGNED_OBSERVATIONS
        assert benchmark["level"]["data"][0] == pytest.approx(100.0)

    def test_benchmark_return_matches_the_series(self,
                                                client):
        """MKT grows 10% across the window by construction."""
        job = run_backtest(client, benchmark={"kind": "identifier", "id": "MKT"})

        assert job["result"]["benchmark"]["benchmark_return"] == pytest.approx(
            0.10, abs=0.01)

    def test_excess_return_reconciles(self,
                                      client):
        benchmark = run_backtest(
            client, benchmark={"kind": "identifier", "id": "MKT"})["result"]["benchmark"]

        assert benchmark["excess_return"] == pytest.approx(
            benchmark["total_return"] - benchmark["benchmark_return"])

    def test_unknown_identifier_fails_the_job_with_a_mapped_error(self,
                                                                 client):
        job = run_backtest(client, benchmark={"kind": "identifier", "id": "NOPE"})

        assert job["status"] == "failed"
        assert "NOPE" in job["error"]

    def test_unknown_price_column_fails(self,
                                       client):
        job = run_backtest(client,
                           benchmark={"kind": "identifier", "id": "MKT",
                                      "price_column": "NOT_A_COLUMN"})

        assert job["status"] == "failed"


class TestBenchmarkAgainstAnotherIndex:
    """The acceptance criterion: another stored index works."""

    def test_produces_aligned_series_and_metrics(self,
                                                 client):
        client.post("/indices", json=index_document("SOLO", universe=["AAA"]),
                    headers=auth())

        job = run_backtest(client, benchmark={"kind": "index", "id": "SOLO"})

        assert job["status"] == SUCCEEDED, job
        benchmark = job["result"]["benchmark"]

        assert benchmark["reference"]["kind"] == "index"
        assert benchmark["reference"]["id"] == "SOLO"
        assert benchmark["observations"] > MINIMUM_ALIGNED_OBSERVATIONS

    def test_an_all_winner_benchmark_beats_the_mixed_index(self,
                                                           client):
        """SOLO holds only the riser, so the mixed index must trail it."""
        client.post("/indices", json=index_document("SOLO", universe=["AAA"]),
                    headers=auth())

        benchmark = run_backtest(
            client, benchmark={"kind": "index", "id": "SOLO"})["result"]["benchmark"]

        assert benchmark["excess_return"] < 0

    def test_unknown_benchmark_index_fails_the_job(self,
                                                  client):
        job = run_backtest(client, benchmark={"kind": "index", "id": "ABSENT"})

        assert job["status"] == "failed"
        assert "ABSENT" in job["error"]


class TestBenchmarkIsOptional:

    def test_omitted_benchmark_leaves_the_field_null(self,
                                                    client):
        job = run_backtest(client)

        assert job["status"] == SUCCEEDED
        assert job["result"]["benchmark"] is None

    def test_replication_metrics_are_unaffected(self,
                                                client):
        """The tracked index comparison is separate and always present."""
        without = run_backtest(client)["result"]
        with_benchmark = run_backtest(
            client, benchmark={"kind": "identifier", "id": "MKT"})["result"]

        assert without["metrics"]["tracking_error"] is not None
        assert with_benchmark["metrics"]["tracking_error"] == pytest.approx(
            without["metrics"]["tracking_error"])

    def test_tracked_index_series_is_still_reported(self,
                                                   client):
        result = run_backtest(
            client, benchmark={"kind": "identifier", "id": "MKT"})["result"]

        assert result["index_level"]["data"][0] == pytest.approx(100.0)
        assert result["benchmark"]["level"]["data"][0] == pytest.approx(100.0)

    def test_an_invalid_kind_is_rejected_before_the_job(self,
                                                       client):
        response = client.post("/beacon/BT/backtest",
                               json={"benchmark": {"kind": "nonsense", "id": "MKT"}},
                               headers=auth())

        assert response.status_code == 422

    def test_the_schema_documents_the_benchmark_field(self,
                                                     client):
        schema = client.app.openapi()

        assert "BenchmarkRef" in schema["components"]["schemas"]
        assert "benchmark" in schema["components"]["schemas"]["BacktestRequest"][
            "properties"]

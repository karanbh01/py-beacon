# tests/test_server_views.py
"""BN-71: Beacon View read endpoints.

Every test drives a real backtest through the job machinery first, then reads
the panes off the stored result — which is the flow the endpoints exist to
serve, and the only way to check that a run and the views of it agree.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.exceptions import DataNotFoundError
from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "test-token-value"
INDEX_ID = "canon"
CAPPED_ID = "canon-capped"

# Short enough to keep the suite quick, long enough for several quarterly
# rebalances and a meaningful attribution window.
START = "2023-01-02"
END = "2024-06-28"


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def definition(index_id: str,
               cap: float | None = None) -> dict:
    """A stored index definition over the canonical universe."""
    # Equal weighting hands every name exactly 1/n, so no cap above 1/n can
    # ever bind at a rebalance. The capped variant is therefore market-cap
    # weighted, where the weights are uneven and a cap has something to bite on.
    if cap is None:
        weighting: dict = {"id": "weighting", "scheme": "EqualWeighted",
                           "params": {}}
    else:
        weighting = {"id": "weighting", "scheme": "MarketCapWeighted",
                     "params": {"use_free_float": True}, "max_weight": cap}

    return {
        "id": index_id,
        "name": f"Canonical {index_id}",
        "base_date": START,
        "base_value": 1000.0,
        "currency": "USD",
        "rebalancing_frequency": "QUARTERLY",
        "universe": {"universe_id": None, "identifiers": list(dataset.UNIVERSE)},
        "pipeline": {
            "selection": [],
            "weighting": weighting,
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    }


@pytest.fixture(scope="module")
def client():
    """A server with the canonical dataset and two backtests already run."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as storage:
        config = ServerConfig(auth_token=TOKEN,
                              data_fetcher=dataset.data_fetcher(),
                              storage_root=Path(storage))

        with TestClient(create_app(config),
                        raise_server_exceptions=False) as started:
            for index_id, cap in ((INDEX_ID, None), (CAPPED_ID, 0.20)):
                started.put(f"/indices/{index_id}",
                            json=definition(index_id, cap), headers=auth())
                started.post(f"/beacon/{index_id}/backtest",
                             json={"start": START, "end": END,
                                   "transaction_cost_bps": 10.0},
                             headers=auth())

            started.portal.call(started.app.state.jobs.drain)

            yield started


class TestOverview:

    def test_it_describes_the_run(self,
                                  client):
        payload = client.get(f"/beacon/{INDEX_ID}/overview", headers=auth()).json()

        assert payload["index_id"] == INDEX_ID
        assert payload["observations"] > 300
        assert payload["rebalances"] >= 5

    def test_it_carries_the_headline_metrics(self,
                                             client):
        metrics = client.get(f"/beacon/{INDEX_ID}/overview",
                             headers=auth()).json()["metrics"]

        assert "total_return" in metrics
        assert "max_drawdown" in metrics

    def test_it_reports_concentration_of_the_latest_rebalance(self,
                                                              client):
        payload = client.get(f"/beacon/{INDEX_ID}/overview", headers=auth()).json()
        concentration = payload["concentration"]

        assert concentration["constituents"] == len(dataset.UNIVERSE)
        # Equal weights over six names: effective count equals the real count.
        assert concentration["effective_assets"] == pytest.approx(6.0, abs=1e-6)

    def test_an_unknown_index_is_a_404(self,
                                       client):
        response = client.get("/beacon/nope/overview", headers=auth())

        assert response.status_code == 404

    def test_it_requires_authentication(self,
                                        client):
        assert client.get(f"/beacon/{INDEX_ID}/overview").status_code == 401


class TestWeights:

    def test_it_returns_the_latest_composition(self,
                                               client):
        payload = client.get(f"/beacon/{INDEX_ID}/weights", headers=auth()).json()

        assert set(payload["weights"]) == set(dataset.UNIVERSE)
        assert sum(payload["weights"].values()) == pytest.approx(1.0, abs=1e-9)

    def test_an_as_of_date_resolves_to_the_rebalance_in_force(self,
                                                              client):
        """An index holds the weights set at its last rebalance until the next,
        so the answer is dated earlier than the question."""
        payload = client.get(f"/beacon/{INDEX_ID}/weights",
                             params={"asof": "2023-08-15"}, headers=auth()).json()

        assert payload["as_of"] == "2023-08-15"
        assert payload["rebalance_date"] <= "2023-08-15"

    def test_a_date_before_the_first_rebalance_is_a_404(self,
                                                        client):
        """Returning the first rebalance would report weights not yet in force."""
        response = client.get(f"/beacon/{INDEX_ID}/weights",
                              params={"asof": "2020-01-01"}, headers=auth())

        assert response.status_code == 404

    def test_drift_is_measured_from_the_rebalance_targets(self,
                                                          client):
        """Not rebalance-to-rebalance: an equal-weighted index resets to 1/n
        every time, so that comparison would report zero drift forever. The
        question worth answering is how far prices have moved the held weights
        away from the targets since the last reset."""
        payload = client.get(f"/beacon/{INDEX_ID}/weights",
                             params={"asof": "2023-08-15"}, headers=auth()).json()

        assert payload["drift"] is not None
        assert payload["drift"]["total_absolute"] > 0.0
        assert payload["drift"]["worst"] in dataset.UNIVERSE
        assert payload["drift"]["since"] == payload["rebalance_date"]

    def test_drift_is_null_on_the_rebalance_itself(self,
                                                  client):
        """The weights were just set; reporting zeros would claim a
        measurement rather than the absence of one."""
        first = client.get(f"/beacon/{INDEX_ID}/weights",
                           params={"asof": START}, headers=auth()).json()

        assert first["drift"] is None

    def test_turnover_is_half_the_total_drift(self,
                                              client):
        drift = client.get(f"/beacon/{INDEX_ID}/weights",
                           params={"asof": "2023-08-15"},
                           headers=auth()).json()["drift"]

        assert drift["turnover"] == pytest.approx(drift["total_absolute"] / 2)

    def test_an_uncapped_index_reports_no_cap(self,
                                              client):
        payload = client.get(f"/beacon/{INDEX_ID}/weights", headers=auth()).json()

        assert payload["cap"] is None
        assert payload["capped"] == []

    def test_a_capped_index_reports_its_cap(self,
                                            client):
        """The cap comes from the definition, not from whether it bound.

        "A cap applies and nothing reached it" and "no cap applies" are
        different statements about a methodology, so the figure is reported
        either way.
        """
        payload = client.get(f"/beacon/{CAPPED_ID}/weights", headers=auth()).json()

        assert payload["cap"] == pytest.approx(0.20)
        assert all(weight <= 0.20 + 1e-9 for weight in payload["weights"].values())

    def test_a_binding_cap_names_what_it_held(self,
                                              client):
        """Market-cap weights are uneven, so the cap has something to bite on."""
        payload = client.get(f"/beacon/{CAPPED_ID}/weights", headers=auth()).json()

        assert payload["capped"]
        assert payload["cap_redistributed"] > 0.0


class TestAttribution:

    @pytest.fixture(scope="class")
    def payload(self,
                client):
        return client.get(f"/beacon/{INDEX_ID}/attribution", headers=auth()).json()

    def test_contributions_reconcile(self,
                                     payload):
        """The acceptance criterion.

        Carino linking makes the contributions sum to the compounded total
        return exactly rather than approximately, so this is an equality and
        not a tolerance dressed up as one.
        """
        assert payload["reconciles"] is True
        assert abs(payload["residual"]) < 1e-12

    def test_the_contributions_sum_to_the_total(self,
                                                payload):
        explained = sum(item["contribution"] for item in payload["contributions"])

        assert explained == pytest.approx(payload["total_return"], abs=1e-12)

    def test_every_constituent_is_accounted_for(self,
                                                payload):
        assert {item["asset_id"] for item in payload["contributions"]} == set(
            dataset.UNIVERSE)

    def test_it_reports_the_cost_drag(self,
                                      payload):
        """The run was costed at 10bp, so costs took something out."""
        assert payload["cost_drag"] is not None
        assert payload["cost_drag"] < 0.0

    def test_an_uncapped_index_reports_no_cap_drag(self,
                                                   payload):
        """0.0 would claim capping happened and made no difference."""
        assert payload["cap_drag"] is None

    def test_a_capped_index_reports_a_cap_drag(self,
                                               client):
        payload = client.get(f"/beacon/{CAPPED_ID}/attribution",
                             headers=auth()).json()

        assert payload["cap_drag"] is not None

    def test_a_window_narrows_the_period_count(self,
                                               client,
                                               payload):
        narrowed = client.get(f"/beacon/{INDEX_ID}/attribution",
                              params={"start": "2023-06-01", "end": "2023-12-31"},
                              headers=auth()).json()

        assert narrowed["periods"] < payload["periods"]

    def test_a_windowed_attribution_still_reconciles(self,
                                                     client):
        narrowed = client.get(f"/beacon/{INDEX_ID}/attribution",
                              params={"start": "2023-06-01", "end": "2023-12-31"},
                              headers=auth()).json()

        assert narrowed["reconciles"] is True


class TestAssetView:

    @pytest.fixture(scope="class")
    def payload(self,
                client):
        return client.get(f"/beacon/{INDEX_ID}/assets/AAA", headers=auth()).json()

    def test_it_reports_the_weight_history(self,
                                           payload):
        assert payload["rebalances_held"] >= 5
        assert all(0.0 < weight < 1.0 for weight in payload["weight_history"].values())

    def test_the_history_is_keyed_by_rebalance_date(self,
                                                    payload):
        for date in payload["weight_history"]:
            assert len(date) == 10 and date[4] == "-"

    def test_it_compares_the_name_against_the_index(self,
                                                    payload):
        assert payload["excess_return"] == pytest.approx(
            payload["total_return"] - payload["index_return"], abs=1e-9)

    def test_it_reports_a_beta_against_the_index(self,
                                                 payload):
        assert payload["beta"] > 0.0
        assert -1.0 <= payload["correlation"] <= 1.0

    def test_it_carries_the_price_series(self,
                                         payload):
        assert len(payload["price"]["data"]) > 300

    def test_a_name_the_index_never_held_is_a_404(self,
                                                  client):
        response = client.get(f"/beacon/{INDEX_ID}/assets/NOT_HELD", headers=auth())

        assert response.status_code == 404


class TestCompare:

    @pytest.fixture(scope="class")
    def payload(self,
                client):
        return client.get("/beacon/compare",
                          params={"ids": [INDEX_ID, CAPPED_ID]},
                          headers=auth()).json()

    def test_it_returns_an_entry_per_index(self,
                                           payload):
        assert [entry["index_id"] for entry in payload["entries"]] == [
            INDEX_ID, CAPPED_ID]

    def test_every_series_covers_the_same_dates(self,
                                                payload):
        """The acceptance criterion: aligned on the common window.

        Two indices with different spans compared over different periods would
        differ for no reason but their history, which is exactly the artefact
        this endpoint exists to remove.
        """
        lengths = {len(entry["level"]["data"]) for entry in payload["entries"]}

        assert lengths == {payload["observations"]}

    def test_the_windows_agree_index_for_index(self,
                                               payload):
        first, second = (entry["level"]["index"] for entry in payload["entries"])

        assert first == second

    def test_every_series_starts_at_one_hundred(self,
                                                payload):
        """Rebased on the first shared date, so the comparison is of shape."""
        for entry in payload["entries"]:
            assert entry["level"]["data"][0] == pytest.approx(100.0)

    def test_the_reported_window_matches_the_series(self,
                                                    payload):
        for entry in payload["entries"]:
            assert entry["level"]["index"][0].startswith(payload["start"])
            assert entry["level"]["index"][-1].startswith(payload["end"])

    def test_total_return_is_measured_on_the_shared_window(self,
                                                           payload):
        for entry in payload["entries"]:
            level = entry["level"]["data"]
            assert entry["total_return"] == pytest.approx(
                level[-1] / level[0] - 1.0, abs=1e-9)

    def test_a_single_id_is_refused(self,
                                    client):
        response = client.get("/beacon/compare", params={"ids": [INDEX_ID]},
                              headers=auth())

        assert response.status_code == 404

    def test_an_unknown_id_is_a_404(self,
                                    client):
        """Naming the unknown id, rather than quietly comparing fewer."""
        response = client.get("/beacon/compare",
                              params={"ids": [INDEX_ID, "nope"]}, headers=auth())

        assert response.status_code == 404

    def test_compare_is_not_captured_by_the_index_route(self,
                                                        client):
        """`/beacon/compare` must not read as an index called "compare"."""
        response = client.get("/beacon/compare",
                              params={"ids": [INDEX_ID, CAPPED_ID]},
                              headers=auth())

        assert response.status_code == 200


class TestWithoutARun:

    def test_reading_an_index_nobody_backtested_is_a_404(self):
        """There is no view of an index nobody has calculated."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as storage:
            config = ServerConfig(auth_token=TOKEN,
                                  data_fetcher=dataset.data_fetcher(),
                                  storage_root=Path(storage))

            with TestClient(create_app(config),
                            raise_server_exceptions=False) as client:
                client.put(f"/indices/{INDEX_ID}", json=definition(INDEX_ID),
                           headers=auth())

                response = client.get(f"/beacon/{INDEX_ID}/overview", headers=auth())

        assert response.status_code == 404
        assert "backtest" in response.json()["error"]["message"].lower()


class TestOpenApi:

    def test_every_view_endpoint_is_documented(self,
                                               client):
        paths = client.app.openapi()["paths"]

        for path in ("/beacon/{index_id}/overview",
                     "/beacon/{index_id}/weights",
                     "/beacon/{index_id}/attribution",
                     "/beacon/{index_id}/assets/{identifier}",
                     "/beacon/compare"):
            assert path in paths, path


class TestErrorPaths:
    """The refusals, driven against the derivation functions directly.

    Each is a case where saying nothing is better than answering: a run stored
    before composition was recorded, a window the run does not cover, indices
    that never traded on the same day. Serving a plausible-looking empty answer
    for any of them would be worse than a 404.
    """

    def test_a_run_without_composition_is_refused(self):
        """A result stored before BN-71 has a level but no weights."""
        from beacon.server.views import snapshots_from

        with pytest.raises(DataNotFoundError, match="rebalance snapshots"):
            snapshots_from({"level": {"index": [], "data": []}})

    def test_a_run_with_a_malformed_rebalance_list_is_refused(self):
        from beacon.server.views import snapshots_from

        with pytest.raises(DataNotFoundError, match="rebalance snapshots"):
            snapshots_from({"rebalances": "not a list"})

    def test_a_window_the_run_does_not_cover_is_refused(self):
        """Attributing over no periods would report a flat index rather than a
        question that could not be answered."""
        from beacon.server.views import _window_of

        index = pd.to_datetime(["2024-01-01", "2024-01-02"])

        with pytest.raises(DataNotFoundError, match="any index dates"):
            _window_of(index, "2030-01-01", None)

    def test_indices_that_never_overlap_are_refused(self):
        from beacon.server.views import _common_window

        first = pd.Series([1.0], index=pd.to_datetime(["2020-01-01"]))
        second = pd.Series([1.0], index=pd.to_datetime(["2024-01-01"]))

        with pytest.raises(DataNotFoundError, match="window these indices share"):
            _common_window({"a": first, "b": second})

    def test_no_prices_for_any_constituent_is_refused(self,
                                                      client):
        """The index is known but the data source cannot price it."""
        from beacon.data.base import MarketData
        from beacon.data.fetcher import DataFetcher
        from beacon.server.views import build_attribution

        empty = DataFetcher(MarketData.from_dataframe(pd.DataFrame(
            {"IDENTIFIER": ["OTHER"], "DATE": ["2024-01-01"], "CLOSE": [1.0]})))
        run = client.get(f"/beacon/{INDEX_ID}/overview", headers=auth())

        with pytest.raises(DataNotFoundError, match="prices for any constituent"):
            build_attribution(INDEX_ID,
                              client.app.state.jobs.latest_result(
                                  f"backtest:{INDEX_ID}"),
                              empty, None, None)

        assert run.status_code == 200

    def test_a_name_with_no_prices_is_refused(self,
                                              client):
        from beacon.data.base import MarketData
        from beacon.data.fetcher import DataFetcher
        from beacon.server.views import build_asset_view

        empty = DataFetcher(MarketData.from_dataframe(pd.DataFrame(
            {"IDENTIFIER": ["OTHER"], "DATE": ["2024-01-01"], "CLOSE": [1.0]})))

        with pytest.raises(DataNotFoundError, match="prices for 'AAA'"):
            build_asset_view(INDEX_ID, "AAA",
                             client.app.state.jobs.latest_result(
                                 f"backtest:{INDEX_ID}"),
                             empty)

    def test_drift_is_null_when_prices_are_unavailable(self,
                                                       client):
        """A drift that cannot be measured is reported as absent, not zero."""
        from beacon.data.base import MarketData
        from beacon.data.fetcher import DataFetcher
        from beacon.server.weights import build_weights

        empty = DataFetcher(MarketData.from_dataframe(pd.DataFrame(
            {"IDENTIFIER": ["OTHER"], "DATE": ["2024-01-01"], "CLOSE": [1.0]})))

        view = build_weights(INDEX_ID,
                             client.app.state.jobs.latest_result(
                                 f"backtest:{INDEX_ID}"),
                             "2023-08-15", empty)

        assert view.drift is None

    def test_a_zero_cost_run_reports_no_cost_drag(self):
        """Null rather than 0.0: costs of zero and costs not recorded are the
        same number and different statements, and the run that paid nothing has
        no drag to report."""
        from beacon.server.views import _cost_drag

        assert _cost_drag({"total_costs": 0.0, "initial_capital": 1_000_000.0}) is None
        assert _cost_drag({"initial_capital": 1_000_000.0}) is None

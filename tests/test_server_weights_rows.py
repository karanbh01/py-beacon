# tests/test_server_weights_rows.py
"""BN-116: per-constituent rows on the weights pane.

The pane already published a weights mapping and an aggregate drift figure.
What the table needed was a *row* — raw weight beside applied, shares, and this
name's own drift — and the interesting assertions are the ones that tie the
new per-row numbers back to the aggregates that were already there. A row set
that does not reconcile with `cap_redistributed`, or a per-name delta whose
worst case disagrees with `drift.maximum`, would render as a plausible table
full of wrong numbers.

The index here is market-cap weighted with a cap, because equal weighting hands
every name exactly 1/n and no cap above 1/n can ever bind — on an equal-weighted
index every row's raw weight trivially equals its applied one and none of this
would be under test.
"""
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "test-token-value"
INDEX_ID = "capped"
FLAT_ID = "flat"

START = "2023-01-02"
END = "2024-06-28"

# Below 1/6, so it binds on the canonical universe's larger names.
CAP = 0.20


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def definition(index_id: str,
               cap: float | None) -> dict:
    """A market-cap definition, capped or not."""
    weighting: dict = {"id": "weighting", "scheme": "MarketCapWeighted",
                       "params": {"use_free_float": True}}
    if cap is not None:
        weighting["max_weight"] = cap

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
    """A server with a capped and an uncapped market-cap index already run."""
    with tempfile.TemporaryDirectory() as storage:
        config = ServerConfig(auth_token=TOKEN,
                              data_fetcher=dataset.data_fetcher(),
                              storage_root=Path(storage))

        with TestClient(create_app(config),
                        raise_server_exceptions=False) as started:
            for index_id, cap in ((INDEX_ID, CAP), (FLAT_ID, None)):
                started.put(f"/indices/{index_id}",
                            json=definition(index_id, cap), headers=auth())
                started.post(f"/beacon/{index_id}/backtest",
                             json={"start": START, "end": END,
                                   "transaction_cost_bps": 10.0},
                             headers=auth())

            started.portal.call(started.app.state.jobs.drain)

            yield started


def weights(client: TestClient,
            index_id: str = INDEX_ID,
            **params) -> dict:
    """GET the weights pane."""
    response = client.get(f"/beacon/{index_id}/weights", params=params,
                          headers=auth())

    assert response.status_code == 200, response.text

    return response.json()


class TestRows:
    """Shape and ordering."""

    def test_there_is_a_row_per_constituent(self, client):
        payload = weights(client)

        assert len(payload["rows"]) == len(payload["weights"])
        assert ({row["identifier"] for row in payload["rows"]}
                == set(payload["weights"]))

    def test_rows_are_heaviest_first(self, client):
        """A weights table is read from the top."""
        applied = [row["weight"] for row in weights(client)["rows"]]

        assert applied == sorted(applied, reverse=True)

    def test_row_weights_match_the_mapping(self, client):
        """Both are published, so they must not be able to disagree."""
        payload = weights(client)

        for row in payload["rows"]:
            assert row["weight"] == pytest.approx(
                payload["weights"][row["identifier"]])

    def test_the_mapping_is_still_published(self, client):
        """Charts and concentration maths want a mapping; only the table wants
        rows. Replacing one with the other would have broken the pane."""
        assert weights(client)["weights"]


class TestRawWeights:
    """What the cap did, per name."""

    def test_capped_names_are_flagged(self, client):
        payload = weights(client)
        flagged = {row["identifier"] for row in payload["rows"] if row["capped"]}

        assert flagged == set(payload["capped"])
        assert flagged, "the cap never bound, so this file tests nothing"

    def test_at_least_one_capped_name_wanted_more_than_it_got(self, client):
        """The first round's breaches: names whose own weight exceeded the cap
        before anything was redistributed."""
        capped = [row for row in weights(client)["rows"] if row["capped"]]

        assert any(row["raw_weight"] > row["weight"] for row in capped)

    def test_a_capped_row_need_not_have_wanted_more(self, client):
        """Capping is iterative, and this is the consequence a client must not
        assume away. Weight pushed off the first round's breaches can carry a
        name that was *under* the cap over it, and the second round caps it
        too — so its raw weight, which is the figure before *any* capping, sits
        below the cap it ends up at.

        Rendering "raw > applied" as an invariant for capped rows would
        therefore be wrong. What holds is that a capped row ends at the cap,
        and that the set reconciles in aggregate."""
        payload = weights(client)
        capped = [row for row in payload["rows"] if row["capped"]]

        assert all(row["weight"] == pytest.approx(payload["cap"], abs=1e-9)
                   for row in capped)

    def test_an_uncapped_name_got_more_than_it_asked_for(self, client):
        """The other half of redistribution: weight taken off the capped names
        has to land somewhere, and it lands here."""
        rows = [row for row in weights(client)["rows"] if not row["capped"]]

        assert rows
        assert all(row["weight"] >= row["raw_weight"] for row in rows)

    def test_capped_weights_sit_at_the_cap(self, client):
        payload = weights(client)

        for row in payload["rows"]:
            if row["capped"]:
                assert row["weight"] == pytest.approx(payload["cap"], abs=1e-9)

    def test_both_weight_sets_sum_to_one(self, client):
        rows = weights(client)["rows"]

        assert sum(row["weight"] for row in rows) == pytest.approx(1.0, abs=1e-9)
        assert sum(row["raw_weight"] for row in rows) == pytest.approx(1.0, abs=1e-9)

    def test_the_excess_reconciles_with_cap_redistributed(self, client):
        """The acceptance criterion: the weight the capped names gave up is
        exactly what the pane reports as redistributed. If these drift apart,
        the table and the summary above it describe different indices."""
        payload = weights(client)

        given_up = sum(row["raw_weight"] - row["weight"]
                       for row in payload["rows"] if row["capped"])

        assert payload["cap_redistributed"] > 0.0, "nothing moved, so this is vacuous"
        assert given_up == pytest.approx(payload["cap_redistributed"], abs=1e-9)

    def test_an_uncapped_index_has_raw_equal_to_applied(self, client):
        """Nothing was capped, so there is no counterfactual to report and the
        two columns must agree exactly rather than approximately."""
        payload = weights(client, FLAT_ID)

        assert not payload["capped"]
        for row in payload["rows"]:
            assert row["raw_weight"] == pytest.approx(row["weight"], abs=1e-12)


class TestSharesOutstanding:
    """The company's share count, not the index's holding."""

    def test_it_is_populated_from_market_data(self, client):
        rows = weights(client)["rows"]

        assert all(row["shares_outstanding"] > 0 for row in rows)

    def test_it_is_the_figure_the_fetcher_holds(self, client):
        """Pinned against the data layer rather than against itself, so a row
        that silently reported the wrong name's shares would fail."""
        payload = weights(client)
        fetcher = dataset.data_fetcher()

        for row in payload["rows"]:
            expected = fetcher.fetch_shares_outstanding(row["identifier"],
                                                        payload["as_of"])
            assert row["shares_outstanding"] == pytest.approx(expected)

    def test_the_field_is_not_called_shares(self, client):
        """An index fact sheet's "shares" column means shares held per index
        unit — a different figure needing a divisor this endpoint has not got.
        The name is what stops the two being confused silently."""
        row = weights(client)["rows"][0]

        assert "shares_outstanding" in row
        assert "shares" not in row


class TestPerNameDrift:
    """`delta_since_rebalance`, and its agreement with the aggregate."""

    def test_it_is_null_at_the_rebalance_itself(self, client):
        """The weights were just set, so nothing has drifted; a zero would
        claim a measurement rather than its absence."""
        payload = weights(client)
        as_of = payload["rebalance_date"]

        at_rebalance = weights(client, asof=as_of)

        assert at_rebalance["drift"] is None
        assert all(row["delta_since_rebalance"] is None
                   for row in at_rebalance["rows"])

    def test_it_is_populated_after_the_rebalance(self, client):
        payload = weights(client, asof=END)

        assert payload["drift"] is not None
        assert all(row["delta_since_rebalance"] is not None
                   for row in payload["rows"])

    def test_the_deltas_cancel(self, client):
        """Held weights renormalise to 1, so what one name gained another lost.
        A non-zero sum would mean the drifted vector was not a weight vector."""
        rows = weights(client, asof=END)["rows"]

        assert sum(row["delta_since_rebalance"]
                   for row in rows) == pytest.approx(0.0, abs=1e-9)

    def test_the_worst_row_is_the_one_the_aggregate_names(self, client):
        """Both come from one held-weight walk. Computing them separately is
        how a total stops matching the rows it is a total of."""
        payload = weights(client, asof=END)

        worst = max(payload["rows"],
                    key=lambda row: abs(row["delta_since_rebalance"]))

        assert worst["identifier"] == payload["drift"]["worst"]
        assert abs(worst["delta_since_rebalance"]) == pytest.approx(
            payload["drift"]["maximum"], abs=1e-9)

    def test_the_absolute_deltas_sum_to_the_aggregate(self, client):
        payload = weights(client, asof=END)

        total = sum(abs(row["delta_since_rebalance"]) for row in payload["rows"])

        assert total == pytest.approx(payload["drift"]["total_absolute"],
                                      abs=1e-9)


class TestAssetDrilldown:
    """Raw alongside applied, per rebalance."""

    def _asset(self, client, identifier):
        response = client.get(f"/beacon/{INDEX_ID}/assets/{identifier}",
                              headers=auth())

        assert response.status_code == 200, response.text

        return response.json()

    def _a_capped_name(self, client) -> str:
        payload = weights(client)

        return payload["capped"][0]

    def test_both_histories_cover_the_same_rebalances(self, client):
        """Keyed off the applied history, so a drilldown can plot one against
        the other without aligning anything."""
        payload = self._asset(client, self._a_capped_name(client))

        assert set(payload["raw_weight_history"]) == set(payload["weight_history"])

    def test_a_capped_name_shows_the_cap_biting(self, client):
        identifier = self._a_capped_name(client)
        payload = self._asset(client, identifier)

        larger = [date for date, raw in payload["raw_weight_history"].items()
                  if raw > payload["weight_history"][date] + 1e-9]

        assert larger, f"{identifier} is capped but its raw history never exceeds"

    def test_the_applied_history_is_unchanged(self, client):
        """Added alongside rather than replacing, so a client reading only the
        applied series keeps working."""
        payload = self._asset(client, self._a_capped_name(client))

        assert payload["weight_history"]
        assert payload["rebalances_held"] == len(payload["weight_history"])

    def test_an_uncapped_index_has_matching_histories(self, client):
        identifier = weights(client, FLAT_ID)["rows"][0]["identifier"]
        response = client.get(f"/beacon/{FLAT_ID}/assets/{identifier}",
                              headers=auth())
        payload = response.json()

        assert payload["raw_weight_history"] == pytest.approx(
            payload["weight_history"])

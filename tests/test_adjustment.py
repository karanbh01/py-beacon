# tests/test_adjustment.py
"""BN-146: `ADJ_CLOSE`.

Splits *and* dividends, matching the vendor convention — the owner's call, so
that the name means what people reading it expect. The consequence is that an
adjusted series is no longer a price, which is what most of these tests pin.
"""
import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.adjustment import adjust_closes
from beacon.server import ServerConfig, create_app
from beacon.synthetic import SyntheticConfig, generate

TOKEN = "adj-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
DATES = pd.date_range("2024-01-01", periods=6, freq="D")


def closes(*values) -> pd.Series:
    return pd.Series([float(value) for value in values],
                     index=DATES[:len(values)])


def action(index: int,
           action_type: str,
           value: float) -> pd.DataFrame:
    return pd.DataFrame([{"EX_DATE": DATES[index], "TYPE": action_type,
                          "VALUE": value}])


class TestSplits:
    """The step a chart should not have."""

    def test_a_two_for_one_becomes_continuous(self):
        adjusted = adjust_closes(closes(100, 100, 100, 50, 50, 50),
                                 action(3, "SPLIT", 2.0))

        assert adjusted.tolist() == [50.0] * 6

    def test_the_last_value_is_the_raw_close(self):
        """Adjusted backwards, so the right-hand edge of a chart matches the
        quote a user can see elsewhere and the series stays checkable."""
        raw = closes(100, 100, 100, 50, 50, 50)
        adjusted = adjust_closes(raw, action(3, "SPLIT", 2.0))

        assert adjusted.iloc[-1] == raw.iloc[-1]

    def test_the_ex_date_itself_is_not_adjusted(self):
        """The ex-date price has already dropped; adjusting it again would
        halve it twice."""
        adjusted = adjust_closes(closes(100, 100, 100, 50, 50, 50),
                                 action(3, "SPLIT", 2.0))

        assert adjusted.iloc[3] == 50.0

    def test_two_splits_compound(self):
        """Two 2-for-1s give 4, not 4 in any additive sense."""
        raw = closes(400, 200, 200, 100, 100, 100)
        actions = pd.concat([action(1, "SPLIT", 2.0),
                             action(3, "SPLIT", 2.0)], ignore_index=True)

        assert adjust_closes(raw, actions).tolist() == [100.0] * 6

    def test_a_reverse_split_scales_up(self):
        adjusted = adjust_closes(closes(10, 10, 100, 100),
                                 action(2, "REVERSE_SPLIT", 0.1))

        assert adjusted.tolist() == [100.0] * 4

    def test_a_non_positive_ratio_is_ignored(self):
        """It would erase or invert the series."""
        logging.disable(logging.ERROR)

        try:
            adjusted = adjust_closes(closes(100, 100), action(1, "SPLIT", 0.0))
        finally:
            logging.disable(logging.NOTSET)

        assert adjusted.tolist() == [100.0, 100.0]


class TestDividends:
    """What makes this a total-return series rather than a price."""

    def test_a_dividend_flattens_a_price_drop(self):
        """A holder who was paid 10 out of a 100 stock is flat, so the
        adjusted series is flat."""
        adjusted = adjust_closes(closes(100, 100, 90, 90),
                                 action(2, "DIVIDEND", 10.0))

        assert adjusted.tolist() == [90.0] * 4

    def test_it_is_a_fraction_of_the_preceding_close(self):
        """Paying 1.0 out of a 10 stock is a different event from paying it
        out of a 100 stock, so the factor cannot be an absolute amount."""
        small = adjust_closes(closes(10, 10, 10), action(2, "DIVIDEND", 1.0))
        large = adjust_closes(closes(100, 100, 100),
                              action(2, "DIVIDEND", 1.0))

        assert small.iloc[0] / 10 < large.iloc[0] / 100

    def test_the_close_before_the_ex_date_is_used(self):
        """Not the ex-date close, which has already fallen by the dividend —
        dividing by it would overstate the adjustment."""
        adjusted = adjust_closes(closes(100, 100, 90),
                                 action(2, "DIVIDEND", 10.0))

        assert adjusted.iloc[0] == pytest.approx(90.0)

    def test_a_special_dividend_counts(self):
        adjusted = adjust_closes(closes(100, 100, 90, 90),
                                 action(2, "SPECIAL_DIVIDEND", 10.0))

        assert adjusted.iloc[0] < 100

    def test_a_dividend_with_no_preceding_close_is_skipped(self):
        """A guessed factor silently misstates every earlier value."""
        adjusted = adjust_closes(closes(100, 100), action(0, "DIVIDEND", 5.0))

        assert adjusted.tolist() == [100.0, 100.0]

    def test_a_dividend_exceeding_the_price_is_skipped(self):
        logging.disable(logging.ERROR)

        try:
            adjusted = adjust_closes(closes(10, 10, 10),
                                     action(2, "DIVIDEND", 50.0))
        finally:
            logging.disable(logging.NOTSET)

        assert adjusted.iloc[0] == 10.0

    def test_splits_and_dividends_compose(self):
        actions = pd.concat([action(2, "DIVIDEND", 10.0),
                             action(4, "SPLIT", 2.0)], ignore_index=True)
        adjusted = adjust_closes(closes(100, 100, 90, 90, 45, 45), actions)

        assert adjusted.iloc[-1] == 45.0
        assert adjusted.iloc[0] == pytest.approx(45.0)


class TestWhatIsNotAdjusted:
    def test_a_structural_action_is_left_alone(self):
        """A merger is not a scaling of the same instrument's price, so no
        factor makes the series continuous. Approximating one would invent a
        number."""
        adjusted = adjust_closes(closes(100, 100, 50),
                                 action(2, "MERGER", 1.0))

        assert adjusted.tolist() == [100.0, 100.0, 50.0]

    def test_no_actions_leaves_the_series_alone(self):
        raw = closes(100, 101, 102)

        assert adjust_closes(raw, pd.DataFrame()).tolist() == raw.tolist()

    def test_an_empty_series_is_returned_as_is(self):
        assert adjust_closes(pd.Series(dtype=float), pd.DataFrame()).empty

    def test_an_action_after_the_series_ends_changes_nothing(self):
        raw = closes(100, 100)
        late = pd.DataFrame([{"EX_DATE": pd.Timestamp("2030-01-01"),
                              "TYPE": "SPLIT", "VALUE": 2.0}])

        assert adjust_closes(raw, late).tolist() == raw.tolist()


@pytest.fixture(scope="module")
def panel():
    logging.disable(logging.ERROR)

    try:
        return generate(SyntheticConfig(assets=12, start="2022-01-03",
                                        end="2024-12-31", seed=1))
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture
def client(panel):
    return TestClient(create_app(ServerConfig(
        auth_token=TOKEN, data_fetcher=panel.fetcher(),
        storage_root=Path(tempfile.mkdtemp()))))


class TestThroughTheApi:
    def frame(self,
              client,
              name: str,
              **params):
        return client.get(f"/data/prices/{name}", headers=HEADERS,
                          params=params).json()["prices"]

    def test_the_column_appears_only_when_asked_for(self,
                                                    client):
        assert "ADJ_CLOSE" not in self.frame(client, "CMPC")["columns"]
        assert "ADJ_CLOSE" in self.frame(client, "CMPC",
                                         adjusted="true")["columns"]

    def test_the_last_adjusted_value_is_the_last_close(self,
                                                       client):
        frame = self.frame(client, "CMPC", adjusted="true")
        row = frame["data"][-1]

        assert row[frame["columns"].index("ADJ_CLOSE")] == pytest.approx(
            row[frame["columns"].index("CLOSE")])

    def test_a_split_is_removed_across_the_series(self,
                                                  client):
        """CMPC has one 2-for-1 and no dividends, so its earliest adjusted
        close is exactly half its raw one."""
        frame = self.frame(client, "CMPC", adjusted="true")
        first = frame["data"][0]

        assert first[frame["columns"].index("ADJ_CLOSE")] == pytest.approx(
            first[frame["columns"].index("CLOSE")] / 2)

    def test_dividends_move_it_too(self,
                                   panel):
        """The owner's decision: `ADJ_CLOSE` is splits *and* dividends. A
        splits-only series under this name would be a plausible wrong number.
        CMPA pays two dividends and never splits, so any difference from raw
        is the cash.
        """
        actions = panel.actions.get("CMPA")
        raw = panel.market.data.loc["CMPA"]["CLOSE"]

        assert set(actions["TYPE"]) == {"DIVIDEND"}
        assert adjust_closes(raw, actions).iloc[0] < raw.iloc[0]

    def test_asking_for_adjusted_without_close_is_refused(self,
                                                          client):
        """Rather than quietly returning neither, which would look like the
        instrument has no adjusted history."""
        response = client.get("/data/prices/CMPC", headers=HEADERS,
                              params={"adjusted": "true", "columns": "VOLUME"})

        assert response.status_code == 404

    def test_it_is_in_the_spec(self,
                               client):
        spec = client.get("/openapi.json").json()
        params = spec["paths"]["/data/prices/{identifier}"]["get"]["parameters"]

        assert "adjusted" in {entry["name"] for entry in params}

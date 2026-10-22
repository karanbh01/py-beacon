# tests/test_backtest_progress.py
"""BN-342: a backtest reports its progress through the simulation, by date.

Between its start and assembling its results a backtest job used to report
nothing, so a twenty-minute run and a hung one looked the same.
"""
import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from test_server_backtest import TOKEN, auth, build_fetcher, index_document


@pytest.fixture
def client(tmp_path):
    config = ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher(), storage_root=tmp_path)

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        assert entered.post("/indices", json=index_document(), headers=auth()).status_code == 200
        assert entered.put("/strategies/BT", json={"id": "BT", "name": "BT", "index_id": "BT"},
                           headers=auth()).status_code == 200
        yield entered


def run(client,
        monkeypatch) -> list[dict]:
    """Backtest BT and return every job event it published."""
    events: list[dict] = []
    registry = client.app.state.jobs
    publish = registry.publish
    monkeypatch.setattr(registry, "publish", lambda event: events.append(event) or publish(event))

    client.post("/strategies/BT/backtest", json={}, headers=auth())
    client.portal.call(registry.drain)

    return [event for event in events if event.get("type") == "job"]


class TestTheLibrary:

    def test_reports_each_stage_then_the_days(self):
        from beacon.backtest.main import Backtest
        from beacon.server.definitions import build_definition
        from beacon.server.schemas import IndexDocument

        seen: list[tuple[float, str]] = []
        definition = build_definition(IndexDocument.model_validate(index_document()), None)
        Backtest(initial_capital=1e6, data_provider=build_fetcher(), cache=False).run(
            definition, end="2024-03-29",
            progress=lambda fraction, message: seen.append((fraction, message)))
        fractions = [fraction for fraction, _ in seen]

        assert seen[0][1].startswith("Calculating the index")
        assert fractions == sorted(fractions) and fractions[-1] == 1.0
        # The last session: 29 March 2024 was Good Friday.
        assert seen[-1][1] == "Simulating 2024-03-28, day 124 of 124"


class TestTheJob:

    def test_publishes_the_simulation_by_date(self,
                                              client,
                                              monkeypatch):
        events = run(client, monkeypatch)
        simulating = [event for event in events if event["message"].startswith("Simulating ")]

        assert len(simulating) > 10
        assert all(0.05 < event["progress"] <= 0.8 for event in simulating)
        assert [event["progress"] for event in simulating] == sorted(event["progress"]
                                                                    for event in simulating)
        assert events[-1]["status"] == "succeeded"

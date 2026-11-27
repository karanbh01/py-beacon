# tests/test_server_analytics_cache.py
"""BN-481: the last stored run's analytics, kept between requests, so
Drilldown opening one name after another reuses the run's loaded tables and
its whole-run work."""
from pathlib import Path

import pytest

from beacon.server import analytics_cache
from beacon.server.analytics_cache import AnalyticsCache, run_stamp
from beacon.server.jobs import SUCCEEDED
from test_server_analysis import ANALYSIS, client  # noqa: F401
from test_server_strategy_runs import auth, finished


@pytest.fixture
def built(monkeypatch):
    """Each analytics the cache builds, in order."""
    found = []
    original = analytics_cache.stored_analytics

    def counted(folder,
                source):
        found.append(original(folder, source))

        return found[-1]

    monkeypatch.setattr(analytics_cache, "stored_analytics", counted)

    return found


class TestTheCache:

    @pytest.fixture
    def folder(self,
               tmp_path,
               monkeypatch):
        monkeypatch.setattr(analytics_cache, "stored_analytics",
                            lambda folder, source: object())
        (tmp_path / "run.json").write_text("{}", encoding="utf-8")

        return tmp_path

    def test_the_same_run_on_the_same_data_is_kept(self,
                                                   folder):
        cache = AnalyticsCache()

        with cache.using(folder, None, "v1") as first:
            pass

        with cache.using(folder, None, "v1") as second:
            assert second is first

    def test_the_run_written_again_is_built_again(self,
                                                  folder):
        cache = AnalyticsCache()

        with cache.using(folder, None, "v1") as first:
            pass

        (folder / "analytics.json").write_text("{}", encoding="utf-8")

        with cache.using(folder, None, "v1") as second:
            assert second is not first

    def test_other_data_is_built_again(self,
                                       folder):
        cache = AnalyticsCache()

        with cache.using(folder, None, "v1") as first:
            pass

        with cache.using(folder, None, "v2") as second:
            assert second is not first

    def test_one_run_is_kept_at_a_time(self,
                                       folder,
                                       tmp_path_factory):
        cache = AnalyticsCache()
        other = tmp_path_factory.mktemp("other")

        with cache.using(folder, None, "v1") as first:
            pass

        with cache.using(other, None, "v1"):
            pass

        with cache.using(folder, None, "v1") as again:
            assert again is not first

    def test_the_stamp_names_every_file_with_its_time_and_size(self,
                                                               folder):
        (folder / "risk").mkdir()
        (folder / "risk" / "risk.json").write_text("{}", encoding="utf-8")

        assert {Path(path).parts for path, _, _ in run_stamp(folder)} == {
            ("run.json",), ("risk", "risk.json")}


class TestThroughTheServer:

    def test_drilldown_s_requests_share_one_run_s_analytics(self,
                                                            client,  # noqa: F811
                                                            built):
        client.app.state.analytics_cache.clear()
        names = client.get("/strategies/analysed/instrument", headers=auth()).json()["names"]

        for name in names[:2]:
            assert client.get(f"/strategies/analysed/instrument/{name}",
                              headers=auth()).status_code == 200

        assert client.get(f"/strategies/analysed/assets/{names[0]}",
                          headers=auth()).status_code == 200
        assert len(built) == 1

    def test_another_strategy_s_run_takes_its_place(self,
                                                    client,  # noqa: F811
                                                    built):
        client.app.state.analytics_cache.clear()

        for strategy_id in ("analysed", "plain", "analysed"):
            assert client.get(f"/strategies/{strategy_id}/instrument",
                              headers=auth()).status_code == 200

        assert len(built) == 3

    def test_an_analysis_rerun_is_read_afresh(self,
                                              client,  # noqa: F811
                                              built):
        client.app.state.analytics_cache.clear()
        assert client.get("/strategies/analysed/instrument", headers=auth()).status_code == 200

        added = {**ANALYSIS, "add": {"factsheet": ["sortino_ratio", "calmar_ratio"]}}
        job = finished(client, client.post("/strategies/analysed/analytics",
                                           json={"analysis": added}, headers=auth()))

        assert job["status"] == SUCCEEDED, job
        assert client.get("/strategies/analysed/instrument", headers=auth()).status_code == 200
        assert len(built) == 2

# tests/test_session_complete.py
"""BN-347: a panel built from a whole session answers for every name on it.

A name with no row that day has no price, which is what the data's own read
says too, so a whole-session panel answers None for it rather than sending
the caller back to the data. On a session the data does not trade at all,
such as a London session that New York data has no rows for, every holding
used to go back to the data, one slow read each.
"""
import pandas as pd

from beacon.data.base import MarketData
from beacon.data.fetcher import DataFetcher
from beacon.data.session import SessionPanel

DAYS = pd.bdate_range("2024-01-02", periods=5)


def fetcher() -> DataFetcher:
    rows = [{"IDENTIFIER": name, "DATE": day, "CLOSE": 10.0 + i}
            for i, day in enumerate(DAYS) for name in ("AAA", "BBB")
            if not (name == "BBB" and day == DAYS[2])]

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)))


class TestAWholeSessionPanel:

    def test_answers_for_a_name_without_a_row(self):
        source = fetcher()
        source.warm_session(["AAA"], DAYS[2])
        panel = source._session_panel
        stamp = f"{DAYS[2]:%Y-%m-%d}"

        assert panel is not None and panel.complete
        assert panel.answers("BBB", stamp) and panel.value("BBB", "CLOSE") is None
        assert not panel.answers("AAA", f"{DAYS[3]:%Y-%m-%d}")

    def test_a_panel_of_chosen_names_still_does_not(self):
        frame = pd.DataFrame({"CLOSE": [1.0]}, index=pd.Index(["AAA"], name="IDENTIFIER"))
        panel = SessionPanel(DAYS[0], frame)

        assert not panel.complete
        assert not panel.answers("BBB", f"{DAYS[0]:%Y-%m-%d}")


class TestPricesOn:

    def test_gives_the_datas_answers_without_reading_name_by_name(self,
                                                                   monkeypatch):
        source = fetcher()
        stamp = f"{DAYS[2]:%Y-%m-%d}"
        expected = {name: source.fetch_price(name, stamp) for name in ("AAA", "BBB", "ZZZ")}
        source._session_panel = None
        reads = []
        single = source.fetch_price
        monkeypatch.setattr(source, "fetch_price", lambda *args, **kwargs: reads.append(args)
                            or single(*args, **kwargs))

        assert source.prices_on(["AAA", "BBB", "ZZZ"], stamp) == expected
        assert reads == []

    def test_on_a_date_the_data_does_not_trade(self):
        source = fetcher()

        assert source.prices_on(["AAA", "BBB"], "2024-01-13") == {"AAA": None, "BBB": None}


class TestALazyPanel:
    """BN-348: a whole-session panel builds a column only when it is read."""

    def test_its_columns_are_the_session_columns(self):
        source = fetcher()
        source.warm_session(["AAA"], DAYS[1])
        panel = source._session_panel
        values, _ = source._market.session_columns(DAYS[1])

        assert panel is not None and panel._values == {}
        assert panel.column("CLOSE") == values["CLOSE"]
        assert list(panel._values) == ["CLOSE"]


class TestMemoisedAnswers:
    """BN-348: a resolved session is answered once per date, and a merge
    starts afresh."""

    def test_a_merge_clears_them(self):
        source = fetcher()
        late = DAYS[-1] + pd.Timedelta(days=3)

        assert source.resolve_session(late) is None

        source.merge_market_data(pd.DataFrame({"IDENTIFIER": ["AAA"], "DATE": [late],
                                               "CLOSE": [20.0]}))

        assert source.resolve_session(late) == late

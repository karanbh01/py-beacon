# tests/test_rebalance_carry.py
"""BN-336: a rebalance carries a missing close as a day between rebalances
does, within the same staleness limit.

Before it, a name the weighting kept with no bar on a rebalance day failed the
whole index calculation ("IndexUnits ... could not be priced"), whether the bar
was absent or, since BN-334, read as missing because it was not above zero.
Decided with Karan: the two should behave the same.
"""
import pandas as pd

from test_non_positive_prices import backtest, market

# The quarter's first rebalance inside the test window, and the day after.
REBALANCE = pd.bdate_range("2024-01-02", "2024-01-03")


def gaps_of(job: dict,
            name: str) -> list[str]:
    return [gap["date"] for gap in job["result"]["price_gaps"] if gap["asset_id"] == name]


class TestARebalanceDayWithoutABar:

    def test_a_bad_close_is_carried_and_recorded(self,
                                                 tmp_path):
        frame = market(None)
        frame.loc[(frame["IDENTIFIER"] == "BBB") & frame["DATE"].isin(REBALANCE), "CLOSE"] *= -1

        job = backtest(tmp_path, frame)

        assert job["status"] == "succeeded", job["error"]
        assert gaps_of(job, "BBB") == [f"{day:%Y-%m-%d}" for day in REBALANCE]

    def test_an_absent_bar_is_carried_too(self,
                                          tmp_path):
        frame = market(None)
        frame = frame[~((frame["IDENTIFIER"] == "BBB") & frame["DATE"].isin(REBALANCE))]

        job = backtest(tmp_path, frame)

        assert job["status"] == "succeeded", job["error"]
        assert gaps_of(job, "BBB") == [f"{day:%Y-%m-%d}" for day in REBALANCE]

    def test_the_rebalance_holds_it_at_its_weight(self,
                                                  tmp_path):
        frame = market(None)
        frame = frame[~((frame["IDENTIFIER"] == "BBB") & frame["DATE"].isin(REBALANCE))]

        job = backtest(tmp_path, frame)
        rebalance = next(snapshot for snapshot in job["result"]["rebalances"]
                         if snapshot["date"] == f"{REBALANCE[0]:%Y-%m-%d}")

        assert rebalance["weights"]["BBB"] > 0.0


class TestTheStalenessLimit:

    def test_drops_a_name_missing_past_it(self,
                                          tmp_path):
        """Missing for a fortnight before the rebalance, with a five-day limit:
        the stale-price step drops it rather than carrying it."""
        missing = pd.bdate_range("2023-12-15", "2024-01-03")
        frame = market(None)
        frame = frame[~((frame["IDENTIFIER"] == "BBB") & frame["DATE"].isin(missing))]

        job = backtest(tmp_path, frame,
                       body={"modelling_assumptions": {"max_price_staleness_days": 5}})
        rebalance = next(snapshot for snapshot in job["result"]["rebalances"]
                         if snapshot["date"] == f"{REBALANCE[0]:%Y-%m-%d}")

        assert job["status"] == "succeeded", job["error"]
        assert "BBB" not in rebalance["weights"]

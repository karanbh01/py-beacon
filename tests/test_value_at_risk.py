# tests/test_value_at_risk.py
"""BN-388: value at risk, expected shortfall and stress windows, worked out
by hand on known distributions.

The hand-made returns are 100 evenly spaced from -5.0% to 4.9%, so their
quantiles can be read off; the sampled ones are 20,000 draws from a normal
distribution, which every method must agree with.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance.crises import Episode
from beacon.analysis.risk import var
from beacon.analysis.risk.var import RiskOfLoss, expected_shortfall, risk_of_loss, value_at_risk

EVEN = pd.Series(np.arange(-50, 50) / 1000.0)
NORMAL = pd.Series(np.random.default_rng(5).normal(0.001, 0.01, 20_000))
Z95 = NormalDist().inv_cdf(0.05)


class TestByHand:

    def test_historical_value_at_risk_interpolates_the_quantile(self):
        # The 5% quantile of 100 sits 4.95 places in: -0.046 + 0.95 x 0.001.
        assert value_at_risk(EVEN, 0.95) == pytest.approx(0.04505)

    def test_historical_shortfall_is_the_mean_beyond_it(self):
        assert expected_shortfall(EVEN, 0.95) == pytest.approx(0.048)

    def test_normal_value_at_risk_is_the_formula(self):
        mean, spread = EVEN.mean(), EVEN.std()

        assert value_at_risk(EVEN, 0.95, var.NORMAL) == pytest.approx(-(mean + Z95 * spread))

    def test_normal_shortfall_is_the_formula(self):
        mean, spread = EVEN.mean(), EVEN.std()
        expected = -mean + spread * NormalDist().pdf(Z95) / 0.05

        assert expected_shortfall(EVEN, 0.95, var.NORMAL) == pytest.approx(expected)

    def test_ten_days_is_root_ten_times_one(self):
        for method in var.VAR_METHODS:
            assert value_at_risk(EVEN, 0.99, method, horizon=10) == pytest.approx(
                math.sqrt(10) * value_at_risk(EVEN, 0.99, method))


class TestOnANormalDistribution:

    @pytest.mark.parametrize("method", var.VAR_METHODS)
    def test_every_method_finds_the_normal_s_value_at_risk(self,
                                                           method):
        truth = -(0.001 + Z95 * 0.01)

        assert value_at_risk(NORMAL, 0.95, method) == pytest.approx(truth, rel=0.03)

    @pytest.mark.parametrize("method", var.ES_METHODS)
    def test_and_its_shortfall(self,
                               method):
        truth = -0.001 + 0.01 * NormalDist().pdf(Z95) / 0.05

        assert expected_shortfall(NORMAL, 0.95, method) == pytest.approx(truth, rel=0.03)


class TestCornishFisher:

    def test_a_fat_left_tail_brings_it_to_the_historical(self):
        """Normal days with a 3% fall one day in fifty: a skew of about -0.45
        and an excess kurtosis of about 1, where the expansion holds."""
        random = np.random.default_rng(5)
        crashes = pd.Series(random.normal(0.0005, 0.01, 20_000)
                            + np.where(random.random(20_000) < 0.02, -0.03, 0.0))
        historical = value_at_risk(crashes, 0.99)

        assert value_at_risk(crashes, 0.99, var.CORNISH_FISHER) == pytest.approx(historical,
                                                                                rel=0.05)
        assert value_at_risk(crashes, 0.99, var.NORMAL) < 0.85 * historical

    def test_it_moves_z_by_the_skew_and_kurtosis(self):
        skew, kurtosis = float(EVEN.skew()), float(EVEN.kurt())
        z = Z95
        moved = (z + (z ** 2 - 1) * skew / 6 + (z ** 3 - 3 * z) * kurtosis / 24
                 - (2 * z ** 3 - 5 * z) * skew ** 2 / 36)

        assert value_at_risk(EVEN, 0.95, var.CORNISH_FISHER) == pytest.approx(
            -(EVEN.mean() + moved * EVEN.std()))


class TestTheTable:

    def test_every_measure_method_level_and_horizon(self):
        table = risk_of_loss(EVEN)

        assert list(table.index) == [("var", "historical"), ("var", "normal"),
                                     ("var", "cornish_fisher"), ("es", "historical"),
                                     ("es", "normal")]
        assert list(table.columns) == [(0.95, 1), (0.95, 10), (0.99, 1), (0.99, 10)]
        assert table.loc[("var", "historical"), (0.95, 1)] == pytest.approx(0.04505)

    def test_too_few_returns_have_none(self):
        assert value_at_risk(EVEN.iloc[:10]) is None

    @pytest.mark.parametrize(("arguments", "message"), [
        ({"method": "monte_carlo"}, "monte_carlo"), ({"level": 95}, "level")])
    def test_settings_out_of_range(self,
                                   arguments,
                                   message):
        with pytest.raises(ValueError, match=message):
            value_at_risk(EVEN, **arguments)

    def test_shortfall_has_no_cornish_fisher(self):
        with pytest.raises(ValueError, match="cornish_fisher"):
            expected_shortfall(EVEN, method=var.CORNISH_FISHER)


class TestARun:

    @pytest.fixture(scope="class")
    def run(self):
        from test_strategy_futures import index, run

        return RunData.from_result(run(index()))

    def test_each_book(self,
                       run):
        risk = RiskOfLoss(run)

        assert risk.books() == ["fund", "index", "active"]
        assert risk.table("index").loc[("var", "historical"), (0.95, 1)] == pytest.approx(
            value_at_risk(risk.returns_of("index"), 0.95))
        assert risk.table("active").loc[("var", "normal"), (0.99, 1)] < risk.table().loc[
            ("var", "normal"), (0.99, 1)]

    def test_stress_windows_are_each_book_s_return_and_fall(self,
                                                           run):
        risk = RiskOfLoss(run)
        episode = Episode("spring", pd.Timestamp("2024-03-01"), pd.Timestamp("2024-05-31"))

        table = risk.stress_windows(named=[episode])
        inside = risk.returns_of()[lambda found: (found.index > episode.start)
                                   & (found.index <= episode.end)]

        assert table["crisis"].tolist() == ["spring"]
        assert table.loc[0, "fund_return"] == pytest.approx(float((1 + inside).prod() - 1))
        assert table.loc[0, "index_drawdown"] <= 0.0

    def test_an_unknown_book(self,
                             run):
        with pytest.raises(KeyError, match="benchmark"):
            RiskOfLoss(run).returns_of("benchmark")

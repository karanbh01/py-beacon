# tests/test_performance_formulas.py
"""BN-370: each performance formula equals the definition it replaces, to
the last bit; BN-405: the Sharpe ratio, the comparisons' information ratio,
correlation and beta as decided, each checked by hand.

The definitions are copied here as they stood before BN-370 (the backtest
metrics, `analysis.comparison`, `analysis.relative`, the share-class
benchmark figures and the ETF analytics), so the check outlives the code it
replaces. Inputs are generated return series of several lengths, scales and
shapes, including flat ones and ones with a single return.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.performance import formulas as f

PPY = 252


def series(seed: int,
           days: int,
           scale: float = 0.01,
           start: str = "2020-01-02") -> pd.Series:
    """Daily returns on business days."""
    random = np.random.default_rng(seed)

    return pd.Series(random.normal(0.0003, scale, days), index=pd.bdate_range(start, periods=days))


CASES = [series(seed, days, scale) for seed, days, scale in
         [(1, 600, 0.01), (2, 1, 0.01), (3, 2, 0.02), (4, 45, 0.0), (5, 300, 1e-14),
          (6, 1300, 0.03), (7, 23, 0.005)]]


def pair(seed: int) -> tuple[pd.Series, pd.Series]:
    """Two related return series over the same days."""
    theirs = series(seed, 500)
    noise = series(seed + 100, 500, 0.002)

    return theirs * 0.9 + noise, theirs


PAIRS = [pair(seed) for seed in range(5)] + [(CASES[3], CASES[3]), (CASES[1], CASES[1]),
                                             (CASES[4], CASES[4] * 2)]


def same(a, b) -> bool:
    """Equal to the last bit, or both missing."""
    if a is None or b is None:
        return a is b

    return (np.isnan(a) and np.isnan(b)) or a == b


# Today's definitions, as they stood before BN-370.

def old_headline(returns, total_return, path, periods_per_year=PPY, risk_free_rate=0.0):
    n_periods = len(returns)
    years = n_periods / float(periods_per_year)
    annualised_return = (float((1 + total_return) ** (1 / years) - 1)
                         if n_periods > 0 and years > 0 else 0.0)
    volatility = (float(returns.std() * np.sqrt(periods_per_year)) if n_periods > 1 else 0.0)
    sharpe_ratio = (float((annualised_return - risk_free_rate) / volatility)
                    if volatility > 0 else 0.0)
    max_drawdown = (float((path / path.cummax() - 1).min()) if not path.empty else 0.0)

    return {"total_return": total_return, "annualised_return": annualised_return,
            "volatility": volatility, "sharpe_ratio": sharpe_ratio, "max_drawdown": max_drawdown}


def old_metrics_hit_rate(returns):
    if returns.empty:
        return None

    index = pd.DatetimeIndex(returns.index)
    monthly = (1.0 + returns).groupby([index.year, index.month]).prod() - 1.0

    return float((monthly > 0).mean())


def old_from_start(levels):
    returns = levels.pct_change()

    if not returns.empty:
        returns.iloc[0] = 0.0

    return returns.dropna()


def old_tracking_error(port_returns, index_returns, periods_per_year=PPY):
    aligned = pd.DataFrame({"port": port_returns, "index": index_returns}).dropna()

    if aligned.empty:
        return None

    return float((aligned["port"] - aligned["index"]).std() * np.sqrt(periods_per_year))


def old_tracking_difference(port_returns, index_returns):
    return float(((1 + port_returns).prod() - 1) - ((1 + index_returns).prod() - 1))


def old_information_ratio(active, periods_per_year=PPY):
    spread = float(active.std())

    if len(active) < 2 or spread == 0.0:
        return None

    return float(active.mean() / spread * np.sqrt(periods_per_year))


def old_relative(ours, theirs, periods_per_year=PPY):
    n = len(ours)
    years = n / float(periods_per_year)
    our_total = float((1.0 + ours).prod() - 1.0)
    their_total = float((1.0 + theirs).prod() - 1.0)
    annualised = (((1.0 + our_total) ** (1.0 / years) - (1.0 + their_total) ** (1.0 / years))
                  if years > 0 else 0.0)
    difference = ours - theirs
    tracking_error = (float(difference.std() * np.sqrt(periods_per_year)) if n > 1 else None)
    variance = float(theirs.var()) if n > 1 else 0.0
    monthly_ours = old_compounded(ours)
    monthly_theirs = old_compounded(theirs)

    return {
        "relative_return": our_total - their_total,
        "annualised_relative_return": float(annualised),
        "tracking_error": tracking_error,
        "information_ratio": float(annualised / tracking_error) if tracking_error else None,
        "correlation": (float(ours.corr(theirs))
                        if n > 1 and ours.std() > 0 and theirs.std() > 0 else None),
        "beta": float(ours.cov(theirs) / variance) if variance > 0 else None,
        "hit_rate_monthly": (float((monthly_ours > monthly_theirs).mean())
                             if len(monthly_ours) else None),
    }


def old_compounded(returns):
    if returns.empty:
        return pd.Series(dtype=float)

    index = pd.DatetimeIndex(returns.index)
    keys = index.strftime("%Y-%m")

    return pd.Series({str(label): float((1.0 + period).prod() - 1.0) for label, period
                      in returns.groupby(pd.Index(keys, name="period"), sort=False)})


def old_comparison_hit_rate(returns):
    monthly = old_compounded(returns)

    return float((monthly > 0).mean()) if len(monthly) else None


def old_path(returns):
    return pd.concat([pd.Series([1.0]), (1.0 + returns.reset_index(drop=True)).cumprod()],
                     ignore_index=True)


def old_varies(returns):
    return len(returns) >= 2 and float(returns.std(ddof=1)) > 1e-12


def old_relative_correlation(a, b):
    if not old_varies(a) or not old_varies(b):
        return 0.0

    value = a.corr(b)

    return 0.0 if pd.isna(value) else float(value)


def old_relative_beta(a, b):
    if not old_varies(b):
        return 0.0

    return float(a.cov(b)) / float(b.var(ddof=1))


def old_drawdown(level):
    return level if level.empty else level / level.cummax() - 1.0


def levels_of(returns: pd.Series) -> pd.Series:
    return 100.0 * (1.0 + returns).cumprod()


class TestHeadline:

    @pytest.mark.parametrize("returns", CASES, ids=range(len(CASES)))
    @pytest.mark.parametrize("rate", [0.0, 0.02])
    def test_the_five_figures(self,
                              returns,
                              rate):
        path = old_path(returns)
        total = float(path.iloc[-1] - 1.0)
        figures = f.headline_figures(returns, total, path, PPY, rate)
        old = old_headline(returns, total, path, PPY, rate)

        # All but the Sharpe ratio, which BN-405 redefined.
        assert {k: v for k, v in figures.items() if k != "sharpe_ratio"} == {
            k: v for k, v in old.items() if k != "sharpe_ratio"}
        assert figures["sharpe_ratio"] == f.sharpe_ratio(returns, PPY, rate)

    @pytest.mark.parametrize("returns", CASES, ids=range(len(CASES)))
    def test_each_figure_on_its_own(self,
                                    returns):
        old = old_headline(returns, 0.37, old_path(returns), 12)

        assert f.annualised_return(0.37, len(returns), 12) == old["annualised_return"]
        assert f.volatility(returns, 12) == old["volatility"]
        assert f.max_drawdown(old_path(returns)) == old["max_drawdown"]

    def test_no_returns(self):
        empty = pd.Series(dtype=float)

        assert f.headline_figures(empty, 0.0, empty, PPY) == old_headline(empty, 0.0, empty)

    @pytest.mark.parametrize("returns", CASES, ids=range(len(CASES)))
    def test_the_path_and_compounding(self,
                                      returns):
        assert f.path_of(returns).equals(old_path(returns))
        assert f.compounded_return(returns) == float((1 + returns).prod() - 1)
        levels = levels_of(returns)
        assert f.total_return(float(levels.iloc[0]), float(levels.iloc[-1])) == float(
            levels.iloc[-1] / levels.iloc[0] - 1)


class TestSeries:

    @pytest.mark.parametrize("returns", CASES, ids=range(len(CASES)))
    def test_returns_from_the_first_day(self,
                                        returns):
        levels = levels_of(returns)

        assert f.returns_from_start(levels).dropna().equals(old_from_start(levels))

    @pytest.mark.parametrize("returns", CASES, ids=range(len(CASES)))
    def test_drawdown(self,
                      returns):
        assert f.drawdown(levels_of(returns)).equals(old_drawdown(levels_of(returns)))

    @pytest.mark.parametrize("returns", CASES, ids=range(len(CASES)))
    def test_hit_rates_agree_both_ways_they_were_measured(self,
                                                         returns):
        assert same(f.hit_rate_monthly(returns), old_comparison_hit_rate(returns))
        assert same(f.hit_rate_monthly(returns), old_metrics_hit_rate(returns))
        assert f.positive_share(returns) == float((returns > 0).mean())

    @pytest.mark.parametrize("frequency", f.FREQUENCIES)
    def test_periods_are_named_and_compounded(self,
                                              frequency):
        returns = CASES[5]
        periods = f.period_returns(returns, frequency)

        assert np.isclose((1 + periods).prod(), (1 + returns).prod(), rtol=1e-12)
        assert periods.index[0] == {"annual": "2020", "quarterly": "2020 Q1",
                                    "monthly": "2020-01"}[frequency]

    def test_an_unknown_frequency_is_refused(self):
        with pytest.raises(ValueError, match="weekly"):
            f.grouped(CASES[0], "weekly")


class TestRelative:

    @pytest.mark.parametrize("ours, theirs", PAIRS, ids=range(len(PAIRS)))
    def test_the_comparison_figures(self,
                                    ours,
                                    theirs):
        old = old_relative(ours, theirs)
        our_total, their_total = f.compounded_return(ours), f.compounded_return(theirs)
        annualised = f.annualised_relative_return(our_total, their_total, len(ours), PPY)
        active_risk = f.tracking_error(ours - theirs, PPY)

        assert same(our_total - their_total, old["relative_return"])
        assert same(annualised, old["annualised_relative_return"])
        assert same(active_risk, old["tracking_error"])
        assert same(f.relative_hit_rate_monthly(ours, theirs), old["hit_rate_monthly"])

    @pytest.mark.parametrize("ours, theirs", PAIRS, ids=range(len(PAIRS)))
    def test_the_backtest_s_tracking_figures(self,
                                             ours,
                                             theirs):
        active = (ours - theirs).dropna()
        old = old_tracking_error(ours, theirs)

        # One active return measures no spread: it was NaN, and is now None,
        # as every other figure that cannot be measured is.
        assert same(f.tracking_error(active, PPY), None if len(active) == 1 else old)
        assert len(active) > 1 or np.isnan(old)
        assert same(f.tracking_difference(ours, theirs), old_tracking_difference(ours, theirs))
        # BN-405: an active return that varies by no more than rounding has no
        # ratio; it was one rounding error over another (about -5e11 here).
        flat = len(active) > 1 and float(active.std()) <= f.NEGLIGIBLE_SPREAD
        expected = None if flat else old_information_ratio(active)
        assert same(f.information_ratio(active, PPY), expected)

    @pytest.mark.parametrize("ours, theirs", PAIRS, ids=range(len(PAIRS)))
    def test_correlation_and_beta_are_relative_metrics_s(self,
                                                         ours,
                                                         theirs):
        """BN-405: 0 when a series does not vary, as relative_metrics had it,
        everywhere."""
        assert same(f.correlation(ours, theirs), old_relative_correlation(ours, theirs))
        assert same(f.beta(ours, theirs), old_relative_beta(ours, theirs))

    def test_the_etf_tracking_figures(self):
        from beacon.analysis.etf.analytics import (
            calculate_tracking_difference,
            calculate_tracking_error,
        )

        for ours, theirs in PAIRS[:5]:
            assert f.tracking_difference(ours, theirs) == calculate_tracking_difference(
                ours, theirs)
            assert f.tracking_error(ours - theirs, PPY) == calculate_tracking_error(ours, theirs)


class TestOthers:

    def test_the_money_weighted_return(self):
        from beacon.backtest.flows import money_weighted_return

        days = pd.bdate_range("2020-01-02", periods=4)
        flows = [(days[0], -100.0), (days[1], -50.0), (days[2], 20.0), (days[3], 140.0)]

        assert f.money_weighted_return(flows) == money_weighted_return(flows)
        assert f.money_weighted_return([(days[0], 100.0)]) is None

    def test_the_market_return(self):
        assert f.market_return(101.0, 110.5) == float(110.5 / 101.0 - 1.0)


class TestTheDecidedDefinitions:
    """BN-405: the Sharpe ratio and the information ratio as the mean daily
    return over its spread, annualised; correlation and beta 0 when a series
    does not vary."""

    def test_the_sharpe_ratio_by_hand(self):
        returns = pd.Series([0.01, -0.02, 0.015, 0.005])
        day = 1.03 ** (1 / 252) - 1
        excess = [value - day for value in returns]
        mean = sum(excess) / 4
        spread = (sum((value - mean) ** 2 for value in excess) / 3) ** 0.5

        assert f.sharpe_ratio(returns, PPY, 0.03) == pytest.approx(mean / spread * 252 ** 0.5,
                                                                    rel=1e-12)

    def test_the_daily_rate_compounds_to_the_annual_one(self):
        assert (1 + f.daily_rate(0.03, PPY)) ** PPY == pytest.approx(1.03, rel=1e-14)

    def test_the_sharpe_ratio_does_not_change_with_leverage(self):
        """Borrowing at the risk-free rate scales the excess returns and their
        spread alike."""
        day = f.daily_rate(0.03, PPY)
        excess = CASES[5]

        ratios = [f.sharpe_ratio(day + leverage * excess, PPY, 0.03) for leverage in (1, 2, 3)]

        assert ratios == pytest.approx([ratios[0]] * 3, rel=1e-9)

    def test_returns_that_do_not_vary_have_no_sharpe_ratio(self):
        assert f.sharpe_ratio(pd.Series([0.001] * 30), PPY, 0.0) == 0.0
        assert f.sharpe_ratio(pd.Series([0.001]), PPY, 0.0) == 0.0

    def test_the_information_ratio_by_hand(self):
        active = pd.Series([0.001, -0.0005, 0.002, 0.0])
        mean = 0.0025 / 4
        spread = (sum((value - mean) ** 2 for value in active) / 3) ** 0.5

        assert f.information_ratio(active, PPY) == pytest.approx(mean / spread * 252 ** 0.5,
                                                                 rel=1e-12)

    def test_the_comparisons_information_ratio_is_the_daily_average_one(self):
        from beacon.analysis.performance.comparisons import relative_figures

        ours, theirs = PAIRS[0]

        assert relative_figures(ours, theirs, PPY)["information_ratio"] == f.information_ratio(
            ours - theirs, PPY)

    def test_a_series_that_does_not_vary_has_zero_correlation_and_beta(self):
        flat = pd.Series([0.001] * 50, index=CASES[0].index[:50])
        moving = CASES[0].iloc[:50]

        assert f.correlation(moving, flat) == 0.0
        assert f.correlation(flat, moving) == 0.0
        assert f.beta(moving, flat) == 0.0
        assert f.beta(flat, moving) == pytest.approx(0.0, abs=1e-15)
        assert f.correlation(moving.iloc[:1], moving.iloc[:1]) == 0.0

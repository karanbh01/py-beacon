# src/beacon/analysis/performance/formulas.py
"""
The formulas behind a run's performance figures: pure functions of returns
and levels, one definition each.

Returns are simple daily returns, and a level series is what they move. A
comparator's returns over a run start from its level on the first day, so
its first return is nothing (`returns_from_start`), as a run's first return
runs from its starting capital. Figures annualise over *periods_per_year*
returns a year.

A figure that cannot be measured (too few returns) is None, unless its
definition says otherwise. The ratios of a mean return to its spread, the
Sharpe and information ratios, are the mean daily return over its standard
deviation, annualised by the square root of *periods_per_year*: both halves
from the same daily returns, so the ratio does not change with leverage, and
times the square root of the years it is the t-statistic of the mean.
"""
# BN-370: every figure computed today, defined once. BN-405: one definition
# each for the figures that were measured two ways (the information ratio,
# the Sharpe ratio, correlation and beta), as decided with Karan.
import numpy as np
import pandas as pd

# A return series whose standard deviation is at most this does not vary:
# a hundred-millionth of a basis point a day. A threshold rather than zero,
# because a perfectly smooth series has a standard deviation of about 1e-17
# in floating point, and a beta measured against it is one rounding error
# over another. A ratio over such a spread is 0.
NEGLIGIBLE_SPREAD = 1e-12


# How a calendar period is named and grouped.
ANNUAL, QUARTERLY, MONTHLY = "annual", "quarterly", "monthly"
FREQUENCIES = (ANNUAL, QUARTERLY, MONTHLY)


def returns_from_start(levels: pd.Series) -> pd.Series:
    """Daily returns of a level series, the first day's being nothing: its
    returns over a run, measured from its level on the first day."""
    returns = levels.astype(float).pct_change()

    if len(returns):
        returns.iloc[0] = 0.0

    return returns


def total_return(start: float,
                 end: float) -> float:
    """The return from *start* to *end*."""
    return float(end / start - 1)


def compounded_return(returns: pd.Series) -> float:
    """The return of a run of returns, compounded."""
    return float((1 + returns).prod() - 1)


def annualised_return(total: float,
                      periods: int,
                      periods_per_year: int) -> float:
    """*total*, earned over *periods* returns, as a rate a year; 0 over no
    time."""
    years = periods / float(periods_per_year)

    return float((1 + total) ** (1 / years) - 1) if periods > 0 and years > 0 else 0.0


def volatility(returns: pd.Series,
               periods_per_year: int) -> float:
    """The annualised standard deviation of *returns*; 0 with fewer than two."""
    return float(returns.std() * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0


def daily_rate(annual_rate: float,
               periods_per_year: int) -> float:
    """The rate a period that compounds to *annual_rate* over a year."""
    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)


def sharpe_ratio(returns: pd.Series,
                 periods_per_year: int,
                 risk_free_rate: float = 0.0) -> float:
    """The mean daily return in excess of the risk-free rate, over its
    standard deviation, annualised; 0 with fewer than two returns, or when
    they do not vary (`NEGLIGIBLE_SPREAD`).

    The annual *risk_free_rate* is taken as the daily rate that compounds to
    it (`daily_rate`).
    """
    # BN-405: the mean of the daily excess returns, not the compounded growth
    # rate less the rate, which counted volatility twice (its drag lowers the
    # growth rate) and fell as a strategy was levered.
    excess = returns - daily_rate(risk_free_rate, periods_per_year)
    spread = float(excess.std()) if len(excess) > 1 else 0.0

    if spread <= NEGLIGIBLE_SPREAD:
        return 0.0

    return float(excess.mean() / spread * np.sqrt(periods_per_year))


def drawdown(levels: pd.Series) -> pd.Series:
    """How far below its running peak a level series stands each day."""
    if levels.empty:
        return levels

    return levels / levels.cummax() - 1.0


def max_drawdown(path: pd.Series) -> float:
    """The largest fall of *path* from a running peak; 0 for no path."""
    return float((path / path.cummax() - 1).min()) if not path.empty else 0.0


def path_of(returns: pd.Series) -> pd.Series:
    """The level a run of returns moves, from 1 before its first."""
    return pd.concat([pd.Series([1.0]), (1.0 + returns.reset_index(drop=True)).cumprod()],
                     ignore_index=True)


def headline_figures(returns: pd.Series,
                     total: float,
                     path: pd.Series,
                     periods_per_year: int,
                     risk_free_rate: float = 0.0) -> dict[str, float]:
    """The five headline figures of a performance.

    Args:
        returns: The daily returns.
        total: The return over the whole performance.
        path: The level the drawdown is measured along, from its start.
        periods_per_year: How many returns make a year.
        risk_free_rate: The annual rate the Sharpe ratio is in excess of.

    Returns:
        dict: total_return, annualised_return, volatility, sharpe_ratio and
        max_drawdown.
    """
    return {"total_return": total,
            "annualised_return": annualised_return(total, len(returns), periods_per_year),
            "volatility": volatility(returns, periods_per_year),
            "sharpe_ratio": sharpe_ratio(returns, periods_per_year, risk_free_rate),
            "max_drawdown": max_drawdown(path)}


def period_returns(returns: pd.Series,
                   frequency: str) -> pd.Series:
    """Returns compounded within each calendar year, quarter or month, by
    its name (``2024``, ``2024 Q1``, ``2024-01``), in order."""
    if returns.empty:
        return pd.Series(dtype=float)

    return pd.Series({label: float((1.0 + period).prod() - 1.0)
                      for label, period in grouped(returns, frequency)})


def grouped(returns: pd.Series,
            frequency: str) -> list[tuple[str, pd.Series]]:
    """*returns* split into calendar periods, each named, in order.

    Raises:
        ValueError: If the frequency is not one of `FREQUENCIES`.
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"Unknown frequency {frequency!r}. Known: {', '.join(FREQUENCIES)}.")

    index = pd.DatetimeIndex(returns.index)
    keys = {ANNUAL: index.strftime("%Y"),
            QUARTERLY: [f"{day.year} Q{(day.month - 1) // 3 + 1}" for day in index],
            MONTHLY: index.strftime("%Y-%m")}[frequency]

    return [(str(label), period) for label, period
            in returns.groupby(pd.Index(keys, name="period"), sort=False)]


def hit_rate_monthly(returns: pd.Series) -> float | None:
    """The share of calendar months whose compounded return was positive."""
    monthly = period_returns(returns, MONTHLY)

    return float((monthly > 0).mean()) if len(monthly) else None


def relative_hit_rate_monthly(ours: pd.Series,
                              theirs: pd.Series) -> float | None:
    """The share of calendar months in which *ours* compounded to more than
    *theirs*, over the same days."""
    monthly_ours = period_returns(ours, MONTHLY)
    monthly_theirs = period_returns(theirs, MONTHLY)

    return float((monthly_ours > monthly_theirs).mean()) if len(monthly_ours) else None


def positive_share(returns: pd.Series) -> float:
    """The share of *returns* above zero."""
    return float((returns > 0).mean())


def tracking_error(active: pd.Series,
                   periods_per_year: int) -> float | None:
    """The annualised standard deviation of the active returns (ours less
    theirs, day by day); None with fewer than two."""
    return float(active.std() * np.sqrt(periods_per_year)) if len(active) > 1 else None


def tracking_difference(ours: pd.Series,
                        theirs: pd.Series) -> float:
    """Our compounded return less theirs."""
    return float(((1 + ours).prod() - 1) - ((1 + theirs).prod() - 1))


def annualised_relative_return(our_total: float,
                               their_total: float,
                               periods: int,
                               periods_per_year: int) -> float:
    """Our annualised growth less theirs, each total earned over *periods*
    returns; 0 over no time."""
    years = periods / float(periods_per_year)

    return (float((1.0 + our_total) ** (1.0 / years) - (1.0 + their_total) ** (1.0 / years))
            if years > 0 else 0.0)


def information_ratio(active: pd.Series,
                      periods_per_year: int) -> float | None:
    """The mean active return over its standard deviation, annualised; None
    with fewer than two, or when they do not vary (`NEGLIGIBLE_SPREAD`), as
    there is no active risk to measure it against."""
    spread = float(active.std())

    if len(active) < 2 or not spread > NEGLIGIBLE_SPREAD:
        return None

    return float(active.mean() / spread * np.sqrt(periods_per_year))


def correlation(ours: pd.Series,
                theirs: pd.Series) -> float:
    """The correlation of two return series over the same days; 0 with
    fewer than two, or when either does not vary (`NEGLIGIBLE_SPREAD`), as
    nothing relates them."""
    if not _varies(ours) or not _varies(theirs):
        return 0.0

    value = ours.corr(theirs)

    return 0.0 if pd.isna(value) else float(value)


def beta(ours: pd.Series,
         theirs: pd.Series) -> float:
    """The sensitivity of *ours* to *theirs*: their covariance over the
    variance of *theirs*; 0 with fewer than two, or when *theirs* does not
    vary (`NEGLIGIBLE_SPREAD`), as there is nothing to be sensitive to."""
    if not _varies(theirs):
        return 0.0

    return float(ours.cov(theirs)) / float(theirs.var())


def _varies(returns: pd.Series) -> bool:
    """Whether *returns* vary by more than rounding."""
    return len(returns) > 1 and float(returns.std()) > NEGLIGIBLE_SPREAD


def market_return(first_ask: float,
                  last_bid: float) -> float:
    """An exchange investor's return: bought at *first_ask*, sold at
    *last_bid*."""
    return float(last_bid / first_ask - 1.0)


def money_weighted_return(flows: list[tuple[pd.Timestamp, float]]) -> float | None:
    """The annual rate that discounts an investor's cash flows to zero.

    Args:
        flows: (date, amount) from the investor's side: money put in is
            negative, money taken out (and the final value) positive.

    Returns:
        float or None: The internal rate of return a year, over calendar days
        (ACT/365), or None when the flows do not change sign, so no rate
        exists.
    """
    if not flows or min(amount for _, amount in flows) >= 0 \
            or max(amount for _, amount in flows) <= 0:
        return None

    start = min(date for date, _ in flows)
    years = np.array([(date - start).days / 365.0 for date, _ in flows])
    amounts = np.array([amount for _, amount in flows])

    def value(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    low, high = -0.9999, 1.0

    # Widen until the value changes sign; a rate above 1e6 a year is noise.
    while value(high) > 0 and high < 1e6:
        high *= 10.0

    if value(low) * value(high) > 0:
        return None

    for _ in range(200):
        middle = (low + high) / 2.0

        if value(low) * value(middle) <= 0:
            high = middle
        else:
            low = middle

    return (low + high) / 2.0

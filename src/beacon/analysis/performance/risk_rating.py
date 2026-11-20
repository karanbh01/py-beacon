# src/beacon/analysis/performance/risk_rating.py
"""
A risk rating: the factsheet's 1 to 7 scale on volatility, and a band on
tracking error.

    risk_rating(returns, tracked, books, settings, periods_per_year)
    risk_distribution(returns, tracked, settings, periods_per_year)

## The rating

The synthetic risk and reward indicator (SRRI) of a UCITS factsheet: the
annualised volatility of weekly returns over the last five years, or over
all of a shorter run's weeks, placed on the scale's bounds of 0.5%, 2%, 5%,
10%, 15% and 25%. Under 0.5% is a 1, from 0.5% to under 2% a 2, and so on to
25% and over, a 7: a volatility exactly on a bound takes the higher rating.
Weeks end on Friday, each its daily returns compounded, so a run's first and
last weeks may be part weeks.

## The band

The tracking error against the tracked comparator over the same weeks:
passive under 0.5%, enhanced from 0.5% to under 2%, active from 2% to under
5%, and unconstrained 5% and over. A run with no comparator has no band.

## How the rating moved

Each day's rating and band from the volatility and tracking error of the
daily returns over the year ending that day, and the days spent at each.
The run's first year has no such figures and is not counted. These are
daily, not weekly, figures, so a run's last day can sit at a different
rating from its headline one.

Both need a year of daily returns.
"""
# Agreed with Karan on 2026-11-20 for beacon-ui's Summary risk box (BU-294):
# weekly over five years as the SRRI regulation does (CESR/10-673), bounds as
# settings, too short under a year.
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from . import formulas
from .comparisons import aligned

FUND = "fund"

BANDS = ("passive", "enhanced", "active", "unconstrained")

# The SRRI's volatility bounds and the bands' tracking error bounds.
VOLATILITY_BOUNDS = (0.005, 0.02, 0.05, 0.10, 0.15, 0.25)
TRACKING_BOUNDS = (0.005, 0.02, 0.05)

# Weeks end on Friday, and a year has 52 of them.
WEEK = "W-FRI"
WEEKS_A_YEAR = 52

RATING_COLUMNS = ["book", "rating", "band", "volatility", "tracking_error"]
DISTRIBUTION_COLUMNS = ["rating", "band", "days", "share"]


@dataclass(frozen=True)
class RiskRatingSettings:
    """How the risk rating is worked out.

    Attributes:
        volatility_bounds: The annualised volatilities each rating above 1
            starts at, rising: six of them give the SRRI's seven ratings.
        tracking_bounds: The tracking errors enhanced, active and
            unconstrained start at, rising.
        years: The years of weekly returns the rating is measured over.
    """
    volatility_bounds: tuple[float, ...] = VOLATILITY_BOUNDS
    tracking_bounds: tuple[float, ...] = TRACKING_BOUNDS
    years: int = 5

    def __post_init__(self) -> None:
        _check_bounds(self.volatility_bounds, "volatility_bounds")
        _check_bounds(self.tracking_bounds, "tracking_bounds")

        if len(self.tracking_bounds) != len(BANDS) - 1:
            raise ValueError(f"tracking_bounds needs {len(BANDS) - 1} bounds, one for each "
                             f"band after passive; got {len(self.tracking_bounds)}.")

        if self.years < 1:
            raise ValueError(f"years must be at least 1; got {self.years}.")


def ratings(volatility: float | np.ndarray,
            bounds: tuple[float, ...] = VOLATILITY_BOUNDS) -> np.ndarray:
    """Each volatility's rating: 1, and one more for each bound it reaches."""
    found: np.ndarray = np.searchsorted(bounds, np.atleast_1d(volatility), side="right") + 1

    return found


def bands(tracking_error: float | np.ndarray,
          bounds: tuple[float, ...] = TRACKING_BOUNDS) -> np.ndarray:
    """Each tracking error's band."""
    found: np.ndarray = np.array(BANDS, dtype=object)[
        np.searchsorted(bounds, np.atleast_1d(tracking_error), side="right")]

    return found


def weekly(returns: pd.Series) -> pd.Series:
    """Daily returns compounded into weeks ending on Friday."""
    if returns.empty:
        return returns

    compounded = (1.0 + returns).resample(WEEK).prod(min_count=1).dropna()

    return compounded - 1.0


def risk_rating(returns: pd.Series,
                tracked: pd.Series | None,
                books: Mapping[str, pd.Series],
                settings: RiskRatingSettings,
                periods_per_year: int) -> pd.DataFrame | None:
    """The fund's rating and band, and each comparator's rating.

    Args:
        returns: The fund's daily returns.
        tracked: The levels of the comparator the band is measured against;
            None for a run without one.
        books: The levels of each comparator to rate too, by name.
        settings: The bounds and the years measured.
        periods_per_year: How many daily returns make a year.

    Returns:
        pd.DataFrame | None: A row a portfolio: ``book``, ``rating``,
        ``band``, ``volatility`` and ``tracking_error``, the last two
        annualised; a comparator's band and tracking error are blank. None
        for a run with under a year of returns.
    """
    if len(returns) < periods_per_year:
        return None

    start = returns.index[-1] - pd.DateOffset(years=settings.years)
    recent = returns[returns.index > start]
    error = _weekly_tracking_error(recent, tracked)
    rows = [_rating_row(FUND, recent, error, settings)]

    for name, levels in books.items():
        _, theirs = aligned(recent, levels)

        if len(theirs) > 1:
            rows.append(_rating_row(name, theirs, None, settings))

    return pd.DataFrame(rows, columns=RATING_COLUMNS)


def risk_distribution(returns: pd.Series,
                      tracked: pd.Series | None,
                      settings: RiskRatingSettings,
                      periods_per_year: int) -> pd.DataFrame | None:
    """The days the fund spent at each rating and band, on each day's
    volatility and tracking error over the year ending that day.

    Returns:
        pd.DataFrame | None: A row a rating and band the fund was at:
        ``rating``, ``band`` (blank without a comparator), ``days`` and
        ``share``, the days over all counted, by rating and then band. None
        for a run with under a year of returns.
    """
    if len(returns) < periods_per_year:
        return None

    root = np.sqrt(periods_per_year)
    volatility = (returns.rolling(periods_per_year).std() * root).dropna()
    rated = ratings(volatility.to_numpy(dtype=float), settings.volatility_bounds)
    banded: list[str | None] = [None] * len(volatility)

    if tracked is not None:
        ours, theirs = aligned(returns, tracked)
        error = ((ours - theirs).rolling(periods_per_year).std() * root).reindex(
            volatility.index)
        known = error.notna().to_numpy()
        found = bands(error.fillna(0.0).to_numpy(dtype=float), settings.tracking_bounds)
        banded = [band if ok else None for band, ok in zip(found, known, strict=True)]

    cells = Counter(zip(rated.tolist(), banded, strict=True))
    order = sorted(cells, key=lambda cell: (cell[0], _band_order(cell[1])))
    counted = len(volatility)

    return pd.DataFrame([{"rating": rating, "band": band, "days": cells[(rating, band)],
                          "share": cells[(rating, band)] / counted}
                         for rating, band in order], columns=DISTRIBUTION_COLUMNS)


def _rating_row(book: str,
                returns: pd.Series,
                tracking_error: float | None,
                settings: RiskRatingSettings) -> dict[str, object]:
    volatility = formulas.volatility(weekly(returns), WEEKS_A_YEAR)
    band = (None if tracking_error is None
            else str(bands(tracking_error, settings.tracking_bounds)[0]))

    return {"book": book, "rating": int(ratings(volatility, settings.volatility_bounds)[0]),
            "band": band, "volatility": volatility, "tracking_error": tracking_error}


def _weekly_tracking_error(returns: pd.Series,
                           tracked: pd.Series | None) -> float | None:
    """The annualised spread of the weekly returns less the comparator's,
    on the days both have; None without one."""
    if tracked is None:
        return None

    ours, theirs = aligned(returns, tracked)

    return formulas.tracking_error(weekly(ours) - weekly(theirs), WEEKS_A_YEAR)


def _band_order(band: str | None) -> int:
    return BANDS.index(band) if band is not None else len(BANDS)


def _check_bounds(bounds: tuple[float, ...],
                  name: str) -> None:
    if not bounds or any(low >= high for low, high in pairwise(bounds)):
        raise ValueError(f"{name} must be one or more rising bounds; got {bounds}.")

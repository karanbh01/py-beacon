# src/beacon/analysis/annualising.py
"""
Which figures are annualised, and how short a window each is left blank over.

A figure annualised over a few weeks says more about the scaling than the
fund, so it is left blank unless its window is long enough:

- **Rates** compounded or scaled to a year (the annualised return, alpha and
  the ratios built on an annual return) need at least
  ``min_annualised_rate_window``, a year unless set.
- **Ratios** scaled by the square root of a year (the Sharpe, Sortino,
  information and Lo Sharpe ratios) need at least
  ``min_annualised_ratio_window``, six months unless set.
- **Spreads** (volatility, tracking error, residual risk) are always given:
  scaling a spread to a year is standard over any window.

Both thresholds are modelling assumptions (`beacon.ModellingAssumptions`),
which an analysis can override (`AnalysisSettings`). A window counts as long
enough when it holds the threshold's daily returns less a week's, as a
calendar year holds 250 to 253 trading days. ``0M`` turns a threshold off.

Every annualised statistic's analytic takes ``annualise``: True gives the
annualised figure over any window, False the figure per day (measured with a
year of one period), which is never blanked.
"""
# BN-465 (Karan, 2026-11-25): one rule wherever a figure is asked for, in
# place of the trailing statistics table's own (BN-422, removed in BN-467).
from collections.abc import Mapping
from typing import Any

from ..assumptions import is_window

RATE, RATIO, SPREAD = "rate", "ratio", "spread"

# Each annualised statistic's kind, with the fields of it that are annualised
# when not all of them are (alpha's t-statistic and p-value are not).
ANNUALISED: dict[str, tuple[str, tuple[str, ...]]] = {
    "annualised_return": (RATE, ()), "annualised_relative_return": (RATE, ()),
    "alpha": (RATE, ("alpha",)), "calmar_ratio": (RATE, ()), "martin_ratio": (RATE, ()),
    "sterling_ratio": (RATE, ()), "burke_ratio": (RATE, ()), "treynor_ratio": (RATE, ()),
    "sharpe_ratio": (RATIO, ()), "sortino_ratio": (RATIO, ()),
    "information_ratio": (RATIO, ()), "lo_sharpe_ratio": (RATIO, ()),
    "volatility": (SPREAD, ()), "tracking_error": (SPREAD, ()), "residual_risk": (SPREAD, ()),
    "downside_tracking_error": (SPREAD, ()),
}

# The threshold that turns the rule off.
ANY_WINDOW = "0M"

# A threshold's unit in months.
MONTHS = {"M": 1, "Y": 12}


def days_in(span: str,
            periods_per_year: int) -> int:
    """How many daily returns window *span* holds, such as 126 for ``6M``.

    Raises:
        ValueError: If *span* is not a number of months or years.
    """
    if not is_window(span):
        raise ValueError(f"A window is a number of months or years, such as '6M' or '1Y', "
                         f"got {span!r}.")

    return round(int(span[:-1]) * MONTHS[span[-1:].upper()] * periods_per_year / 12)


def long_enough(days: int,
                span: str,
                periods_per_year: int) -> bool:
    """Whether *days* daily returns are window *span* or longer, short by
    less than a week's."""
    needed = days_in(span, periods_per_year)

    return needed == 0 or days >= needed - round(periods_per_year / 52)


def blanked_over(statistic_id: str,
                 days: int,
                 rate_window: str,
                 ratio_window: str,
                 periods_per_year: int) -> tuple[str, ...] | None:
    """The fields of *statistic_id* to leave blank over *days* daily returns:
    every field when empty, none when None."""
    kind, fields = ANNUALISED.get(statistic_id, (SPREAD, ()))

    if kind == SPREAD:
        return None

    span = rate_window if kind == RATE else ratio_window

    return None if long_enough(days, span, periods_per_year) else fields


def blank(value: Any,
          fields: tuple[str, ...]) -> Any:
    """*value*'s figures left blank: every one when *fields* is empty, else
    those named *fields*, in a mapping nested to any depth."""
    if isinstance(value, Mapping):
        return {key: None if key in fields and not isinstance(item, Mapping)
                else blank(item, fields)
                for key, item in value.items()}

    return value if fields else None


__all__ = ["ANNUALISED", "ANY_WINDOW", "RATE", "RATIO", "SPREAD", "blank", "blanked_over",
           "days_in", "long_enough"]

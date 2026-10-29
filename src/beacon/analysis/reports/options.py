# src/beacon/analysis/reports/options.py
"""
Every modelling choice a run's analysis makes, in sections, each with its
default, and described well enough for a form to draw.

    options = AnalysisOptions(rolling=RollingOptions(window=126))

A section left out takes its defaults. `describe` gives each section's
settings in the shape `GET /backtest/options` describes a backtest's.
"""
# BN-398/BN-400. The sections are those the analysis reads; a setting is here
# only when a calculation takes it.
import types
from dataclasses import dataclass, field, fields
from typing import Any, Union, get_args, get_origin, get_type_hints

from ..performance.settings import AnalysisSettings

DAILY, MONTH_END = "daily", "month_end"
STEPS = (DAILY, MONTH_END)

INDEX, BENCHMARK, OPTIMISED = "index", "benchmark", "optimised"


@dataclass(frozen=True)
class RollingOptions:
    """How rolling figures are measured.

    Attributes:
        window: The trailing window, in days.
        step: How often a window ends: every day or every month end; by each
            statistic's cost when None (month end for the heavy ones).
    """
    window: int = 252
    step: str | None = None


@dataclass(frozen=True)
class ReturnsOptions:
    """Choices the return, drawdown and distribution figures make.

    Attributes:
        worst_drawdowns: How many of the deepest drawdowns are listed.
        gains_frequency: The returns average gains and losses are measured on.
        sharpe_benchmark: The Sharpe ratio the probabilistic Sharpe ratio is
            the chance of beating.
        trials: How many strategies were tried, for the deflated Sharpe
            ratio; the run alone when None.
        trial_variance: The variance of the tried strategies' Sharpe ratios;
            the run's own when None.
        bootstrap_samples: How many resamples the ranges are drawn from.
        bootstrap_confidence: The share of resamples the ranges hold.
        bootstrap_block: Consecutive days drawn together.
        bootstrap_seed: The random seed, so a range is the same each time.
        histogram_bins: How many bins the return histogram has.
    """
    worst_drawdowns: int = 5
    gains_frequency: str = "daily"
    sharpe_benchmark: float = 0.0
    trials: int | None = None
    trial_variance: float | None = None
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    bootstrap_block: int = 1
    bootstrap_seed: int = 0
    histogram_bins: int = 50


@dataclass(frozen=True)
class CrisesOptions:
    """How crises and regimes are found.

    Attributes:
        market: The comparator crises and regimes are read from: the
            benchmark when the run has one, else the index, when None.
        threshold: The fall from a peak that makes a crisis.
    """
    market: str | None = None
    threshold: float = 0.15


@dataclass(frozen=True)
class LossOptions:
    """How value at risk and expected shortfall are given.

    Attributes:
        levels: The confidence levels.
        horizons: The horizons, in days; longer ones scale the day's by the
            square root of time.
    """
    levels: tuple[float, ...] = (0.95, 0.99)
    horizons: tuple[int, ...] = (1, 10)


@dataclass(frozen=True)
class RiskModelOptions:
    """How expected risk is worked out.

    Attributes:
        window: The days of returns before each rebalance the factor model
            is fitted to.
    """
    window: int = 252


@dataclass(frozen=True)
class LiquidityOptions:
    """How fast the book could be sold.

    Attributes:
        participation: The share of a name's daily traded value the book
            could sell.
        lookback_days: The days a name's average traded value is taken over.
        horizons: The days the sellable share is given at.
    """
    participation: float = 0.2
    lookback_days: int = 20
    horizons: tuple[int, ...] = (1, 5, 20)


@dataclass(frozen=True)
class AttributionOptions:
    """How returns are attributed.

    Attributes:
        grouping: The reference field names are grouped by.
        frequency: The periods attribution is also given by: ``Y``, ``Q``
            or ``M``.
    """
    grouping: str = "SECTOR"
    frequency: str = "Y"


@dataclass(frozen=True)
class AnalysisOptions:
    """Every choice a run's analysis makes, by section."""
    window: AnalysisSettings = field(default_factory=AnalysisSettings)
    rolling: RollingOptions = field(default_factory=RollingOptions)
    returns: ReturnsOptions = field(default_factory=ReturnsOptions)
    crises: CrisesOptions = field(default_factory=CrisesOptions)
    value_at_risk: LossOptions = field(default_factory=LossOptions)
    risk_model: RiskModelOptions = field(default_factory=RiskModelOptions)
    liquidity: LiquidityOptions = field(default_factory=LiquidityOptions)
    attribution: AttributionOptions = field(default_factory=AttributionOptions)


@dataclass(frozen=True)
class Described:
    """How one setting is shown: what `describe` adds to its name, type and
    default."""
    label: str
    help: str
    unit: str | None = None
    choices: tuple[Any, ...] | None = None
    choice_labels: dict[str, str] | None = None
    suggestions: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    items: str | None = None
    items_unit: str | None = None


COMPARATORS = {INDEX: "The index", BENCHMARK: "The benchmark", OPTIMISED: "The optimised index"}

SECTIONS: dict[str, tuple[str, dict[str, Described]]] = {
    "window": ("Window", {
        "start": Described("Start", "The first day measured; the run's first when empty.",
                           unit="date"),
        "end": Described("End", "The last day measured; the run's last when empty.", unit="date"),
        "periods_per_year": Described("Days a year", "How many daily returns make a year; the "
                                      "run's calendar's when empty.", unit="count", minimum=1),
        "risk_free_rate": Described("Risk-free rate", "The annual rate excess returns are "
                                    "measured over; the run's when empty.", unit="fraction"),
        "tracked": Described("Compared with", "The comparator relative figures and active "
                             "returns are measured against; the index the run traded toward "
                             "when empty.", choices=tuple(COMPARATORS),
                             choice_labels=COMPARATORS),
    }),
    "rolling": ("Rolling figures", {
        "window": Described("Window", "The trailing window each rolling figure covers.",
                            unit="days", suggestions=(63, 126, 252, 756, 1260), minimum=2),
        "step": Described("Step", "How often a window ends; by each figure's cost when empty "
                          "(month end for the heavy ones).", choices=STEPS,
                          choice_labels={DAILY: "Every day", MONTH_END: "Every month end"}),
    }),
    "returns": ("Returns and ratios", {
        "worst_drawdowns": Described("Drawdowns listed", "How many of the deepest drawdowns "
                                     "are listed.", unit="count", minimum=1),
        "gains_frequency": Described("Gains measured on", "The returns average gains and "
                                     "losses are measured on.",
                                     choices=("daily", "weekly", "monthly"),
                                     choice_labels={"daily": "Days", "weekly": "Weeks",
                                                    "monthly": "Months"}),
        "sharpe_benchmark": Described("Sharpe ratio to beat", "The Sharpe ratio the "
                                      "probabilistic Sharpe ratio is the chance of beating.",
                                      unit="ratio"),
        "trials": Described("Strategies tried", "How many strategies were tried, for the "
                            "deflated Sharpe ratio; the run alone when empty.", unit="count",
                            minimum=1),
        "trial_variance": Described("Variance of their Sharpe ratios", "The variance of the "
                                    "tried strategies' Sharpe ratios; the run's own when "
                                    "empty.", minimum=0.0),
        "bootstrap_samples": Described("Resamples", "How many resamples the ranges are "
                                       "drawn from.", unit="count", minimum=1),
        "bootstrap_confidence": Described("Range held", "The share of resamples a range "
                                          "holds.", unit="fraction", minimum=0.0, maximum=1.0),
        "bootstrap_block": Described("Days drawn together", "Consecutive days drawn "
                                     "together, which keeps runs of days intact.",
                                     unit="days", minimum=1),
        "bootstrap_seed": Described("Seed", "The random seed, so a range is the same each "
                                    "time."),
        "histogram_bins": Described("Histogram bins", "How many bins the return histogram "
                                    "has.", unit="count", minimum=1),
    }),
    "crises": ("Crises and regimes", {
        "market": Described("Market", "The comparator crises and regimes are read from; the "
                            "benchmark when the run has one, else the index, when empty.",
                            choices=(INDEX, BENCHMARK),
                            choice_labels={INDEX: COMPARATORS[INDEX],
                                           BENCHMARK: COMPARATORS[BENCHMARK]}),
        "threshold": Described("Fall that makes a crisis", "The fall from a peak that makes "
                               "a crisis.", unit="fraction", minimum=0.0, maximum=1.0),
    }),
    "value_at_risk": ("Value at risk", {
        "levels": Described("Confidence levels", "Each level value at risk and expected "
                            "shortfall are given at.", items="number", items_unit="fraction",
                            choices=(0.9, 0.95, 0.975, 0.99, 0.995)),
        "horizons": Described("Horizons", "Each horizon they are given over; longer ones "
                              "scale a day's by the square root of time.", items="integer",
                              items_unit="days", choices=(1, 5, 10, 20)),
    }),
    "risk_model": ("Risk model", {
        "window": Described("Lookback", "The days of returns before each rebalance the factor "
                            "model is fitted to.", unit="days", suggestions=(126, 252, 504),
                            minimum=20),
    }),
    "liquidity": ("Liquidity", {
        "participation": Described("Participation", "The share of a name's daily traded "
                                   "value the book could sell.", unit="fraction",
                                   minimum=0.0, maximum=1.0),
        "lookback_days": Described("Traded value over", "The days a name's average traded "
                                   "value is taken over.", unit="days", minimum=1),
        "horizons": Described("Days to sell", "The days the sellable share is given at.",
                              items="integer", items_unit="days", choices=(1, 2, 5, 10, 20)),
    }),
    "attribution": ("Attribution", {
        "grouping": Described("Grouped by", "The reference field names are grouped by.",
                              suggestions=("SECTOR", "INDUSTRY", "COUNTRY", "REGION")),
        "frequency": Described("Periods", "The periods attribution is also given by.",
                               choices=("Y", "Q", "M"),
                               choice_labels={"Y": "Years", "Q": "Quarters", "M": "Months"}),
    }),
}


def describe(options: AnalysisOptions | None = None) -> list[dict[str, Any]]:
    """Each section's settings, in order, with their defaults from *options*
    (the defaults when None)."""
    options = options or AnalysisOptions()
    sections = []

    for order, (name, (label, described)) in enumerate(SECTIONS.items(), start=1):
        section = getattr(options, name)
        hints = get_type_hints(type(section))
        parameters = [_parameter(item.name, hints[item.name], described[item.name],
                                 getattr(section, item.name), place)
                      for place, item in enumerate(fields(section), start=1)
                      if item.name in described]
        sections.append({"name": name, "label": label, "order": order, "parameters": parameters})

    return sections


def _parameter(name: str,
               annotation: Any,
               described: Described,
               default: Any,
               order: int) -> dict[str, Any]:
    """One setting described."""
    kind = _type(annotation, described)
    entry: dict[str, Any] = {"name": name, "type": kind, "required": False,
                             "default": list(default) if isinstance(default, tuple) else default,
                             "label": described.label, "order": order, "help": described.help,
                             "unit": None if kind == "array" else described.unit,
                             "minimum": described.minimum, "maximum": described.maximum}

    if kind == "array":
        entry["items"] = {"type": described.items, "unit": described.items_unit,
                          "choices": list(described.choices) if described.choices else None}
    else:
        entry["choices"] = list(described.choices) if described.choices else None
        entry["choice_labels"] = described.choice_labels
        entry["suggestions"] = list(described.suggestions) if described.suggestions else None

    return entry


def _type(annotation: Any,
          described: Described) -> str:
    """The display type of a setting's annotation: its first type other
    than None."""
    if described.items is not None:
        return "array"

    union = get_origin(annotation) in (Union, types.UnionType)
    chosen = next((arg for arg in get_args(annotation) if arg is not type(None)),
                  annotation) if union else annotation

    return {bool: "boolean", int: "integer", float: "number"}.get(chosen, "string")

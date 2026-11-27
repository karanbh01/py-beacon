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

from ..performance.members import FREQUENCIES, MONTHLY, WEEKLY, RiskReturnSettings
from ..performance.risk_rating import RiskRatingSettings
from ..performance.settings import AnalysisSettings
from ..performance.style import THREE_YEARS, YEAR, StyleSettings

DAILY, MONTH_END = "daily", "month_end"
STEPS = (DAILY, MONTH_END)

INDEX, BENCHMARK, OPTIMISED = "index", "benchmark", "optimised"

MARKET_FALLS, NAMED = "market", "named"

# How a contribution over each period is linked (BN-469, Karan 2026-11-26).
WINDOW, PERIOD = "window", "period"
LINKINGS = (WINDOW, PERIOD)


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
        histogram_horizon: The days each return in the return distribution
            is compounded over, in periods that do not overlap.
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
    histogram_horizon: int = 1


@dataclass(frozen=True)
class CrisesOptions:
    """How crises and regimes are found.

    Attributes:
        market: The comparator crises and regimes are read from: the
            benchmark when the run has one, else the index, when None.
        threshold: The fall from a peak that makes a crisis.
        episodes: Which crises: ``market``, the market's falls of
            *threshold* or more, or ``named``, the named list from the
            dot-com bust on, each with a row whether the run saw it or not.
    """
    market: str | None = None
    threshold: float = 0.15
    episodes: str = MARKET_FALLS


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
            is fitted to, when it is fitted on daily returns.
        structure: ``global`` (the styles, with country, sector and currency
            factors), ``styles`` (the market and the styles alone), or
            ``auto``: global when the names span more than one country.
        country_field: The reference field each name's country is read
            from; empty for no country factors.
        minimum_names: The fewest names a country needs for a factor of its
            own; one with fewer joins its region's.
        sectors: Whether sectors are factors in the global model.
        currencies: Whether currencies are factors in the global model.
        weekly: Whether the global model is fitted on weekly returns.
        weeks: The weeks of returns before each rebalance the weekly model
            is fitted to.
    """
    window: int = 252
    structure: str = "auto"
    country_field: str = "COUNTRY_DOMICILE"
    minimum_names: int = 10
    sectors: bool = True
    currencies: bool = True
    weekly: bool = True
    weeks: int = 104

    def settings(self) -> Any:
        """These choices as the risk model's settings
        (`beacon.analysis.risk.model.RiskModelSettings`)."""
        # The risk model reads a RunData, whose module imports the backtest,
        # which imports these options: so it is imported here.
        from ..risk.model import RiskModelSettings  # noqa: PLC0415

        return RiskModelSettings(window=self.window, structure=self.structure,
                                 country_field=self.country_field or None,
                                 minimum_names=self.minimum_names, sectors=self.sectors,
                                 currencies=self.currencies, weekly=self.weekly,
                                 weeks=self.weeks)


@dataclass(frozen=True)
class LiquidityOptions:
    """How fast the portfolio could be sold.

    Attributes:
        participation: The share of a name's daily traded value the portfolio
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
        linking: How a contribution over each period is linked: ``window``,
            as its share of the whole window's return, so the periods add up
            to the window's contribution; or ``period``, each period on its
            own, as the window from the last day before it to its own last
            day would give it.

    Raises:
        ValueError: If *linking* is neither.
    """
    grouping: str = "SECTOR"
    frequency: str = "Y"
    linking: str = WINDOW

    def __post_init__(self) -> None:
        if self.linking not in LINKINGS:
            raise ValueError(f"Unknown linking {self.linking!r}. Known: {', '.join(LINKINGS)}.")


@dataclass(frozen=True)
class SelectionOptions:
    """How a name's selection is read.

    Attributes:
        near_miss: The margin, as a share of the threshold, under which a
            name that passed an eligibility rule came close to failing it,
            and is listed among its events.
    """
    # BN-474 (Karan, 2026-11-26).
    near_miss: float = 0.1


@dataclass(frozen=True)
class ExposuresOptions:
    """Where the fund's latest weights by sector and by country are read
    from.

    Attributes:
        sector_field: The reference field each name's sector is read from.
        country_field: The reference field each name's country is read
            from: its domicile by default, as the risk model's countries.
    """
    sector_field: str = "SECTOR"
    country_field: str = "COUNTRY_DOMICILE"


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
    selection: SelectionOptions = field(default_factory=SelectionOptions)
    exposures: ExposuresOptions = field(default_factory=ExposuresOptions)
    style: StyleSettings = field(default_factory=StyleSettings)
    risk_rating: RiskRatingSettings = field(default_factory=RiskRatingSettings)
    risk_return: RiskReturnSettings = field(default_factory=RiskReturnSettings)


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

# The windows the annualising thresholds suggest.
WINDOWS = ("0M", "3M", "6M", "1Y", "3Y")

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
        "min_annualised_rate_window": Described(
            "Shortest window for a rate a year", "The shortest window a rate compounded or "
            "scaled to a year, such as the annualised return, is given over; 0M gives it over "
            "any. The run's when empty.", suggestions=WINDOWS),
        "min_annualised_ratio_window": Described(
            "Shortest window for a ratio a year", "The shortest window a ratio scaled by the "
            "square root of a year, such as the Sharpe ratio, is given over; 0M gives it over "
            "any. The run's when empty.", suggestions=WINDOWS),
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
        "histogram_horizon": Described("Distribution horizon", "The days each return in the "
                                       "return distribution is compounded over, in periods "
                                       "that do not overlap.", unit="days",
                                       suggestions=(1, 5, 21), minimum=1),
    }),
    "crises": ("Crises and regimes", {
        "market": Described("Market", "The comparator crises and regimes are read from; the "
                            "benchmark when the run has one, else the index, when empty.",
                            choices=(INDEX, BENCHMARK),
                            choice_labels={INDEX: COMPARATORS[INDEX],
                                           BENCHMARK: COMPARATORS[BENCHMARK]}),
        "threshold": Described("Fall that makes a crisis", "The fall from a peak that makes "
                               "a crisis.", unit="fraction", minimum=0.0, maximum=1.0),
        "episodes": Described("Crises", "The market's falls of the threshold or more, or the "
                              "named list from the dot-com bust on, each listed whether the "
                              "run saw it or not.", choices=(MARKET_FALLS, NAMED),
                              choice_labels={MARKET_FALLS: "The market's falls",
                                             NAMED: "Named crises"}),
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
                            "model is fitted to, when it is fitted on daily returns.",
                            unit="days", suggestions=(126, 252, 504), minimum=20),
        "structure": Described("Factors", "Which factors the model has. Global adds country, "
                               "sector and currency factors to the styles; automatic picks it "
                               "when the names span more than one country.",
                               choices=("auto", "global", "styles"),
                               choice_labels={"auto": "Automatic", "global": "Global",
                                              "styles": "Styles only"}),
        "country_field": Described("Country from", "The reference field each name's country "
                                   "is read from; no country factors when empty.",
                                   suggestions=("COUNTRY_DOMICILE", "COUNTRY_LISTING",
                                                "REGION")),
        "minimum_names": Described("Fewest names a country", "A country with fewer names "
                                   "joins its region's factor.", minimum=1),
        "sectors": Described("Sector factors", "Whether the global model has a factor for "
                             "each sector."),
        "currencies": Described("Currency factors", "Whether the global model has a factor "
                                "for each currency, its return against the base."),
        "weekly": Described("Weekly returns", "Whether the global model is fitted on weekly "
                            "returns, which allow for markets closing at different times."),
        "weeks": Described("Weeks", "The weeks of returns before each rebalance the weekly "
                           "model is fitted to.", unit="weeks", suggestions=(52, 104, 156),
                           minimum=8),
    }),
    "liquidity": ("Liquidity", {
        "participation": Described("Participation", "The share of a name's daily traded "
                                   "value the portfolio could sell.", unit="fraction",
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
        "linking": Described("Periods linked", "How a contribution over each period, such as "
                             "a name's each quarter, is linked: as its share of the whole "
                             "window's return, so the periods add up to the window's "
                             "contribution, or each period on its own, so each is the "
                             "contribution over that period alone and the periods do not add "
                             "up.", choices=LINKINGS,
                             choice_labels={WINDOW: "As a share of the window",
                                            PERIOD: "Each period on its own"}),
    }),
    "selection": ("Selection", {
        "near_miss": Described("Near miss within", "The margin, as a share of the threshold, "
                               "under which a name that passed an eligibility rule came close "
                               "to failing it, and is listed among its events.",
                               unit="fraction", suggestions=(0.05, 0.1, 0.2), minimum=0.0),
    }),
    "exposures": ("Exposures", {
        "sector_field": Described("Sector from", "The reference field each name's sector is "
                                  "read from.", suggestions=("SECTOR", "GICS_INDUSTRY_GROUP")),
        "country_field": Described("Country from", "The reference field each name's country "
                                   "is read from.",
                                   suggestions=("COUNTRY_DOMICILE", "COUNTRY_LISTING",
                                                "REGION")),
    }),
    "style": ("Style box", {
        "large_from": Described("Large above", "The market cap, in US dollars, large starts "
                                "above.", unit="usd", minimum=0.0),
        "mid_from": Described("Mid from", "The market cap, in US dollars, mid starts at; "
                              "small is below it.", unit="usd", minimum=0.0),
        "top": Described("Top of large", "The market cap at the top of the box's scale; a "
                         "larger name sits above it.", unit="usd", minimum=0.0),
        "bottom": Described("Bottom of small", "The market cap at the bottom of the box's "
                            "scale; a smaller name sits below it.", unit="usd", minimum=0.0),
        "growth": Described("Growth over", "The earnings and sales growth the style is "
                            "measured on.", choices=(YEAR, THREE_YEARS),
                            choice_labels={YEAR: "A year", THREE_YEARS: "Three years, a year"}),
    }),
    "risk_rating": ("Risk rating", {
        "volatility_bounds": Described("Volatility bounds", "The annualised volatility each "
                                       "rating above 1 starts at, rising; the factsheet "
                                       "scale's by default.", items="number",
                                       items_unit="fraction"),
        "tracking_bounds": Described("Tracking error bounds", "The tracking error enhanced, "
                                     "active and unconstrained start at, rising.",
                                     items="number", items_unit="fraction"),
        "years": Described("Years measured", "The years of weekly returns the rating is "
                           "measured over; all of a shorter run's.", unit="count", minimum=1),
    }),
    "risk_return": ("Risk and return", {
        "frequency": Described("Measured on", "The returns each name's and portfolio's risk "
                               "and return, and the frontier, are measured on.",
                               choices=tuple(FREQUENCIES),
                               choice_labels={WEEKLY: "Weeks", MONTHLY: "Months"}),
        "frontier_names": Described("Frontier names", "How many of the largest latest index "
                                    "members with a return in every period the frontier is "
                                    "drawn from.", unit="count", minimum=2),
        "frontier_points": Described("Frontier points", "How many portfolios the frontier is "
                                     "traced at.", unit="count", minimum=2),
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

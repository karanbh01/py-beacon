# src/beacon/derivatives/term_structure.py
"""
Several expiries on one underlying, and what they say about each other.

A single futures price is not very informative. A strip of them is: the shape
of the basis across expiries is where financing assumptions, dividend
expectations and plain supply-and-demand show up, and none of that is visible
one contract at a time.

Two views of the same disagreement, and it is worth being clear that they are
the same disagreement:

* **Basis** — market price minus theoretical price. Answers "how much is this
  contract off my model", in price terms.
* **Implied repo** — the financing rate that would make the model agree with
  the market. Answers "what would I have to believe for this price to be
  right", in rate terms.

A rich contract has a positive basis and an implied repo above the curve. They
cannot disagree about direction, and a test holds them to that.
"""
from dataclasses import dataclass, field

import pandas as pd

from ..exceptions import CalculationError
from .curves import RateCurve
from .pricing import cost_of_carry_fair_value, implied_repo_rate

# Time to expiry uses ACT/365, matching DerivativeBase.time_to_expiry.
DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class FuturesQuote:
    """One expiry and the price the market puts on it.

    Attributes:
        expiry: Contract expiry date.
        market_price: Traded price. None when only a theoretical value is
            wanted, in which case basis and implied repo are not reported for
            this pillar rather than being invented.
        label: Optional contract code, for display.
    """
    expiry: pd.Timestamp
    market_price: float | None = None
    label: str = ""


@dataclass
class TermStructure:
    """A strip of futures on one underlying, valued off one curve.

    Attributes:
        underlying: Identifier of the underlying.
        spot: Spot price at *valuation_date*.
        valuation_date: The date everything is measured from.
        quotes: The expiries, in any order; they are sorted on construction.
        curve: Financing curve. A flat curve reproduces scalar-rate pricing
            exactly.
        dividend_yield: Continuous dividend yield on the underlying.
        borrow_cost: Continuous borrow spread.
    """
    underlying: str
    spot: float
    valuation_date: pd.Timestamp
    quotes: list[FuturesQuote]
    curve: RateCurve
    dividend_yield: float = 0.0
    borrow_cost: float = 0.0
    _sorted: list[FuturesQuote] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.quotes:
            raise CalculationError("TermStructure", "no expiries were supplied.")

        if self.spot <= 0.0:
            raise CalculationError("TermStructure",
                                   f"spot must be positive, got {self.spot}.")

        self.valuation_date = pd.Timestamp(self.valuation_date)
        self._sorted = sorted(self.quotes, key=lambda quote: pd.Timestamp(quote.expiry))

        earliest = pd.Timestamp(self._sorted[0].expiry)
        if earliest < self.valuation_date:
            raise CalculationError(
                "TermStructure",
                f"expiry {earliest.date()} is before the valuation date "
                f"{self.valuation_date.date()}.")

    @property
    def expiries(self) -> list[pd.Timestamp]:
        """Expiry dates, nearest first."""
        return [pd.Timestamp(quote.expiry) for quote in self._sorted]

    def times_to_expiry(self) -> list[float]:
        """Year fractions to each expiry, ACT/365."""
        return [(pd.Timestamp(quote.expiry) - self.valuation_date).days / DAYS_PER_YEAR
                for quote in self._sorted]

    def financing_rates(self) -> list[float]:
        """The curve's rate at each expiry."""
        return [self.curve.zero_rate(tenor) for tenor in self.times_to_expiry()]

    def theoretical_prices(self) -> pd.Series:
        """Fair value at each expiry, off the curve.

        Returns:
            pd.Series: Indexed by expiry date.
        """
        values = [
            cost_of_carry_fair_value(spot=self.spot,
                                     risk_free_rate=rate,
                                     dividend_yield=self.dividend_yield,
                                     time_to_expiry_years=tenor,
                                     borrow_cost=self.borrow_cost)
            for rate, tenor in zip(self.financing_rates(),
                                   self.times_to_expiry(),
                                   strict=True)
        ]

        return pd.Series(values, index=self.expiries, name="theoretical")

    def market_prices(self) -> pd.Series:
        """Quoted prices, NaN where a quote carries none."""
        return pd.Series([quote.market_price for quote in self._sorted],
                         index=self.expiries,
                         dtype=float,
                         name="market")

    def basis(self) -> pd.Series:
        """Market minus theoretical, per expiry.

        Positive means the contract trades rich to the model.
        """
        return (self.market_prices() - self.theoretical_prices()).rename("basis")

    def implied_repo(self) -> pd.Series:
        """The financing rate each quoted price implies.

        NaN for expiries with no quote, and for an expiry today — a zero year
        fraction carries no information about a rate, and dividing by it would
        manufacture one.

        Returns:
            pd.Series: Continuously compounded rates, indexed by expiry.
        """
        rates: list[float] = []

        for quote, tenor in zip(self._sorted, self.times_to_expiry(), strict=True):
            if quote.market_price is None or tenor <= 0.0:
                rates.append(float("nan"))
                continue

            rates.append(implied_repo_rate(futures_price=quote.market_price,
                                           spot=self.spot,
                                           dividend_yield=self.dividend_yield,
                                           time_to_expiry_years=tenor))

        return pd.Series(rates, index=self.expiries, name="implied_repo")

    def to_frame(self) -> pd.DataFrame:
        """Everything the strip says, one row per expiry."""
        frame = pd.DataFrame({
            "label": [quote.label for quote in self._sorted],
            "time_to_expiry": self.times_to_expiry(),
            "financing_rate": self.financing_rates(),
            "theoretical": self.theoretical_prices().to_numpy(),
            "market": self.market_prices().to_numpy(),
            "basis": self.basis().to_numpy(),
            "implied_repo": self.implied_repo().to_numpy(),
        }, index=self.expiries)
        frame.index.name = "expiry"

        return frame


def sensitivity_grid(spot: float,
                     tenors: list[float],
                     rates: list[float],
                     dividend_yield: float = 0.0,
                     borrow_cost: float = 0.0) -> pd.DataFrame:
    """Fair value across a tenor × rate grid.

    What a position is worth if the curve is somewhere else and expiry is
    further out — the two axes a Delta-1 desk actually moves along, laid out so
    the shape is visible at once rather than one revaluation at a time.

    Args:
        spot: Spot price of the underlying.
        tenors: Times to expiry in years, one per row.
        rates: Continuously compounded financing rates, one per column.
        dividend_yield: Continuous dividend yield.
        borrow_cost: Continuous borrow spread.

    Returns:
        pd.DataFrame: Fair values, tenors on the index and rates on the
        columns, both labelled with their values.

    Raises:
        CalculationError: If either axis is empty.
    """
    if not tenors or not rates:
        raise CalculationError("SensitivityGrid",
                               "both tenors and rates must be non-empty.")

    grid = [
        [cost_of_carry_fair_value(spot=spot,
                                  risk_free_rate=rate,
                                  dividend_yield=dividend_yield,
                                  time_to_expiry_years=tenor,
                                  borrow_cost=borrow_cost)
         for rate in rates]
        for tenor in tenors
    ]

    frame = pd.DataFrame(grid, index=list(tenors), columns=list(rates))
    frame.index.name = "time_to_expiry"
    frame.columns.name = "rate"

    return frame

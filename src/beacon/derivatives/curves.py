# src/beacon/derivatives/curves.py
"""
Interest-rate curves.

`pricing.py` takes rates as scalars and must stay free of Beacon imports, so the
curve lives here rather than inside it. The interface between the two is plain
floats: a curve is asked for a rate at a tenor and the answer goes into a
pricing function. Nothing about the curve crosses that boundary, which is what
keeps the purity test passing and keeps the pricing maths independently
checkable against a textbook.

Rates are **continuously compounded zero rates**, matching what `pricing.py`
already expects, so a flat curve and the scalar rate it replaces are the same
number and produce the same answer to the last bit.

## Interpolation, and what happens off the ends

Linear in the zero rate between pillars — the conventional choice, and the one
whose failure modes are understood. Beyond the first and last pillar the curve
is **flat**, holding the nearest pillar's rate rather than continuing its slope.
Extrapolating a slope off the end of a curve is how a two-year rate becomes a
negative thirty-year rate: it is arithmetically reasonable and financially
nonsense, and the error appears far from the code that caused it.
"""
import bisect
import math
from dataclasses import dataclass

from ..exceptions import CalculationError

# One basis point, the standard bump for a rate sensitivity.
BASIS_POINT = 0.0001

# Two tenors closer than this are the same pillar. Sub-second differences on a
# curve are meaningless and would make the interpolation divide by ~zero.
TENOR_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RateCurve:
    """A zero-rate curve defined by pillar points.

    Attributes:
        tenors: Pillar tenors in years, strictly increasing.
        rates: Continuously compounded zero rate at each pillar.
    """
    tenors: tuple[float, ...]
    rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.tenors:
            raise CalculationError("RateCurve", "a curve needs at least one pillar.")

        if len(self.tenors) != len(self.rates):
            raise CalculationError(
                "RateCurve",
                f"{len(self.tenors)} tenors but {len(self.rates)} rates.")

        if any(tenor < 0.0 for tenor in self.tenors):
            raise CalculationError("RateCurve", "tenors must be non-negative.")

        for earlier, later in zip(self.tenors, self.tenors[1:], strict=False):
            if later - earlier <= TENOR_TOLERANCE:
                raise CalculationError(
                    "RateCurve",
                    f"tenors must be strictly increasing, got {earlier} then {later}.")

    @classmethod
    def flat(cls,
             rate: float) -> "RateCurve":
        """A curve with the same rate at every tenor.

        The bridge back to scalar-rate pricing: a flat curve returns exactly the
        rate it was given, so every existing result is reproduced bit for bit
        rather than approximately.

        Args:
            rate: The continuously compounded rate.

        Returns:
            RateCurve: A single-pillar curve.
        """
        return cls(tenors=(1.0,), rates=(float(rate),))

    @classmethod
    def from_pillars(cls,
                     pillars: dict[float, float]) -> "RateCurve":
        """Build a curve from a ``{tenor: rate}`` mapping, sorted by tenor.

        Args:
            pillars: Tenor in years to continuously compounded zero rate.

        Returns:
            RateCurve: The curve.
        """
        if not pillars:
            raise CalculationError("RateCurve", "a curve needs at least one pillar.")

        ordered = sorted(pillars.items())

        return cls(tenors=tuple(float(tenor) for tenor, _ in ordered),
                   rates=tuple(float(rate) for _, rate in ordered))

    @property
    def is_flat(self) -> bool:
        """Whether every pillar carries the same rate."""
        return len(set(self.rates)) == 1

    def zero_rate(self,
                  tenor: float) -> float:
        """The zero rate at *tenor*, interpolated between pillars.

        Args:
            tenor: Years from the valuation date. Must be non-negative.

        Returns:
            float: The continuously compounded zero rate. Flat beyond the first
            and last pillar.

        Raises:
            CalculationError: If *tenor* is negative.
        """
        if tenor < 0.0:
            raise CalculationError("RateCurve",
                                   f"tenor must be non-negative, got {tenor}.")

        # Single pillar, or every pillar equal: one answer, returned exactly.
        # Not an optimisation — it is what makes a flat curve reproduce a scalar
        # rate without any interpolation arithmetic in between.
        if len(self.tenors) == 1 or self.is_flat:
            return self.rates[0]

        if tenor <= self.tenors[0]:
            return self.rates[0]

        if tenor >= self.tenors[-1]:
            return self.rates[-1]

        return self._interpolate(tenor)

    def _interpolate(self,
                     tenor: float) -> float:
        """Linear interpolation between the two bracketing pillars."""
        position = bisect.bisect_left(self.tenors, tenor)

        left_tenor, right_tenor = self.tenors[position - 1], self.tenors[position]
        left_rate, right_rate = self.rates[position - 1], self.rates[position]

        weight = (tenor - left_tenor) / (right_tenor - left_tenor)

        return left_rate + weight * (right_rate - left_rate)

    def discount_factor(self,
                        tenor: float) -> float:
        """Present value of one unit paid at *tenor*.

        Args:
            tenor: Years from the valuation date.

        Returns:
            float: ``exp(-z(T) * T)``. Exactly 1.0 at tenor zero.
        """
        if tenor == 0.0:
            return 1.0

        return math.exp(-self.zero_rate(tenor) * tenor)

    def forward_rate(self,
                     start: float,
                     end: float) -> float:
        """The rate implied between two future dates.

        The rate that makes discounting to *end* the same as discounting to
        *start* and then forward at this rate — which is what a financing leg
        resetting at *start* should be projected at.

        Args:
            start: Start of the forward period, in years.
            end: End of the forward period, in years.

        Returns:
            float: Continuously compounded forward rate.

        Raises:
            CalculationError: If *end* is not after *start*.
        """
        if end - start <= TENOR_TOLERANCE:
            raise CalculationError(
                "RateCurve",
                f"the forward period must be positive, got {start} to {end}.")

        # (z_end * end - z_start * start) / (end - start), which is the same as
        # -ln(DF_end / DF_start) / (end - start) without the round trip through
        # exp and log.
        return ((self.zero_rate(end) * end - self.zero_rate(start) * start)
                / (end - start))

    def shifted(self,
                bump: float) -> "RateCurve":
        """A copy with every pillar moved by *bump*.

        The parallel shift a DV01 is measured against.

        Args:
            bump: Amount to add to every rate, in decimal. One basis point is
                ``BASIS_POINT``.

        Returns:
            RateCurve: The shifted curve. The original is unchanged.
        """
        return RateCurve(tenors=self.tenors,
                         rates=tuple(rate + bump for rate in self.rates))

    def with_pillar_bump(self,
                         tenor: float,
                         bump: float) -> "RateCurve":
        """A copy with one pillar moved, for a key-rate sensitivity.

        Args:
            tenor: The pillar to move. Must be an existing pillar — bumping a
                tenor that is not there would silently add a pillar and change
                the curve's shape rather than its level, which is not what a
                key-rate bump means.
            bump: Amount to add to that pillar's rate.

        Returns:
            RateCurve: The bumped curve.

        Raises:
            CalculationError: If *tenor* is not a pillar.
        """
        for position, pillar in enumerate(self.tenors):
            if abs(pillar - tenor) <= TENOR_TOLERANCE:
                rates = list(self.rates)
                rates[position] += bump

                return RateCurve(tenors=self.tenors, rates=tuple(rates))

        raise CalculationError(
            "RateCurve",
            f"{tenor} is not a pillar on this curve. Available: "
            f"{', '.join(str(pillar) for pillar in self.tenors)}.")

    def to_dict(self) -> dict[float, float]:
        """The pillars as a ``{tenor: rate}`` mapping."""
        return dict(zip(self.tenors, self.rates, strict=True))

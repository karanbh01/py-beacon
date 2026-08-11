# src/beacon/synthetic/regions.py
"""
Listing venues: where a generated company trades, and in what currency.

A single-currency universe cannot exercise most of what Beacon does with
currency. The calculator converts every market value through
`fetch_fx_rates`, corporate actions convert their cash amounts the same way,
and both paths are dead code against a universe that is entirely USD — they
run, they multiply by 1.0, and nothing they could get wrong would show up.

## What a region is here

A listing venue with one currency, one exchange and a share of the world.
Deliberately *not* a country: "Europe" prices in EUR and lists on XETR, which
is true of enough of the continent to be useful and wrong about Switzerland.
Modelling countries properly would mean a currency per country and a
correlation structure between them, which is a bigger claim than generated
data should make.

## The weights are roughly MSCI ACWI

The United States really is around 60% of global equity market
capitalisation. A synthetic global universe that splits evenly across regions
would make every FX exposure look far more material than it is, and a
"global" index built on it would behave nothing like one built on the real
thing.

## Prices are quoted in local currency

A name's `initial_price` is in its own currency, and its market cap is
converted from the USD-scale draw so the *size* distribution stays global
while the quoted price stays local. Without that conversion a 5 billion
company would mean five billion of whatever it happens to be quoted in, and
the biggest companies in the universe would be an artefact of the exchange
rate.

The price *range* is deliberately not localised. Real JPY quotes run to
thousands of yen, but the split rule, the tick size and the four-decimal
rounding are all calibrated to the 12-480 band, and localising the range
would mean localising all three. The prices are local-currency amounts drawn
from a common band, which is a limitation worth stating rather than hiding.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE_CURRENCY = "USD"


@dataclass(frozen=True)
class Region:
    """One listing venue.

    Attributes:
        name: What the reference data reports as REGION.
        currency: ISO code every name listed here is quoted in.
        exchange: MIC the names carry.
        weight: Share of the universe, roughly MSCI ACWI.
        rate: Units of `currency` per one US dollar at the start of the panel.
            Quoted this way round -- rather than as the market convention,
            which differs per pair -- because it is the direction the
            conversion needs and a single convention cannot be misread.
        volatility: Annualised volatility of the exchange rate.
    """
    name: str
    currency: str
    exchange: str
    weight: float
    rate: float
    volatility: float


# Rates are approximately end-2024. FX volatilities are the realised figures
# for the major pairs: 7-11% for a floating currency, and 0.4% for the Hong
# Kong dollar, which is not floating at all -- it runs inside a band the HKMA
# defends, and a generator that gave it 9% like everything else would produce
# a "diversified" currency exposure that does not exist.
REGIONS = (
    Region("United States", "USD", "XNAS", 0.60, 1.00, 0.000),
    Region("Europe", "EUR", "XETR", 0.13, 0.95, 0.075),
    Region("Asia Pacific", "HKD", "XHKG", 0.12, 7.78, 0.004),
    Region("Japan", "JPY", "XTKS", 0.06, 157.0, 0.095),
    Region("United Kingdom", "GBP", "XLON", 0.04, 0.80, 0.085),
    Region("Canada", "CAD", "XTSE", 0.03, 1.44, 0.065),
    Region("Australia", "AUD", "XASX", 0.02, 1.61, 0.105),
)


def assign(count: int,
           rng: np.random.Generator) -> np.ndarray:
    """Which region each name lists in.

    Allocated by *quota* rather than drawn independently, so a 20-name
    universe still contains a non-US name and the realised weights match the
    targets at any size. An independent draw at 0.02 leaves Australia absent
    from most small universes, which is the size a test uses.

    Args:
        count: How many names.
        rng: Seeded generator, used only to shuffle the assignment so region
            does not correlate with position -- and therefore not with sector,
            which is assigned round-robin.

    Returns:
        np.ndarray: One region index per name.
    """
    quotas = [round(region.weight * count) for region in REGIONS]

    # Rounding leaves the total slightly off; the remainder goes to the
    # largest region, where a name either way changes nothing.
    quotas[0] += count - sum(quotas)
    quotas[0] = max(quotas[0], 0)

    assigned = np.repeat(np.arange(len(REGIONS)), quotas)[:count]

    if len(assigned) < count:
        assigned = np.concatenate(
            [assigned, np.zeros(count - len(assigned), dtype=int)])

    rng.shuffle(assigned)

    return assigned


def pairs() -> list[tuple[str, float, float]]:
    """The FX pairs a generated panel needs, as (identifier, rate, volatility).

    One per non-base currency, named the way `fetch_fx_rates` looks it up:
    ``f"{from}{to}"``, so converting EUR into USD reads ``EURUSD``. The rate
    is inverted from the region's, which stores units per dollar.
    """
    return [(f"{region.currency}{BASE_CURRENCY}", 1.0 / region.rate,
             region.volatility)
            for region in REGIONS if region.currency != BASE_CURRENCY]


def frame(assigned: np.ndarray) -> pd.DataFrame:
    """Region, currency and exchange per name, given the assignment."""
    chosen = [REGIONS[index] for index in assigned]

    return pd.DataFrame({
        "REGION": [region.name for region in chosen],
        "CURRENCY": [region.currency for region in chosen],
        "EXCHANGE": [region.exchange for region in chosen],
        "fx_to_base": [1.0 / region.rate for region in chosen],
    })

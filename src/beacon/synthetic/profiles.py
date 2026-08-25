# src/beacon/synthetic/profiles.py
"""
The reference fields a profile view expects: identifiers, classification
depth, and corporate facts.

The universe generator produces what the engine needs to *calculate* — sector,
currency, exchange, share count. A client showing an instrument's profile
needs a good deal more, and every field it cannot fill renders as a dash. This
generates the rest.

## Coherent with what is already generated

The rule the features work established, applied again. A generated fact that
contradicts another generated fact is worse than an absent one, because the
contradiction is invisible until somebody checks and by then it has been
believed.

So:

* `trading_status` follows the listings model — a name delisted in 2021 is not
  "Active", and a screen on status has to agree with one on dates
* `dividend_frequency` follows the dividend yield actually generated. A name
  that pays nothing is not "Quarterly"; saying so would make a
  dividend-frequency screen select names with no dividends
* `ipo_date` is the listing date, not a second date drawn beside it
* the GICS levels nest: sub-industry inside industry inside industry group
  inside sector. A four-level hierarchy whose levels do not contain each other
  is decoration, and any roll-up computed from it would be wrong

## The identifiers are structurally valid

`isin`, `cusip` and `sedol` carry their real check digits, and `figi` its real
shape. Not for authenticity — nothing here is a real security — but because a
client that validates an identifier before using it would otherwise reject the
whole store, and an ISIN is the field somebody is most likely to parse.

A wrong check digit is also the kind of thing that works until the day
something checks it, which is the failure mode worth spending twenty lines to
avoid.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# GICS has four levels. Only the first and last were generated before, which
# left the middle two empty and the hierarchy unusable for a roll-up.
INDUSTRY_GROUPS = ("Group I", "Group II")
INDUSTRIES = ("Industry A", "Industry B")

SECURITY_TYPE = "Common Stock"
ASSET_CLASS = "Equity"
INSTRUMENT_SUBTYPES = ("Ordinary Share", "Class A Share", "Class B Share")

ACTIVE = "Active"
DELISTED = "Delisted"

# What a name pays, and how often. Tied to the generated yield rather than
# drawn: a non-payer reporting "Quarterly" would make a frequency screen
# select names with no dividends.
NO_DIVIDEND = "None"
DIVIDEND_FREQUENCIES = ("Annual", "Semi-Annual", "Quarterly")

# Cities, paired with the listing country so a headquarters is not in a
# country the instrument has no connection to.
CITIES = {
    "US": ("New York", "San Francisco", "Chicago", "Boston", "Austin"),
    "GB": ("London", "Manchester", "Edinburgh"),
    "JP": ("Tokyo", "Osaka", "Nagoya"),
    "HK": ("Hong Kong",),
    "CA": ("Toronto", "Vancouver", "Calgary"),
    "AU": ("Sydney", "Melbourne", "Brisbane"),
    "DE": ("Frankfurt", "Munich", "Berlin"),
    "FR": ("Paris", "Lyon"),
    "CH": ("Zurich", "Geneva"),
    "NL": ("Amsterdam",),
    "SG": ("Singapore",),
}
DEFAULT_CITIES = ("New York",)

# Employees scale with market cap, because they do. Drawn around a headcount
# implied by size rather than independently, so a mega-cap does not come out
# with forty staff.
EMPLOYEES_PER_MILLION_CAP = 3.0
EMPLOYEE_DISPERSION = 0.6
MIN_EMPLOYEES = 25

# Founding years, and the gap before listing. A company is not usually listed
# the year it is founded.
EARLIEST_FOUNDED = 1890
MIN_YEARS_BEFORE_LISTING = 2

FISCAL_YEAR_ENDS = ("31 December", "31 March", "30 June", "30 September")

# How far ahead the next reporting date sits. Roughly a quarter, jittered.
MIN_DAYS_TO_EARNINGS = 5
MAX_DAYS_TO_EARNINGS = 95

# ISIN and SEDOL alphabets.
_DIGITS = "0123456789"
_ALPHANUMERIC = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# SEDOL omits vowels so a code cannot spell a word.
_SEDOL_ALPHABET = "0123456789BCDFGHJKLMNPQRSTVWXYZ"
_SEDOL_WEIGHTS = (1, 3, 1, 7, 3, 9, 1)


def build(universe: pd.DataFrame,
          rng: np.random.Generator,
          as_of: pd.Timestamp) -> pd.DataFrame:
    """Generate the profile columns for a universe.

    Args:
        universe: Output of `universe.build`, indexed by identifier.
        rng: Seeded generator.
        as_of: The panel's last date, for `next_earnings` and status.

    Returns:
        pd.DataFrame: Indexed by identifier, one column per profile field.
    """
    count = len(universe)
    identifiers = list(universe.index)

    countries = _column(universe, "COUNTRY_LISTING", "US")
    exchanges = _column(universe, "EXCHANGE", "XNAS")
    caps = _numeric(universe, "market_cap", 1e9)
    listed_from = _dates(universe, "listed_from", as_of)
    listed_to = _dates(universe, "listed_to", pd.NaT)
    yields = _numeric(universe, "dividend_yield", 0.0)

    sectors = _column(universe, "SECTOR", "Industrials")
    groups = rng.choice(INDUSTRY_GROUPS, size=count)
    industries = rng.choice(INDUSTRIES, size=count)

    frame = pd.DataFrame({
        # --- identifiers ---------------------------------------------------
        "TICKER": identifiers,
        "ISIN": [_isin(country, identifier, rng)
                 for country, identifier in zip(countries, identifiers,
                                                strict=True)],
        "CUSIP": [_cusip(rng) for _ in range(count)],
        "SEDOL": [_sedol(rng) for _ in range(count)],
        "FIGI": [_figi(rng) for _ in range(count)],
        "SECURITY_TYPE": SECURITY_TYPE,

        # --- classification -------------------------------------------------
        #
        # Nested rather than drawn flat, so a roll-up from sub-industry to
        # sector is a real aggregation.
        "GICS_INDUSTRY_GROUP": [f"{sector} — {group}" for sector, group
                                in zip(sectors, groups, strict=True)],
        "GICS_INDUSTRY": [f"{sector} — {group} — {industry}"
                          for sector, group, industry
                          in zip(sectors, groups, industries, strict=True)],
        "ASSET_CLASS": ASSET_CLASS,
        "INSTRUMENT_SUBTYPE": rng.choice(INSTRUMENT_SUBTYPES, size=count,
                                         p=[0.9, 0.06, 0.04]),
        "TRADING_STATUS": [DELISTED if pd.notna(end) and end <= as_of
                           else ACTIVE for end in listed_to],
        "PRIMARY_LISTING": exchanges,
        "ADR": [country != "US" and bool(flag) for country, flag
                in zip(countries, rng.uniform(size=count) < 0.15, strict=True)],
        "OPTIONS_AVAILABLE": caps > np.quantile(caps, 0.4),

        # --- corporate profile ---------------------------------------------
        "EMPLOYEES": _employees(caps, rng),
        "HEADQUARTERS": [_city(country, rng) for country in countries],
        "IPO_DATE": listed_from,
        "FISCAL_YEAR_END": rng.choice(FISCAL_YEAR_ENDS, size=count,
                                      p=[0.7, 0.12, 0.1, 0.08]),
        "DIVIDEND_FREQUENCY": _frequencies(yields, rng),
        "NEXT_EARNINGS": _next_earnings(as_of, count, rng),
    }, index=pd.Index(identifiers, name="IDENTIFIER"))

    # Founded before listing, by construction rather than by a second draw --
    # a company listed in 1994 was not founded in 2011.
    frame["FOUNDED"] = _founded(listed_from, rng)

    logger.info("Generated %d profile field(s) for %d instrument(s).",
                len(frame.columns), count)

    return frame


def _column(universe: pd.DataFrame,
            name: str,
            fallback: str) -> list[str]:
    """One string column, or a fallback when the universe lacks it."""
    if name not in universe:
        return [fallback] * len(universe)

    return [str(value) for value in universe[name]]


def _numeric(universe: pd.DataFrame,
             name: str,
             fallback: float) -> np.ndarray:
    if name not in universe:
        return np.full(len(universe), fallback)

    return np.asarray(universe[name].to_numpy(dtype=float))


def _dates(universe: pd.DataFrame,
           name: str,
           fallback: pd.Timestamp) -> list[pd.Timestamp]:
    if name not in universe:
        return [fallback] * len(universe)

    return [pd.Timestamp(value) if pd.notna(value) else pd.NaT
            for value in universe[name]]


def _employees(caps: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """Headcount, scaled to market cap.

    Drawn around what the size implies rather than independently: a mega-cap
    with forty staff is the sort of detail that makes a demo dataset obviously
    fake at exactly the moment somebody is looking closely.
    """
    implied = (caps / 1e6) * EMPLOYEES_PER_MILLION_CAP
    drawn = implied * rng.lognormal(0.0, EMPLOYEE_DISPERSION, size=len(caps))

    return np.maximum(MIN_EMPLOYEES, np.round(drawn)).astype(int)


def _founded(listed_from: list[pd.Timestamp],
             rng: np.random.Generator) -> list[int]:
    """A founding year, always before the listing year."""
    years = []

    for listed in listed_from:
        listing_year = (pd.Timestamp(listed).year if pd.notna(listed)
                        else EARLIEST_FOUNDED + 100)
        latest = listing_year - MIN_YEARS_BEFORE_LISTING
        earliest = min(EARLIEST_FOUNDED, latest)

        years.append(int(rng.integers(earliest, latest + 1)))

    return years


def _frequencies(yields: np.ndarray,
                 rng: np.random.Generator) -> list[str]:
    """How often a name pays, consistent with whether it pays at all."""
    drawn = rng.choice(DIVIDEND_FREQUENCIES, size=len(yields),
                       p=[0.15, 0.25, 0.60])

    return [NO_DIVIDEND if yield_ <= 0 else str(frequency)
            for yield_, frequency in zip(yields, drawn, strict=True)]


def _next_earnings(as_of: pd.Timestamp,
                   count: int,
                   rng: np.random.Generator) -> list[pd.Timestamp]:
    """The next reporting date, forward of the panel's end."""
    offsets = rng.integers(MIN_DAYS_TO_EARNINGS, MAX_DAYS_TO_EARNINGS,
                           size=count)

    return [as_of + pd.Timedelta(days=int(offset)) for offset in offsets]


def _city(country: str,
          rng: np.random.Generator) -> str:
    """A headquarters in the country the instrument lists in."""
    cities = CITIES.get(country, DEFAULT_CITIES)

    return f"{rng.choice(cities)}, {country}"


# --- identifier construction ------------------------------------------------


def _isin(country: str,
          identifier: str,
          rng: np.random.Generator) -> str:
    """A structurally valid ISIN: country, nine characters, check digit."""
    prefix = country if len(country) == 2 else "US"
    body = "".join(rng.choice(list(_ALPHANUMERIC), size=9))

    return f"{prefix}{body}{_luhn_check(prefix + body)}"


def _cusip(rng: np.random.Generator) -> str:
    """Eight characters and a check digit."""
    body = "".join(rng.choice(list(_ALPHANUMERIC), size=8))

    return f"{body}{_cusip_check(body)}"


def _sedol(rng: np.random.Generator) -> str:
    """Six characters and a weighted check digit."""
    body = "".join(rng.choice(list(_SEDOL_ALPHABET), size=6))

    return f"{body}{_sedol_check(body)}"


def _figi(rng: np.random.Generator) -> str:
    """A FIGI-shaped identifier: BBG, eight consonants/digits, check digit."""
    body = "".join(rng.choice(list(_SEDOL_ALPHABET), size=8))

    return f"BBG{body}{rng.integers(0, 10)}"


def _value_of(character: str) -> int:
    """A character's numeric value: digits as themselves, letters from 10."""
    return int(character) if character.isdigit() else ord(character) - 55


def _luhn_check(body: str) -> int:
    """The ISIN check digit.

    Every character is expanded to its numeric value first, then the Luhn
    doubling runs over the resulting digit string — not over the original
    characters. Getting that order wrong produces a check digit that looks
    plausible and fails every validator.
    """
    digits = "".join(str(_value_of(character)) for character in body)
    total = 0

    for position, digit in enumerate(reversed(digits)):
        value = int(digit)

        if position % 2 == 0:
            value *= 2

            if value > 9:
                value -= 9

        total += value

    return (10 - total % 10) % 10


def _cusip_check(body: str) -> int:
    """The CUSIP check digit: alternate doubling over character values."""
    total = 0

    for position, character in enumerate(body):
        value = _value_of(character)

        if position % 2 == 1:
            value *= 2

        total += value // 10 + value % 10

    return (10 - total % 10) % 10


def _sedol_check(body: str) -> int:
    """The SEDOL check digit: fixed weights over character values."""
    total = sum(_value_of(character) * weight
                for character, weight in zip(body, _SEDOL_WEIGHTS,
                                             strict=False))

    return (10 - total % 10) % 10

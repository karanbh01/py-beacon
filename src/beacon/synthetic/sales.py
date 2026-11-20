# src/beacon/synthetic/sales.py
"""
Synthetic revenue: each name's trailing-twelve-month sales, quarter by
quarter, for `beacon.synthetic.features`.

A name starts from its market cap over its sector's price-to-sales and grows
at a yearly rate drawn around its sector's, with noise each quarter. The
growth is drawn rather than read off the price, so a name's sales growth and
its earnings growth (with a held multiple, its price's) differ.
"""
# BN-426.
import numpy as np
import pandas as pd

# Price to sales by sector, for a name's starting revenue, and each sector's
# typical yearly sales growth: technology and health care grow faster than
# utilities and energy.
SECTOR_PS = {
    "Information Technology": 6.0, "Health Care": 4.0,
    "Consumer Discretionary": 1.8, "Communication Services": 3.0,
    "Industrials": 1.8, "Consumer Staples": 1.5, "Materials": 1.5,
    "Real Estate": 5.0, "Financials": 2.5, "Energy": 1.2,
    "Utilities": 2.0,
}
SECTOR_SALES_GROWTH = {
    "Information Technology": 0.10, "Health Care": 0.07,
    "Consumer Discretionary": 0.06, "Communication Services": 0.06,
    "Industrials": 0.05, "Consumer Staples": 0.04, "Materials": 0.04,
    "Real Estate": 0.04, "Financials": 0.05, "Energy": 0.03,
    "Utilities": 0.03,
}

# How widely a name's price-to-sales sits around its sector's, and its yearly
# sales growth around its sector's; and how far a quarter's revenue strays from
# its path, as lognormal sigmas.
PS_DISPERSION = 0.35
SALES_GROWTH_DISPERSION = 0.06
SALES_NOISE = 0.03

DAYS_A_YEAR = 365.25

# A name's sales path: its trailing-year revenue at the path's start, and its
# yearly growth.
SalesPath = tuple[float, float]


def sales_path(sector: str,
               market_cap: float,
               rng: np.random.Generator) -> SalesPath:
    """A name's starting trailing-year revenue and its yearly growth."""
    ps = SECTOR_PS.get(sector, 2.0) * rng.lognormal(0.0, PS_DISPERSION)
    growth = SECTOR_SALES_GROWTH.get(sector, 0.05) + rng.normal(0.0, SALES_GROWTH_DISPERSION)

    return float(market_cap / ps), float(growth)


def revenue_on(path: SalesPath,
               years: float,
               rng: np.random.Generator) -> float:
    """A name's trailing-year revenue *years* along its sales path."""
    start, growth = path

    return float(start * np.exp(growth * years + rng.normal(0.0, SALES_NOISE)))


def stored_sales(stored: pd.DataFrame) -> dict[str, tuple[SalesPath, pd.Timestamp]]:
    """Each name's sales path as its stored revenue rows show it, from their
    latest: its latest revenue, the yearly growth from its first to its
    latest, and the period the latest describes."""
    rows = stored.loc[stored["FIELD"] == "revenue"]
    paths: dict[str, tuple[SalesPath, pd.Timestamp]] = {}

    for identifier, found in rows.groupby("IDENTIFIER"):
        held = found.assign(PERIOD=found["DETAIL"].map(period_of)).sort_values("PERIOD")
        first, last = held.iloc[0], held.iloc[-1]
        years = (last["PERIOD"] - first["PERIOD"]).days / DAYS_A_YEAR
        growth = (float(np.log(last["VALUE"] / first["VALUE"])) / years
                  if years > 0 and first["VALUE"] > 0 and last["VALUE"] > 0 else 0.0)
        paths[str(identifier)] = ((float(last["VALUE"]), growth), last["PERIOD"])

    return paths


def period_of(detail: str) -> pd.Timestamp:
    """The period a quarterly row describes, from its DETAIL
    (``period ending 2024-03-31, reported 2024Q1``)."""
    return pd.Timestamp(detail.split("period ending ", maxsplit=1)[1].split(",", maxsplit=1)[0])

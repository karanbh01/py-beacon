# src/beacon/backtest/withholding.py
"""
Withholding tax on a fund's dividends, by its domicile and the dividend's
source.

A fund's domicile sets the tax it suffers on each market's dividends, because
treaties are between countries and not every fund can claim them. On US
dividends an Irish fund suffers 15%, a Luxembourg SICAV usually 30% (the
treaty's limitation on benefits keeps most from its 15% rate) and a US fund
nothing. A net total return index assumes one rate for every holder, so the
difference is part of a fund's tracking difference against it.

**The rate for a dividend** is, in order:

1. the vehicle's own `withholding_rates` for the dividend's source country,
2. the table below, for the vehicle's domicile and the source country,
3. the run's flat `withholding_tax_rate` (zero unless set).

**The table** holds only rates that are well established for funds, since a
wrong rate is worse than an honest fallback:

| Dividend from | Ireland | Luxembourg | United Kingdom | United States |
| --- | --- | --- | --- | --- |
| United States | 15% | 30% | 15% | 0% |
| United Kingdom | 0% | 0% | 0% | 0% |
| Ireland | 0% | | | |

The United Kingdom withholds nothing on dividends, and an Irish fund is exempt
from Irish dividend withholding tax. Other markets' rates for a fund depend
on whether it can claim a treaty and on reclaims, so they are left to
`withholding_rates` and the flat rate.

**The source country** is the company's own: a `COUNTRY_DOMICILE` or
`COUNTRY` reference column when the data has one, otherwise the country of
the exchange it lists on (XNYS is the United States, XLON the United Kingdom).
Countries are ISO 3166 two-letter codes.
"""
# BN-271, phase 8 of decisions/0006. Decided with Karan on 2026-10-11: a
# built-in table that a vehicle can override, the source from a country field
# or the listing exchange, the domicile from the vehicle preset. Rates checked
# on 2026-10-11; secondary sources disagreed on most other markets, so only
# these went in.
from collections.abc import Mapping
from typing import Any

IRELAND = "IE"
LUXEMBOURG = "LU"
UNITED_KINGDOM = "GB"
UNITED_STATES = "US"

# (fund domicile, dividend source) -> the share withheld.
TREATY_RATES: dict[tuple[str, str], float] = {
    (IRELAND, UNITED_STATES): 0.15,
    (LUXEMBOURG, UNITED_STATES): 0.30,
    (UNITED_KINGDOM, UNITED_STATES): 0.15,
    (UNITED_STATES, UNITED_STATES): 0.0,
    (IRELAND, UNITED_KINGDOM): 0.0,
    (LUXEMBOURG, UNITED_KINGDOM): 0.0,
    (UNITED_KINGDOM, UNITED_KINGDOM): 0.0,
    (UNITED_STATES, UNITED_KINGDOM): 0.0,
    (IRELAND, IRELAND): 0.0,
}

# The country of each exchange a listing names, by its MIC.
EXCHANGE_COUNTRIES: dict[str, str] = {
    "XNYS": "US", "XNAS": "US", "ARCX": "US", "BATS": "US", "XASE": "US",
    "XLON": "GB", "XDUB": "IE", "XETR": "DE", "XFRA": "DE", "XPAR": "FR",
    "XAMS": "NL", "XBRU": "BE", "XLIS": "PT", "XMIL": "IT", "XMAD": "ES",
    "XSWX": "CH", "XWBO": "AT", "XSTO": "SE", "XCSE": "DK", "XHEL": "FI",
    "XOSL": "NO", "XTSE": "CA", "XASX": "AU", "XTKS": "JP", "XHKG": "HK",
    "XSES": "SG", "XKRX": "KR", "XTAI": "TW", "XSHG": "CN", "XSHE": "CN",
    "XBOM": "IN", "XNSE": "IN", "XJSE": "ZA", "BVMF": "BR", "XMEX": "MX",
}

# Reference columns that name a company's country, best first: where it is
# incorporated decides its dividends' withholding, not where it lists.
COUNTRY_COLUMNS = ("COUNTRY_DOMICILE", "COUNTRY")


def withholding_rate(domicile: str | None,
                     source: str | None,
                     overrides: Mapping[str, float] | None,
                     fallback: float) -> float:
    """The share of a dividend from *source* withheld from a fund domiciled
    in *domicile*: the fund's own override for the source, then the table,
    then *fallback*."""
    if source is not None and overrides and source in overrides:
        return float(overrides[source])

    if domicile is not None and source is not None:
        rate = TREATY_RATES.get((domicile, source))

        if rate is not None:
            return rate

    return fallback


def source_country(fetcher: Any,
                   identifier: str,
                   date: Any) -> str | None:
    """Where *identifier*'s dividends come from: its country column, else its
    listing exchange's country, else None."""
    read = getattr(fetcher, "fetch_classification", None)

    if read is None:
        return None

    for column in COUNTRY_COLUMNS:
        country = read(identifier, date, column)

        if country:
            return str(country).upper()

    exchange = read(identifier, date, "EXCHANGE")

    return EXCHANGE_COUNTRIES.get(str(exchange).upper()) if exchange else None


def checked_rates(rates: Mapping[str, float] | None) -> dict[str, float] | None:
    """A vehicle's withholding rates by source country, checked: each a share
    from 0 up to 1, each key a two-letter country code (upper-cased)."""
    if rates is None:
        return None

    checked = {}

    for country, rate in rates.items():
        if len(country) != 2 or not country.isalpha():
            raise ValueError(f"withholding rates are keyed by two-letter country "
                             f"codes, got {country!r}.")

        if not 0.0 <= rate < 1.0:
            raise ValueError(f"a withholding rate is a share from 0 up to 1, got "
                             f"{rate!r} for {country}.")

        checked[country.upper()] = float(rate)

    return checked

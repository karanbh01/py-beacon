# src/beacon/analysis/performance/futures_figures.py
"""
How a fund holding index futures traded them, and the basis risk of holding
them.

**Trading**, by why each trade was made (opened, rolled, resized): how many
trades, the contracts and notional traded, the cost, the cost per contract
and in basis points of the notional, and the notional traded a year over the
fund's mean assets.

**The basis** each day: the futures' price over the level of the index they
are on, less one, which splits into the carry (fair value over the level)
and the mispricing (the price over fair value).

**Basis risk** splits the gap between the futures' daily return and the
fund's index's into two: the pricing basis, the futures against their own
index (carry and any mispricing), and the underlying mismatch, their index
against the fund's, which brings tracking error even at fair value with no
costs. A day the position rolled is left out of the futures' returns, since
its price moved to another contract.
"""
# BN-409. As agreed with Karan on 2026-10-27.
import numpy as np
import pandas as pd

BASIS_POINTS = 10_000.0

# The parts of basis risk, as a table gives them.
PRICING = "pricing_basis"
MISMATCH = "underlying_mismatch"
TOTAL = "total"


def futures_trading(trades: pd.DataFrame,
                    nav: pd.Series,
                    periods_per_year: int) -> pd.DataFrame:
    """The futures trading by reason, between rolls, and all of it: the
    trades, the contracts and notional traded, the cost, per contract and in
    basis points of the notional, and the notional a year over the mean NAV."""
    groups = {reason: trades[trades["reason"] == reason]
              for reason in sorted(set(trades["reason"]))}
    groups["between_rolls"] = trades[trades["reason"] != "roll"]
    groups["all"] = trades
    years = len(nav) / float(periods_per_year)
    assets = float(nav.mean()) if len(nav) else 0.0
    rows = {}

    for name, some in groups.items():
        contracts = float(some["traded"].sum())
        notional = float(some["notional"].sum())
        cost = float(some["cost"].sum())
        rows[name] = {
            "trades": len(some), "contracts": contracts, "notional": notional, "cost": cost,
            "cost_per_contract": cost / contracts if contracts > 0 else np.nan,
            "cost_bps": cost / notional * BASIS_POINTS if notional > 0 else np.nan,
            "turnover": notional / (assets * years) if assets > 0 and years > 0 else np.nan}

    return pd.DataFrame.from_dict(rows, orient="index")


def daily_basis(days: pd.DataFrame) -> pd.DataFrame:
    """Each day's basis (the price over the level, less one), its carry
    (fair value over the level) and its mispricing (the price over fair
    value)."""
    return pd.DataFrame({"basis": days["price"] / days["level"] - 1.0,
                         "carry": days["fair_value"] / days["level"] - 1.0,
                         "mispricing": days["price"] / days["fair_value"] - 1.0},
                        index=days.index)


def basis_summary(days: pd.DataFrame) -> pd.DataFrame:
    """The basis, carry and mispricing in basis points: their mean, standard
    deviation, and lowest and highest, a row each."""
    found = daily_basis(days) * BASIS_POINTS

    return pd.DataFrame({"mean_bps": found.mean(), "volatility_bps": found.std(),
                         "lowest_bps": found.min(), "highest_bps": found.max()})


def basis_returns(days: pd.DataFrame) -> pd.DataFrame:
    """Each day's pricing basis (the futures' return less their index's),
    underlying mismatch (their index's less the fund's) and both, on days the
    contract held did not change."""
    held = days["expiry"].eq(days["expiry"].shift(1))
    futures = days["price"].pct_change()
    underlying = days["level"].pct_change()
    target = days["index_level"].pct_change()
    table = pd.DataFrame({PRICING: futures - underlying, MISMATCH: underlying - target,
                          TOTAL: futures - target}, index=days.index)

    return table[held].dropna()


def basis_risk(days: pd.DataFrame,
               periods_per_year: int) -> pd.DataFrame:
    """For the pricing basis, the underlying mismatch and both: the mean a
    year and the tracking error each causes (its daily standard deviation,
    annualised), and the days measured."""
    returns = basis_returns(days)
    root = float(np.sqrt(periods_per_year))

    return pd.DataFrame({"mean_a_year": returns.mean() * periods_per_year,
                         "tracking_error": returns.std() * root,
                         "days": returns.count()})

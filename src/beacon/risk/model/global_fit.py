# src/beacon/risk/model/global_fit.py
"""
A factor model for names in many markets: country, sector and currency
factors beside the styles.

## The factors

Each name belongs to one country and one sector, so it has an exposure of 1
to that country's factor and that sector's, and 0 to the others. A country
with too few names to estimate a factor of its own joins the rest of its
region in one factor, named after the region (``Europe (other)``). The
market factor stays, and each country's and each sector's return is measured
against it: the countries' returns, weighted by their market caps, add up to
zero, and so do the sectors'. Without that the market, the countries and the
sectors would each explain the same moves (every name has a 1 in the market
column and a 1 in one country column) and the split between them would be
arbitrary.

A name priced in another currency than the base has an exposure of 1 to that
currency's factor, whose return is the currency's own return against the
base, read from the rates rather than fitted. The other factors are fitted on
the names' returns in their own currencies:

    1 + r_base = (1 + r_local) (1 + r_currency)

so that a currency's move is not shared out between it and the countries that
use it.

## The fit

Each period's local returns are regressed on the exposures by weighted least
squares, each name weighted by the square root of its market cap, under the
constraints above. The constraints are met by writing one country's return
(and one sector's) in terms of the others', so the regression has only free
factors to fit. The factor covariance is that of the fitted and the currency
returns together, so it carries how currencies and equity factors move with
one another; each name's specific variance is that of what its base-currency
return has left once the factors are taken out.

## Weekly returns

Markets close at different times, so a day's return in Tokyo and the same
day's in New York see different news, and daily returns understate how much
they move together. Over a week the difference is a small part of the
period, so a global model is fitted on weekly returns (`weekly_returns`).
"""
# BN-419. Agreed with Karan on 2026-10-31.
import logging

import numpy as np
import pandas as pd

from ...exceptions import CalculationError
from .factor import MARKET_FACTOR, NEGLIGIBLE_SPREAD, FactorRiskModel, _explained_variance

logger = logging.getLogger(__name__)

COUNTRY = "country"
SECTOR = "sector"
CURRENCY = "currency"

# A year of weekly returns.
WEEKS_PER_YEAR = 52

# Weeks end on Friday's close.
WEEK = "W-FRI"

# The fewest names a country needs for a factor of its own.
MINIMUM_NAMES = 10

UNKNOWN = "Unknown"


def factor_name(kind: str,
                group: str) -> str:
    """The factor for *group* of *kind*: ``country:US``, ``sector:Energy``."""
    return f"{kind}:{group}"


def assign_groups(values: pd.Series,
                  regions: pd.Series | None = None,
                  minimum_names: int = MINIMUM_NAMES) -> pd.Series:
    """Each name's group: its value, or for a value fewer than
    *minimum_names* names share, its region's ``<region> (other)``.

    Args:
        values: Each name's country (or other field), by name; missing as
            ``Unknown``.
        regions: Each name's region, by name; None merges nothing.
        minimum_names: The fewest names a group keeps to itself.
    """
    groups = values.astype(object).where(values.notna(), UNKNOWN).astype(str)

    if regions is None:
        return groups

    counts = groups.value_counts()
    thin = groups.map(counts) < minimum_names
    region = regions.reindex(groups.index)
    merged = region.astype(object).where(region.notna(), UNKNOWN).astype(str) + " (other)"

    return groups.where(~thin, merged)


def group_dummies(groups: pd.Series,
                  kind: str) -> pd.DataFrame:
    """A 1 in each name's group's column and 0 elsewhere, the columns named
    by `factor_name` and sorted."""
    names = sorted(groups.unique())
    columns = [factor_name(kind, group) for group in names]
    codes = pd.Categorical(groups, categories=names).codes
    table = np.zeros((len(groups), len(names)))
    table[np.arange(len(groups)), codes] = 1.0

    return pd.DataFrame(table, index=groups.index, columns=columns)


def weekly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily returns compounded into weeks ending on Friday, each labelled by
    its Friday. A week in which a name missed a day is missing for it."""
    growth = np.log1p(daily.astype(float))
    summed = growth.resample(WEEK).sum(min_count=1)
    sessions = daily.resample(WEEK).size()
    complete = daily.notna().resample(WEEK).sum().eq(sessions, axis=0)

    return np.expm1(summed).where(complete)


def fit_global_model(returns: pd.DataFrame,
                     exposures: pd.DataFrame,
                     constrained: list[list[str]],
                     caps: pd.Series | None = None,
                     currencies: pd.Series | None = None,
                     currency_returns: pd.DataFrame | None = None,
                     periods_per_year: int = WEEKS_PER_YEAR) -> FactorRiskModel:
    """Fit a factor model with constrained group factors and observed
    currency factors.

    Args:
        returns: Period returns in the base currency, dates on the index and
            names on the columns. Rows with any missing value are dropped.
        exposures: The names' exposures to the fitted factors (styles,
            country and sector dummies), names on the index; a market column
            is added when absent.
        constrained: Each set of columns whose returns, weighted by their
            market caps, add up to zero: the countries, the sectors.
        caps: Each name's market cap, for the regression's weights and the
            constraints'; None weights every name the same.
        currencies: Each name's currency, by name, for those priced in
            another than the base; None, or a name left out, has none.
        currency_returns: Each currency's return against the base each
            period, one column per currency.
        periods_per_year: The periods in a year: 52 for weekly returns.

    Returns:
        FactorRiskModel: Its exposures are the fitted factors' and then one
        column per currency, ``currency:<code>``.

    Raises:
        CalculationError: If the returns do not cover every name, a name's
            currency has no returns, or fewer than 2 periods are complete.
    """
    names = list(exposures.index)
    missing = sorted(set(names) - set(returns.columns))

    if missing:
        raise CalculationError("Global risk model",
                               f"the returns do not cover every exposed name: {missing}.")

    loadings = exposures if MARKET_FACTOR in exposures.columns else pd.concat(
        [pd.DataFrame({MARKET_FACTOR: 1.0}, index=exposures.index), exposures], axis=1)
    held = _currency_exposures(names, currencies, currency_returns)
    panel = returns[names]
    fx = _currency_panel(held, currency_returns, panel.index)
    complete = panel.notna().all(axis=1) & fx.notna().all(axis=1)
    panel, fx = panel[complete], fx[complete]

    if len(panel) < 2:
        raise CalculationError("Global risk model",
                               f"at least 2 complete periods are needed, found {len(panel)}.")

    base = panel.to_numpy(dtype=float)
    moves = fx.to_numpy(dtype=float)
    currency_of = held.to_numpy(dtype=float)
    local = (1.0 + base) / (1.0 + moves @ currency_of.T) - 1.0

    weights = _regression_weights(caps, names)
    fitted = _constrained_fit(local, loadings, constrained, weights, caps)

    design = np.hstack([loadings.to_numpy(dtype=float), currency_of])
    factor_returns = np.hstack([fitted, moves])
    residuals = base - factor_returns @ design.T
    columns = [*loadings.columns, *held.columns]
    covariance = np.atleast_2d(np.cov(factor_returns, rowvar=False, ddof=1)) * periods_per_year
    explained = _explained_variance(base, residuals)

    logger.info(f"Fitted a global model of {len(columns)} factors over {len(panel)} periods "
                f"and {len(names)} names: R² {explained:.4f}.")

    return FactorRiskModel(
        exposures=pd.DataFrame(design, index=names, columns=columns),
        factor_covariance=pd.DataFrame(covariance, index=columns, columns=columns),
        specific_variance=pd.Series(residuals.var(axis=0, ddof=1) * periods_per_year,
                                    index=names, name="specific_variance"),
        factor_returns=pd.DataFrame(factor_returns, index=panel.index, columns=columns),
        r_squared=explained,
        periods_per_year=periods_per_year)


def _currency_exposures(names: list[str],
                        currencies: pd.Series | None,
                        currency_returns: pd.DataFrame | None) -> pd.DataFrame:
    """A 1 in each name's currency's column, for the names priced in another
    currency than the base."""
    if currencies is None:
        return pd.DataFrame(index=names)

    codes = currencies.reindex(names).dropna().astype(str)
    available = set(currency_returns.columns) if currency_returns is not None else set()
    unpriced = sorted(set(codes) - available)

    if unpriced:
        raise CalculationError("Global risk model",
                               f"no returns for the currencies {unpriced}.")

    return group_dummies(codes, CURRENCY).reindex(names).fillna(0.0)


def _currency_panel(held: pd.DataFrame,
                    currency_returns: pd.DataFrame | None,
                    dates: pd.Index) -> pd.DataFrame:
    """The returns of the currencies in *held*, on *dates*."""
    codes = [column.split(":", 1)[1] for column in held.columns]

    if not codes or currency_returns is None:
        return pd.DataFrame(index=dates)

    found = currency_returns[codes].reindex(dates)
    found.columns = held.columns

    return found


def _regression_weights(caps: pd.Series | None,
                        names: list[str]) -> np.ndarray:
    """The square root of each name's market cap; 1 each without caps."""
    if caps is None:
        return np.ones(len(names))

    values = caps.reindex(names).fillna(0.0).to_numpy(dtype=float)

    return np.asarray(np.sqrt(np.clip(values, 0.0, None)), dtype=np.float64)


def _constrained_fit(local: np.ndarray,
                     loadings: pd.DataFrame,
                     constrained: list[list[str]],
                     weights: np.ndarray,
                     caps: pd.Series | None) -> np.ndarray:
    """Each period's factor returns, periods by factors, under the
    constraints."""
    design = loadings.to_numpy(dtype=float)
    reduction = _reduction(loadings, constrained, caps)
    scaled = np.sqrt(weights)[:, None]
    free, *_ = np.linalg.lstsq(scaled * (design @ reduction), scaled * local.T, rcond=None)

    return np.asarray((reduction @ free).T, dtype=np.float64)


def _reduction(loadings: pd.DataFrame,
               constrained: list[list[str]],
               caps: pd.Series | None) -> np.ndarray:
    """The matrix taking the free factors to all of them: within each
    constrained set, the largest group's return is minus the others',
    weighted by their caps over its cap."""
    columns = list(loadings.columns)
    dependent = {}

    for block in constrained:
        sizes = _group_sizes(loadings[block], caps)

        if len(block) < 2 or sizes.max() <= NEGLIGIBLE_SPREAD:
            continue

        largest = str(sizes.idxmax())
        dependent[largest] = (sizes.drop(largest) / sizes[largest]).to_dict()

    free = [column for column in columns if column not in dependent]
    reduction = np.zeros((len(columns), len(free)))

    for position, column in enumerate(free):
        reduction[columns.index(column), position] = 1.0

    for column, shares in dependent.items():
        for other, share in shares.items():
            reduction[columns.index(column), free.index(other)] = -share

    return reduction


def _group_sizes(dummies: pd.DataFrame,
                 caps: pd.Series | None) -> pd.Series:
    """Each group's total market cap; its count of names without caps."""
    if caps is None:
        return pd.Series(dummies.sum(axis=0))

    values = caps.reindex(dummies.index).fillna(0.0).to_numpy(dtype=float)

    return pd.Series(values @ dummies.to_numpy(dtype=float), index=dummies.columns)

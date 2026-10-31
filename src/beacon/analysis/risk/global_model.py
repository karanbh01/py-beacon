# src/beacon/analysis/risk/global_model.py
"""
The global factor model of a run whose names span more than one country:
the styles, with country, sector and currency factors beside them, fitted on
weekly returns (see `beacon.risk.model.global_fit`).

At each rebalance the model is fitted on the *weeks* whole weeks before it,
each ending on a Friday, for the names held whose returns cover every one of
them. Each name's country is its reference field (its domicile by default)
as it stood on the rebalance, its sector likewise, and its currency the one
its closes were converted from. Turned off, weekly returns give way to the
daily returns of the model's window, and countries, sectors or currencies to
no factor at all.
"""
# BN-419. Agreed with Karan on 2026-10-31.
from collections.abc import Callable

import pandas as pd

from ...data.base import as_of_position
from ...exceptions import CalculationError
from ...risk.model import FactorRiskModel
from ...risk.model.global_fit import (
    COUNTRY,
    SECTOR,
    WEEKS_PER_YEAR,
    assign_groups,
    fit_global_model,
    group_dummies,
    weekly_returns,
)
from ..data import RunData
from ..panels import NAME_RETURNS, REFERENCE

IDENTIFIER = "IDENTIFIER"
REGION_FIELD = "REGION"
SECTOR_FIELD = "SECTOR"
CURRENCY_FIELD = "CURRENCY"
CURRENCIES = "currencies"
RATES = "rates"

Loadings = Callable[[pd.Timestamp, list[str]], pd.DataFrame]
Caps = Callable[[pd.Timestamp, list[str]], pd.Series]


class GlobalModels:
    """A run's returns and the reference it is grouped by, read once for
    every rebalance's global model.

    Args:
        run: The run's data, with its name returns, reference, currencies
            and rates.
        country_field: The reference field each name's country is read from,
            or None for no country factors.
        minimum_names: The fewest names a country needs for a factor of its
            own; one with fewer joins its region's.
        sectors: Whether sectors are factors.
        currencies: Whether currencies are factors.
        weekly: Whether the model is fitted on weekly returns rather than
            daily.
    """

    def __init__(self,
                 run: RunData,
                 country_field: str | None,
                 minimum_names: int,
                 sectors: bool,
                 currencies: bool,
                 weekly: bool):
        self.run = run
        self.country_field = country_field
        self.minimum_names = minimum_names
        self.sectors = sectors
        self.currencies = currencies
        self.weekly = weekly
        self._returns: pd.DataFrame | None = None
        self._rates: pd.DataFrame | None = None

    @property
    def periods_per_year(self) -> int:
        """The periods in the model's year."""
        return WEEKS_PER_YEAR if self.weekly else self.run.periods_per_year

    def reference_on(self,
                     day: pd.Timestamp) -> pd.DataFrame:
        """The names' reference fields at the rebalance on *day*, by name."""
        reference = self.run.tables[REFERENCE]
        rows = reference.loc[reference.index == day]

        return rows.set_index(rows[IDENTIFIER].astype(str))

    def spans_countries(self,
                        day: pd.Timestamp,
                        names: list[str]) -> bool:
        """Whether *names* were in more than one country on *day*."""
        if self.country_field is None:
            return False

        reference = self.reference_on(day)

        if self.country_field not in reference.columns:
            return False

        return bool(reference[self.country_field].reindex(names).dropna().nunique() > 1)

    def periods_before(self,
                       day: pd.Timestamp) -> pd.DataFrame:
        """The returns of the periods that ended before *day*: whole weeks,
        or sessions."""
        returns = self._periods()
        sessions = self._sorted()
        last = as_of_position(sessions.index, day - pd.Timedelta(days=1))

        if last is None:
            return returns.iloc[:0]

        return returns[returns.index <= sessions.index[last]]

    def estimate(self,
                 day: pd.Timestamp,
                 names: list[str],
                 periods: int,
                 styles: Loadings | None,
                 caps: Caps | None) -> tuple[FactorRiskModel, list[str]]:
        """The global model at the rebalance on *day* over its last *periods*
        periods, and the names held left out of it for want of returns.

        Raises:
            CalculationError: If fewer than *periods* periods come before
                *day*, or no name's returns cover them.
        """
        window = self.periods_before(day)

        if len(window) < periods:
            unit = "weeks" if self.weekly else "sessions"
            raise CalculationError("Risk model", f"{len(window)} {unit} of returns before "
                                                 f"{day.date()}, fewer than the {periods} it "
                                                 f"is estimated on")

        window = window.iloc[len(window) - periods:].reindex(columns=names)
        complete = window.notna().all(axis=0).to_numpy()
        covered = [name for name, whole in zip(names, complete, strict=True) if whole]
        uncovered = [name for name in names if name not in covered]

        if not covered:
            raise CalculationError("Risk model", f"no name held on {day.date()} has returns "
                                                 f"over the {periods} periods before it")

        loadings, constrained = self._loadings(day, covered, styles)
        currency_of, currency_returns = self._currencies(covered, window.index)
        weights = caps(day, covered) if caps is not None else None
        model = fit_global_model(window[covered], loadings, constrained, weights, currency_of,
                                 currency_returns, periods_per_year=self.periods_per_year)

        return model, uncovered

    def _loadings(self,
                  day: pd.Timestamp,
                  names: list[str],
                  styles: Loadings | None) -> tuple[pd.DataFrame, list[list[str]]]:
        """The styles' and the groups' exposures of *names* on *day*, and the
        sets of group columns constrained to add up to zero."""
        reference = self.reference_on(day).reindex(names)
        parts = []
        constrained: list[list[str]] = []

        if styles is not None:
            parts.append(styles(day, names).reindex(names).fillna(0.0))

        for kind, field, regions in ((COUNTRY, self.country_field, REGION_FIELD),
                                     (SECTOR, SECTOR_FIELD if self.sectors else None, None)):
            if field is None or field not in reference.columns:
                continue

            region = (reference[regions] if regions in reference.columns
                      and regions != field else None)
            groups = assign_groups(reference[field], region, self.minimum_names)
            dummies = group_dummies(groups, kind)

            if dummies.shape[1] > 1:
                parts.insert(len(constrained), dummies)
                constrained.append(list(dummies.columns))

        loadings = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=names)

        return loadings, constrained

    def _currencies(self,
                    names: list[str],
                    dates: pd.Index) -> tuple[pd.Series | None, pd.DataFrame | None]:
        """Each of *names*' currency other than the base, and the currencies'
        returns over *dates*."""
        if not self.currencies:
            return None, None

        quoted = self.run.tables.get(CURRENCIES)

        if quoted is None or quoted.empty:
            return None, None

        codes = quoted[CURRENCY_FIELD].astype(str)
        codes.index = codes.index.astype(str)
        codes = codes.reindex(names).dropna()
        codes = codes[codes != self.run.currency.upper()]

        return codes, self._rate_returns().reindex(dates)

    def _sorted(self) -> pd.DataFrame:
        """The run's daily name returns, in date order."""
        return self.run.tables[NAME_RETURNS].sort_index()

    def _periods(self) -> pd.DataFrame:
        """The name returns of every period: weekly, compounded once."""
        if self._returns is None:
            daily = self._sorted()
            self._returns = weekly_returns(daily) if self.weekly else daily

        return self._returns

    def _rate_returns(self) -> pd.DataFrame:
        """Each currency's return against the base each period."""
        if self._rates is None:
            rates = self.run.tables.get(RATES, pd.DataFrame())
            daily = rates.sort_index().pct_change(fill_method=None).iloc[1:]
            self._rates = weekly_returns(daily) if self.weekly else daily

        return self._rates

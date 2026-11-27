# src/beacon/analysis/risk/risk_store.py
"""
Each name's expected risk and its risk model exposures at each rebalance,
kept beside the run, so a name's profile reads them rather than fitting the
risk model at every rebalance again.

    rebalance_risk(over_time)                 # worked out
    keep(tables, factors, settings, folder)   # after working them out
    restore(factors, settings, folder)        # before, on a stored run

Each table has a row for each name at each rebalance with the returns before
it its model needs. ``name_risk`` has the name's contribution to the fund's
expected volatility and its share of it (``contribution/fund``,
``share/fund``), and to the fund's expected tracking error against each
comparator the run kept weights for, from its active weight
(``contribution/index``, ``share/index``, ...), each as
`RiskOverTime.contributions` gives it, for each name a portfolio held.
``name_exposures`` has the model's exposure of each name it covers to each
factor, the market's first (``exposure/market``, ``exposure/value``, ...), and
in a global model the country, sector and currency it put the name in
(``country``, ``sector``, ``currency``, blank for a name in the portfolio's
own currency). They are kept keyed by the factors and the risk model's
settings, so a run asked about another model works them out again.
"""
# BN-470 (Karan, 2026-11-26): kept with the run, as the factor work is
# (BN-399), so Drilldown does not wait for a model fit at every rebalance.
# BN-471 added the exposures.
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ...exceptions import CalculationError
from ...risk.model import FactorRiskModel
from ...risk.model.global_fit import COUNTRY, CURRENCY, SECTOR
from ...strategy.factors import Factor
from ..panels import read_manifest, read_panels, write_panels
from . import factor_store
from .contributions import PORTFOLIOS, RiskOverTime
from .model import RiskModelSettings

FOLDER = "risk"
KEY_FILE = "risk.json"
VERSION = 2

logger = logging.getLogger(__name__)

TABLE, EXPOSURES = "name_risk", "name_exposures"
TABLES = (TABLE, EXPOSURES)
IDENTIFIER, DATE = "IDENTIFIER", "DATE"
CONTRIBUTION, SHARE = "contribution", "share"
EXPOSURE = "exposure/"
GROUPS = (COUNTRY, SECTOR, CURRENCY)


def rebalance_risk(over_time: RiskOverTime) -> dict[str, pd.DataFrame]:
    """Each name's expected risk (``name_risk``) and its model's exposures
    (``name_exposures``) at each rebalance its model can be fitted for;
    long, a row a name a rebalance. A rebalance whose model cannot be
    fitted is left out, with a warning, rather than failing the rest."""
    books = [book for book, table_name in PORTFOLIOS.items()
             if (table := over_time.run.tables.get(table_name)) is not None and not table.empty]
    comparators = [book for book in books if book != "fund"]
    risks, exposures = [], []

    for day in over_time.dates():
        # BN-482: one rebalance the model cannot be fitted for, say for want
        # of returns, failed the whole analysis after all its work.
        try:
            found = {"fund": over_time.contributions(day)}

            for comparator in comparators:
                found[comparator] = over_time.contributions(day, against=comparator)

            model = over_time.models.at(day)
        except CalculationError as error:
            logger.warning("No expected risk kept for %s: %s", day.date(), error)
            continue

        columns = {f"{part}/{book}": table[part] for book, table in found.items()
                   for part in (CONTRIBUTION, SHARE)}
        risks.append(_dated(pd.DataFrame(columns), day))
        assert isinstance(model, FactorRiskModel)
        exposures.append(_dated(grouped_exposures(model.exposures), day))

    return {TABLE: _long(risks), EXPOSURES: _long(exposures)}


def grouped_exposures(loadings: pd.DataFrame) -> pd.DataFrame:
    """A model's exposures, a column a factor, with each kind of group's
    columns read as the group each name is in."""
    grouped = {kind: [column for column in loadings.columns
                      if str(column).startswith(f"{kind}:")] for kind in GROUPS}
    found = loadings.drop(columns=[column for columns in grouped.values()
                                   for column in columns]).add_prefix(EXPOSURE)

    for kind, columns in grouped.items():
        if columns:
            dummies = loadings[columns]
            chosen = dummies.idxmax(axis=1).str.slice(len(kind) + 1).astype(object)
            found[kind] = chosen.where(dummies.max(axis=1) > 0.0, None)

    return found


def _dated(table: pd.DataFrame,
           day: pd.Timestamp) -> pd.DataFrame:
    return table.rename_axis(IDENTIFIER).reset_index().assign(**{DATE: day})


def _long(pieces: list[pd.DataFrame]) -> pd.DataFrame:
    """The rebalances' rows in one table, dated."""
    if not pieces:
        return pd.DataFrame(columns=[IDENTIFIER], index=pd.DatetimeIndex([], name=DATE))

    table = pd.concat(pieces, ignore_index=True).set_index(DATE)
    table[IDENTIFIER] = table[IDENTIFIER].astype(str)

    return table


def fingerprint(factors: list[Factor],
                settings: RiskModelSettings) -> str | None:
    """A key for the risk model *settings* fitted on *factors*; None when
    the factors cannot be keyed."""
    # Over every name, as the analysis's factors are standardised.
    factor_key = factor_store.fingerprint(factors, None)

    if factor_key is None:
        return None

    text = json.dumps({"version": VERSION, "factors": factor_key, "settings": asdict(settings)},
                      sort_keys=True, default=str)

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def keep(tables: dict[str, pd.DataFrame],
         factors: list[Factor],
         settings: RiskModelSettings,
         folder: Path) -> bool:
    """Keep *tables* under *folder*; False, keeping nothing, when the
    factors cannot be keyed or no rebalance's model was fitted."""
    key = fingerprint(factors, settings)

    if key is None or tables[TABLE].empty:
        return False

    write_panels(folder / FOLDER, tables)
    (folder / FOLDER / KEY_FILE).write_text(json.dumps({"key": key}), encoding="utf-8")

    return True


def restore(factors: list[Factor],
            settings: RiskModelSettings,
            folder: Path) -> dict[str, pd.DataFrame] | None:
    """The tables kept under *folder* for these factors and settings; None
    when none were."""
    key = fingerprint(factors, settings)
    path = folder / FOLDER / KEY_FILE

    if key is None or not path.exists() or read_manifest(folder / FOLDER) is None:
        return None

    kept: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    if kept.get("key") != key:
        return None

    tables = read_panels(folder / FOLDER, TABLES)

    for table in tables.values():
        table[IDENTIFIER] = table[IDENTIFIER].astype(str)

    return tables


__all__ = ["fingerprint", "grouped_exposures", "keep", "rebalance_risk", "restore"]

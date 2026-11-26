# src/beacon/analysis/risk/risk_store.py
"""
Each name's share of a run's expected risk at each rebalance, kept beside
the run, so a name's profile reads it rather than fitting the risk model at
every rebalance again.

    name_risk(over_time)                     # worked out
    keep(table, factors, settings, folder)   # after working it out
    restore(factors, settings, folder)       # before, on a stored run

The table has a row for each name each portfolio held at each rebalance with
the returns before it its model needs: the name's contribution to the fund's
expected volatility and its share of it (``contribution/fund``,
``share/fund``), and to the fund's expected tracking error against each
comparator the run kept weights for, from its active weight
(``contribution/index``, ``share/index``, ...), each as
`RiskOverTime.contributions` gives it. It is kept keyed by the factors and the
risk model's settings, so a run asked about another model works it out again.
"""
# BN-470 (Karan, 2026-11-26): kept with the run, as the factor work is
# (BN-399), so Drilldown does not wait for a model fit at every rebalance.
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ...strategy.factors import Factor
from ..panels import read_manifest, read_panels, write_panels
from . import factor_store
from .contributions import BOOKS, RiskOverTime
from .model import RiskModelSettings

FOLDER = "risk"
KEY_FILE = "risk.json"
VERSION = 1

TABLE = "name_risk"
IDENTIFIER, DATE = "IDENTIFIER", "DATE"
CONTRIBUTION, SHARE = "contribution", "share"


def name_risk(over_time: RiskOverTime) -> pd.DataFrame:
    """Each name's contribution to and share of the fund's expected
    volatility, and of its expected tracking error against each comparator,
    at each rebalance its model can be fitted for; long, a row a name a
    rebalance."""
    books = [book for book, table_name in BOOKS.items()
             if (table := over_time.run.tables.get(table_name)) is not None and not table.empty]
    comparators = [book for book in books if book != "fund"]
    pieces = []

    for day in over_time.dates():
        found = {"fund": over_time.contributions(day)}

        for comparator in comparators:
            found[comparator] = over_time.contributions(day, against=comparator)

        columns = {f"{part}/{book}": table[part] for book, table in found.items()
                   for part in (CONTRIBUTION, SHARE)}
        rows = pd.DataFrame(columns).rename_axis(IDENTIFIER).reset_index()
        pieces.append(rows.assign(**{DATE: day}))

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


def keep(table: pd.DataFrame,
         factors: list[Factor],
         settings: RiskModelSettings,
         folder: Path) -> bool:
    """Keep *table* under *folder*; False, keeping nothing, when the factors
    cannot be keyed or it is empty."""
    key = fingerprint(factors, settings)

    if key is None or table.empty:
        return False

    write_panels(folder / FOLDER, {TABLE: table})
    (folder / FOLDER / KEY_FILE).write_text(json.dumps({"key": key}), encoding="utf-8")

    return True


def restore(factors: list[Factor],
            settings: RiskModelSettings,
            folder: Path) -> pd.DataFrame | None:
    """The table kept under *folder* for these factors and settings; None
    when none was."""
    key = fingerprint(factors, settings)
    path = folder / FOLDER / KEY_FILE

    if key is None or not path.exists() or read_manifest(folder / FOLDER) is None:
        return None

    kept: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    if kept.get("key") != key:
        return None

    table = read_panels(folder / FOLDER, [TABLE])[TABLE]
    table[IDENTIFIER] = table[IDENTIFIER].astype(str)

    return table


__all__ = ["fingerprint", "keep", "name_risk", "restore"]

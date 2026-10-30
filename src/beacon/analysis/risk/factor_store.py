# src/beacon/analysis/risk/factor_store.py
"""
A run's factor work kept beside it, so it is worked out once: each
rebalance's exposures, the market caps they were centred and fitted with, the
groups a small group fell back to, and the daily factor returns.

    keep(analytics, folder)      # after working them out
    restore(analytics, folder)   # before, on a stored run

What is kept is keyed by the factors' definitions and the universe they were
standardised over, so a run asked about other factors works them out again.
Factors that read a Python function cannot be told apart that way and are
never kept.
"""
# BN-399: the exposures were most of every report that read them (BN-411),
# worked out again by each request.
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ...strategy.factors import Factor
from ...strategy.signals import FieldSignal, Momentum, Signal
from ..panels import read_manifest, read_panels, write_panels
from .factors import FactorAnalytics

# The analytics' own caches are read and given here, in the same package, so
# FactorAnalytics keeps no knowledge of where its work is stored.
FOLDER = "factors"
KEY_FILE = "factors.json"
VERSION = 1

EXPOSURES, CAPS, RETURNS = "exposures", "caps", "returns"
DAY, NAME, CAP = "DAY", "NAME", "CAP"


def fingerprint(factors: list[Factor],
                universe: list[str] | None) -> str | None:
    """A key for *factors* standardised over *universe*; None when one of
    them reads something that cannot be described, such as a function."""
    described = [_factor(factor) for factor in factors]

    if any(item is None for item in described):
        return None

    text = json.dumps({"version": VERSION, "factors": described, "universe": universe},
                      sort_keys=True)

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def keep(analytics: FactorAnalytics,
         folder: Path) -> bool:
    """Keep what *analytics* has worked out under *folder*; False, keeping
    nothing, when its factors cannot be keyed or it has worked out nothing."""
    key = fingerprint(analytics.factors, analytics.universe)
    worked = sorted(analytics._exposures)

    if key is None or not worked:
        return False

    exposures = pd.concat({day: analytics._exposures[day] for day in worked},
                          names=[DAY, NAME]).reset_index(NAME)
    caps = pd.concat({day: analytics._caps[day].rename(CAP) for day in worked},
                     names=[DAY, NAME]).reset_index(NAME)
    tables = {EXPOSURES: exposures, CAPS: caps.astype({NAME: str})}

    if analytics._returns is not None:
        tables[RETURNS] = analytics._returns

    write_panels(folder / FOLDER, tables)
    fallbacks = {day.isoformat(): {name: [[list(cell), list(wider)]
                                          for cell, wider in found.items()]
                                   for name, found in analytics._fallbacks[day].items()}
                 for day in worked if day in analytics._fallbacks}
    (folder / FOLDER / KEY_FILE).write_text(json.dumps({"key": key, "fallbacks": fallbacks}),
                                            encoding="utf-8")

    return True


def restore(analytics: FactorAnalytics,
            folder: Path) -> bool:
    """Give *analytics* the work kept under *folder* for its factors; False,
    leaving it to work them out, when nothing was kept for them."""
    key = fingerprint(analytics.factors, analytics.universe)
    path = folder / FOLDER / KEY_FILE

    if key is None or not path.exists() or read_manifest(folder / FOLDER) is None:
        return False

    kept: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    if kept.get("key") != key:
        return False

    tables = read_panels(folder / FOLDER)

    for day, rows in tables[EXPOSURES].groupby(level=0, sort=True):
        named = rows.set_index(NAME)
        # Built as the analytics build it, a column a factor, so its labels
        # are of the same kind.
        frame = pd.DataFrame({name: named[name] for name in analytics.names}, index=named.index)
        frame.index.name = None
        analytics._exposures[pd.Timestamp(day)] = frame

    for day, rows in tables[CAPS].groupby(level=0, sort=True):
        caps = rows.set_index(NAME)[CAP]
        caps.index.name, caps.name = None, None
        analytics._caps[pd.Timestamp(day)] = caps

    for day, found in kept.get("fallbacks", {}).items():
        analytics._fallbacks[pd.Timestamp(day)] = {
            name: {tuple(cell): tuple(wider) for cell, wider in pairs}
            for name, pairs in found.items()}

    if RETURNS in tables:
        kept_returns = tables[RETURNS]
        returns = pd.DataFrame({name: kept_returns[name] for name in kept_returns.columns},
                               index=kept_returns.index)
        returns.index.name = None
        analytics._returns = returns

    return True


def _factor(factor: Factor) -> dict[str, Any] | None:
    """A factor's definition as plain data; None when a signal cannot be
    described."""
    signals = [_signal(signal) for signal in factor.signals]

    if any(item is None for item in signals):
        return None

    return {"name": factor.name, "signals": signals, "weights": list(factor.weights),
            "transform": factor.transform, "higher_is_better": factor.higher_is_better,
            "standardise": factor.standardise, "centre": factor.centre,
            "centre_by": list(factor.centre_by), "scale_within_groups": factor.scale_within_groups,
            "returns": factor.returns}


def _signal(signal: Signal) -> dict[str, Any] | None:
    if isinstance(signal, Factor):
        return _factor(signal)

    if isinstance(signal, FieldSignal):
        field = signal.field

        return {"type": "field", "namespace": field.namespace, "name": field.name,
                "dataset": field.dataset,
                "higher_is_better": signal.higher_is_better}

    if isinstance(signal, Momentum):
        return {"type": "momentum", "lookback_days": signal.lookback_days,
                "skip_days": signal.skip_days, "higher_is_better": signal.higher_is_better}

    return None

# src/beacon/synthetic/extend.py
"""
Extend a generated store to a later date, without changing its history.

    from beacon.synthetic.extend import extend

    extend(Path("my-store"))                    # up to today
    extend(Path("my-store"), end="2026-06-30")

or from the command line, `python -m beacon.synthetic --extend my-store`.

## Why not regenerate with a later end date

The generator draws every number from one random stream sized by the whole
date range. A later end date changes every past price, and with it every
index and backtest saved against the store.

## What an extension does

It carries the market on from where the store stops:

- every listed name from its last close and share count, with the
  volatility, factor loadings, alpha and dividend yield it was generated with
  (saved beside the data; see `beacon.synthetic.state`);
- each FX pair from its last rate;
- new listings and delistings at the store's rates, timed by the same crisis
  intensity as before, with new names continuing the ticker sequence;
- dividends and split reviews on the same calendar rules, without a second
  ex-date in a month that already had one;
- features, including quarters that ended before the extension but are
  reported inside it.

The new days draw from a random stream derived from the store's seed and the
dates of the new sessions, so the same store extended to the same date always
gives the same data.

## What changes and what does not

Market and feature rows already in the store are never touched: the new rows
are appended to the files, and the bytes already there stay as they were.
Three kinds of record do change, because they describe a state rather than a
day: a name that delists gets its end date and status, a dividend whose pay
date arrives becomes paid, and each listed name's next earnings date moves
forward.

Crises are the real, dated episodes in `beacon.synthetic.regimes`, so days
after the last of them are calm, as they would be in a store generated over
the same dates.
"""
import gzip
import io
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import store
from ..data.base import MarketData, ReferenceData
from ..data.corporate_actions import ANNOUNCED, PAID, CorporateActions
from ..data.features import FeatureData
from ..index.schedule import sessions
from . import features, fx, listings, prices, profiles, regions, returns, state, universe
from .dataset import Progress

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# Return paths are simulated over at least a year and then cut to the
# extension. The shared factors are pinned to their realised variance over
# the path, which is badly conditioned over a few days and undefined over one.
MIN_SIMULATED_SPAN = pd.DateOffset(years=1)

# How far back from the store's end the recent history is kept in memory:
# enough for a quarter reported late, plus the month of returns sentiment
# reads.
HISTORY_DAYS = features.LOOKBACK_DAYS + 45

# The market file is read in pieces, so extending a large store does not
# need it all in memory at once.
READ_CHUNK_ROWS = 1_000_000

_GZIP = {"method": "gzip", "mtime": 0}

# How dates are written in a store.
DATE_FORMAT = "%Y-%m-%d"


@dataclass(frozen=True)
class Extension:
    """What an extension added.

    Attributes:
        first: The first new session, or None if there was nothing to add.
        last: The last new session, or None.
        sessions: How many sessions were added.
        listed: How many new names listed.
        delisted: How many names left.
    """
    first: str | None
    last: str | None
    sessions: int
    listed: int
    delisted: int


@dataclass(frozen=True)
class _ReferenceChanges:
    """What an extension does to the reference data.

    Attributes:
        rows: New names' records, or None if nobody listed.
        delisted: Each name that left, and its last date.
        next_earnings: Each listed name's next earnings date.
    """
    rows: pd.DataFrame | None
    delisted: dict[str, str]
    next_earnings: dict[str, str]


def _silent(fraction: float,
            message: str) -> None:
    """The default progress callback, which reports nothing."""


def extend(path: Path,
           end: str | date | None = None,
           progress: Progress = _silent) -> Extension:
    """Extend a generated store to *end*, keeping everything already in it.

    Args:
        path: The store folder.
        end: The last date to extend to; today if omitted. A date with no
            new session (a weekend, or a date the store already reaches)
            leaves the store as it was.
        progress: Called at each stage with the fraction done and what is
            happening.

    Returns:
        Extension: What was added.

    Raises:
        ValueError: If the store holds no generator state, which is the case
            for anything not generated by py-beacon 0.1.2 or later.
    """
    settings, names = state.load(path)

    progress(0.02, "Reading the store")
    history, turnover = _read_market(path, pd.Timestamp(settings.end))
    last = pd.Timestamp(history["DATE"].max())

    target = pd.Timestamp(end if end is not None else date.today()).normalize()
    dates = sessions(last + pd.Timedelta(days=1), target, settings.calendar)

    if dates.empty:
        return Extension(None, None, 0, 0, 0)

    rng = np.random.default_rng([settings.seed, dates[0].toordinal(),
                                 dates[-1].toordinal()])

    progress(0.20, "Drawing listings and delistings")
    live = names.loc[names["listed_to"].isna()].copy()
    joined = _newcomers(live, len(names), dates, settings, rng)
    live["listed_to"] = listings.leaving(live["alpha"].to_numpy(), dates, rng,
                                         settings.delisting_rate)

    progress(0.30, "Simulating returns")
    frame = pd.concat([_carried(live, history, turnover), joined])
    horizon = sessions(dates[0], max(dates[-1], dates[0] + MIN_SIMULATED_SPAN),
                       settings.calendar)
    panel = returns.simulate(frame, horizon, rng,
                             risk_free_rate=settings.risk_free_rate,
                             equity_premium=settings.equity_premium
                             ).iloc[:len(dates)]

    progress(0.50, "Building prices and corporate actions")
    schedule = sessions(pd.Timestamp(year=last.year, month=1, day=1),
                        dates[-1], settings.calendar)
    market, actions = prices.build(frame, panel, rng, calendar=schedule)

    progress(0.65, "Carrying on exchange rates")
    rates = fx.build(dates, rng, start_from=_last_rates(history))

    feature_rows = None
    if settings.features:
        progress(0.70, "Carrying on features")
        feature_rows = features.carry_on(
            _read_features(path), pd.concat([names, joined]), joined.index,
            _closes(history, market, names.index.union(joined.index)),
            last, rng)

    progress(0.80, "Updating reference data")
    reference = _reference(joined, live, dates, rng)

    progress(0.85, "Saving the store")
    names.loc[live.index, "listed_to"] = live["listed_to"]
    settings.end = target.date().isoformat()
    settings.extensions.append({"from": dates[0].date().isoformat(),
                                "to": dates[-1].date().isoformat()})

    _write(path, market, rates, actions, feature_rows, reference,
           settings, pd.concat([names, joined]), dates[-1])
    progress(1.0, "Done")

    delisted = int(live["listed_to"].notna().sum())
    logger.info("Extended %s by %d session(s) to %s: %d listed, %d delisted.",
                path, len(dates), dates[-1].date(), len(joined), delisted)

    return Extension(dates[0].date().isoformat(), dates[-1].date().isoformat(),
                     len(dates), len(joined), delisted)


def _newcomers(live: pd.DataFrame,
               first_position: int,
               dates: pd.DatetimeIndex,
               settings: state.Settings,
               rng: np.random.Generator) -> pd.DataFrame:
    """The names that list during the extension.

    As many as the listing rate implies for a universe the size of the one
    listed now, over the extension's length. A newcomer does not also leave
    within the extension it joins in.
    """
    expected = len(live) * settings.listing_rate * len(dates) / TRADING_DAYS
    count = int(rng.poisson(expected)) if expected > 0 else 0

    if count == 0:
        return live.iloc[0:0].copy()

    joined = universe.newcomers(first_position, count, rng, settings.assets)
    joined["listed_from"] = listings.joining(count, dates, rng,
                                             settings.listing_rate).to_numpy()
    joined["listed_to"] = pd.NaT

    return joined


def _carried(live: pd.DataFrame,
             history: pd.DataFrame,
             turnover: pd.Series) -> pd.DataFrame:
    """The listed names, set to start from their last close and share count."""
    ordered = history.sort_values(["IDENTIFIER", "DATE"], kind="mergesort")
    last_bar = ordered.drop_duplicates("IDENTIFIER", keep="last").set_index(
        "IDENTIFIER")

    carried = live.loc[live.index.intersection(last_bar.index)].copy()
    carried["initial_price"] = last_bar.loc[carried.index, "CLOSE"]
    carried["shares_outstanding"] = last_bar.loc[carried.index,
                                                 "SHARES_OUTSTANDING"]
    carried["turnover"] = turnover.reindex(carried.index)

    return carried


def _last_rates(history: pd.DataFrame) -> dict[str, float]:
    """Each FX pair's last stored rate."""
    pairs = {identifier for identifier, _, _ in regions.pairs()}
    rows = history.loc[history["IDENTIFIER"].isin(pairs)].sort_values("DATE")

    return {str(identifier): float(rate) for identifier, rate
            in rows.groupby("IDENTIFIER")["CLOSE"].last().items()}


def _closes(history: pd.DataFrame,
            market: pd.DataFrame,
            equities: pd.Index) -> pd.DataFrame:
    """Wide closes over the recent history and the extension together."""
    recent = history.loc[history["IDENTIFIER"].isin(equities),
                         ["DATE", "IDENTIFIER", "CLOSE"]]
    combined = pd.concat([recent, market[["DATE", "IDENTIFIER", "CLOSE"]]],
                         ignore_index=True)

    return combined.pivot_table(index="DATE", columns="IDENTIFIER",
                                values="CLOSE", aggfunc="last")


def _reference(joined: pd.DataFrame,
               live: pd.DataFrame,
               dates: pd.DatetimeIndex,
               rng: np.random.Generator) -> _ReferenceChanges:
    """The reference rows to add, and the changes to existing ones."""
    rows = None

    if not joined.empty:
        profile = profiles.build(joined, rng, dates[-1])
        rows = ReferenceData.from_dataframe(universe.reference_frame(
            joined, dates[0].date().isoformat(), profile)).data

    leaving = live["listed_to"].dropna()
    staying = live.index.difference(leaving.index)
    earnings = profiles.next_earnings(dates[-1], len(staying), rng)

    return _ReferenceChanges(
        rows=rows,
        delisted={str(identifier): day.date().isoformat()
                  for identifier, day in leaving.items()},
        next_earnings={str(identifier): day.date().isoformat()
                       for identifier, day in zip(staying, earnings,
                                                  strict=True)})


# -- reading ---------------------------------------------------------------


def _read_market(path: Path,
                 end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    """The recent market rows, and each name's average daily turnover.

    Turnover is volume over shares outstanding, averaged over the whole
    history: the rate each name has traded at, which its new volume carries
    on from.
    """
    cutoff = (end - pd.Timedelta(days=HISTORY_DAYS)).date().isoformat()
    columns = ["IDENTIFIER", "DATE", "CLOSE", "VOLUME", "SHARES_OUTSTANDING"]

    recent = []
    total = pd.Series(dtype=float)
    count = pd.Series(dtype=float)

    with gzip.open(path / store.MARKET_FILE, "rt", encoding="utf-8") as handle:
        for chunk in pd.read_csv(handle, usecols=columns,
                                 chunksize=READ_CHUNK_ROWS):
            ratio = chunk["VOLUME"] / chunk["SHARES_OUTSTANDING"]
            valid = ratio.notna() & np.isfinite(ratio)
            grouped = ratio[valid].groupby(chunk.loc[valid, "IDENTIFIER"])
            total = total.add(grouped.sum(), fill_value=0.0)
            count = count.add(grouped.count(), fill_value=0.0)

            recent.append(chunk.loc[chunk["DATE"] >= cutoff])

    history = pd.concat(recent, ignore_index=True)
    history["DATE"] = pd.to_datetime(history["DATE"], format=DATE_FORMAT)

    return history, total / count


def _read_features(path: Path) -> pd.DataFrame:
    """The stored feature rows an extension carries on from."""
    source = path / store.FEATURES_FILE

    if not source.is_file():
        return pd.DataFrame(columns=["IDENTIFIER", "FIELD", "VALUE", "DETAIL"])

    wanted = {"pe_ratio", "pb_ratio", "debt_to_equity", "wikipedia_views"}

    with gzip.open(source, "rt", encoding="utf-8") as handle:
        kept = [chunk.loc[chunk["FIELD"].isin(wanted)]
                for chunk in pd.read_csv(handle,
                                         usecols=["IDENTIFIER", "FIELD",
                                                  "VALUE", "DETAIL"],
                                         chunksize=READ_CHUNK_ROWS)]

    return pd.concat(kept, ignore_index=True)


# -- writing ---------------------------------------------------------------


def _write(path: Path,
           market: pd.DataFrame,
           rates: pd.DataFrame,
           actions: pd.DataFrame,
           feature_rows: pd.DataFrame | None,
           reference: _ReferenceChanges,
           settings: state.Settings,
           names: pd.DataFrame,
           last: pd.Timestamp) -> None:
    """Write the extension into a copy of the store, then swap it in.

    Working on a copy means a failure part-way leaves the store as it was.
    Each new row goes through the same containers a generated store's rows
    do, so it is written exactly as a generated row would be.
    """
    work = path.with_name(f".{path.name}.extending")
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(path, work)

    try:
        _append(work / store.MARKET_FILE, store.flatten_index(
            MarketData.from_dataframe(
                pd.concat([market, rates], ignore_index=True)).data))

        if feature_rows is not None and not feature_rows.empty:
            feature_rows["VALUE"] = feature_rows["VALUE"].astype(float)
            _append(work / store.FEATURES_FILE, store.flatten_index(
                FeatureData.from_dataframe(feature_rows).data))

        _rewrite_actions(work / store.ACTIONS_FILE, actions, last)
        _rewrite_reference(work / store.REFERENCE_FILE, reference)
        _record_datasets(work)
        state.save(work, settings, names)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise

    previous = path.with_name(f".{path.name}.previous")
    shutil.rmtree(previous, ignore_errors=True)
    path.rename(previous)
    work.rename(path)
    shutil.rmtree(previous, ignore_errors=True)


def _header(target: Path) -> list[str] | None:
    """A stored file's columns, or None if there is no file."""
    if not target.is_file():
        return None

    with gzip.open(target, "rt", encoding="utf-8") as handle:
        return handle.readline().rstrip("\n").split(",")


def _append(target: Path,
            rows: pd.DataFrame) -> None:
    """Append rows to a stored file without touching the ones there.

    A gzip file may hold several members one after another, and reading it
    gives them back as one stream. So the new rows go in as a member of
    their own, and the bytes already in the file are left exactly as they
    were.
    """
    columns = _header(target)

    if columns is None:
        rows.to_csv(target, index=False, lineterminator="\n", compression=_GZIP)
        return

    text = rows.reindex(columns=columns).to_csv(index=False, header=False,
                                                lineterminator="\n")

    with gzip.GzipFile(target, "ab", mtime=0) as handle:
        handle.write(text.encode("utf-8"))


def _as_text(target: Path) -> pd.DataFrame:
    """A small stored file as text, so rewriting it changes nothing it holds."""
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle, dtype=str, keep_default_na=False)


def _rows_as_text(rows: pd.DataFrame,
                  columns: list[str]) -> pd.DataFrame:
    """New rows written and read back as text, formatted as stored rows are."""
    text = rows.reindex(columns=columns).to_csv(index=False,
                                                lineterminator="\n")

    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def _save_text(frame: pd.DataFrame,
               target: Path) -> None:
    frame.to_csv(target, index=False, lineterminator="\n", compression=_GZIP)


def _rewrite_actions(target: Path,
                     actions: pd.DataFrame,
                     last: pd.Timestamp) -> None:
    """Add the new actions, and mark dividends paid whose pay date has come."""
    new = store.flatten_index(CorporateActions.from_dataframe(actions).data)

    if not target.is_file():
        if not new.empty:
            _save_text(new, target)
        return

    stored = _as_text(target)
    arrived = ((stored["STATUS"] == ANNOUNCED)
               & (stored["PAY_DATE"] <= last.date().isoformat()))
    stored.loc[arrived, "STATUS"] = PAID

    combined = pd.concat([stored, _rows_as_text(new, list(stored.columns))],
                         ignore_index=True)
    _save_text(combined.sort_values(["IDENTIFIER", "EX_DATE"],
                                    kind="mergesort"), target)


def _rewrite_reference(target: Path,
                       changes: _ReferenceChanges) -> None:
    """Record delistings, move earnings dates on, and add new names."""
    stored = _as_text(target)
    identifiers = stored["IDENTIFIER"]

    leaving = identifiers.isin(changes.delisted.keys())
    stored.loc[leaving, "DATE_TO"] = identifiers[leaving].map(changes.delisted)

    if "TRADING_STATUS" in stored:
        stored.loc[leaving, "TRADING_STATUS"] = profiles.DELISTED

    if "NEXT_EARNINGS" in stored:
        moving = identifiers.isin(changes.next_earnings.keys())
        stored.loc[moving, "NEXT_EARNINGS"] = identifiers[moving].map(
            changes.next_earnings)

    if changes.rows is not None:
        stored = pd.concat([stored, _rows_as_text(
            store.flatten_index(changes.rows), list(stored.columns))],
            ignore_index=True)

    _save_text(stored.sort_values("IDENTIFIER", kind="mergesort"), target)


def _record_datasets(path: Path) -> None:
    """List any dataset the extension created in the manifest."""
    manifest_path = path / store.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    datasets = [name for name, file in store.FILE_FOR_DATASET.items()
                if (path / file).is_file()]

    if datasets != manifest.get("datasets"):
        manifest["datasets"] = datasets
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)
                                 + "\n", encoding="utf-8")

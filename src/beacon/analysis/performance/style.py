# src/beacon/analysis/performance/style.py
"""
A style box: where each name a run holds sits by size (large, mid, small) and
by style (value, blend, growth), and where each portfolio's weight sits.

    positions = style_positions(snapshot, weights, settings)
    style_box(positions)              # the weight in each of the nine cells
    style_classification(positions)   # each portfolio's centre: "Large blend"

## Size

By market cap in US dollars, in fixed bands: large above $10bn, mid from $2bn
to $10bn, small below $2bn. A name's size position runs from 0 (the top of
large) to 3 (the bottom of small) on a log scale within each band: $1tn is 0,
$10bn 1, $2bn 2 and $100m 3. **The scale is not clipped**: a name above $1tn
sits below 0 and one below $100m above 3, still large and small.

## Style

Each name has a value score, the mean cross-sectional z-score of its earnings
yield (1 / P/E), book yield (1 / P/B) and dividend yield, and a growth score,
the mean z-score of its earnings and sales growth (over a year, or the yearly
rate over three). The z-scores are over the index's names, each clipped at
three standard deviations. Its style score is growth less value. Its style
position is the score's cap-weighted percentile among the index's names in
the same size band, times 3: 0 is the deepest value and 3 the most growth,
and each style holds a third of the band's cap. A band the index has no names
in is ranked against all of the index's names.

## What cannot be placed

A score averages the fields a name has; a name with no value or growth field
at all is placed at blend (1.5). A name with no market cap cannot be sized and
is left out: each portfolio's weight is rescaled over the names placed, and
the share placed is its coverage.

A cell is the floor of a position, so a name exactly on an edge, such as
$10bn, is in the smaller or more growth-like cell. A portfolio's centre is the
weighted mean of its names' positions, and its classification the cell the
centre is in.
"""
# BN-427, agreed with Karan on 2026-11-20 for beacon-ui's Summary (BU-294).
# Morningstar's style box, with fixed size bands and real growth data.
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

SIZES = ("large", "mid", "small")
STYLES = ("value", "blend", "growth")

# The size scale's anchors, in US dollars, and the positions they take.
TOP, LARGE_FROM, MID_FROM, BOTTOM = 1e12, 10e9, 2e9, 1e8

# The growth measures: a year's, or the yearly rate over three.
YEAR, THREE_YEARS = "year", "three_years"
GROWTH_FIELDS = {YEAR: ("EPS_GROWTH", "SALES_GROWTH"),
                 THREE_YEARS: ("EPS_GROWTH_3Y", "SALES_GROWTH_3Y")}

# A z-score past this many standard deviations is held there, so one extreme
# name does not set everyone's score.
CLIP = 3.0

# Where a name with no style fields sits: the middle of blend.
BLEND = 1.5


@dataclass(frozen=True)
class StyleSettings:
    """How names are placed in the style box.

    Attributes:
        large_from: The market cap, in US dollars, large starts above.
        mid_from: The market cap mid starts at.
        top: The cap at position 0, the top of large.
        bottom: The cap at position 3, the bottom of small.
        growth: The growth measure: ``year`` or ``three_years``.
    """
    large_from: float = LARGE_FROM
    mid_from: float = MID_FROM
    top: float = TOP
    bottom: float = BOTTOM
    growth: str = YEAR


def size_positions(caps: pd.Series,
                   settings: StyleSettings) -> pd.Series:
    """Each name's size position from its market cap in US dollars: 0 at
    *top*, 1 at *large_from*, 2 at *mid_from*, 3 at *bottom*, on a log scale
    within each band and carried on past both ends; missing for a cap that
    is missing or not above zero."""
    edges = np.log([settings.top, settings.large_from, settings.mid_from, settings.bottom])
    logs = np.log(caps.where(caps > 0).astype(float))
    # Each band's position, the end bands carried on past their ends.
    large = (edges[0] - logs) / (edges[0] - edges[1])
    mid = 1.0 + (edges[1] - logs) / (edges[1] - edges[2])
    small = 2.0 + (edges[2] - logs) / (edges[2] - edges[3])
    found = np.where(logs >= edges[1], large, np.where(logs >= edges[2], mid, small))

    return pd.Series(found, index=caps.index, dtype=float).where(logs.notna())


def style_scores(snapshot: pd.DataFrame,
                 universe: pd.Index,
                 growth: str = YEAR) -> pd.Series:
    """Each name's style score, growth less value, from its characteristics
    (a row a name): z-scores over the names of *universe* that have each
    field, clipped at `CLIP`; missing for a name with no field at all."""
    value = {"earnings yield": _yield(snapshot.get("PE_RATIO")),
             "book yield": _yield(snapshot.get("PB_RATIO")),
             "dividend yield": snapshot.get("DIVIDEND_YIELD")}
    growths = {field: snapshot.get(field) for field in GROWTH_FIELDS[growth]}
    value_score = _mean_score(value, universe, snapshot.index)
    growth_score = _mean_score(growths, universe, snapshot.index)
    either = value_score.notna() | growth_score.notna()

    return (growth_score.fillna(0.0) - value_score.fillna(0.0)).where(either)


def style_positions(snapshot: pd.DataFrame,
                    weights: Mapping[str, pd.Series | None],
                    settings: StyleSettings,
                    usd_rate: float = 1.0) -> pd.DataFrame:
    """Each name held, placed in the box.

    Args:
        snapshot: Each name's characteristics on the day (a row a name, by
            identifier): ``MARKET_CAP`` in the run's currency, the ratios,
            the dividend yield and the growth figures.
        weights: Each portfolio's weights on the day, by name; None for one
            the run kept no weights for. The index's names are the universe
            scores and percentiles are measured over.
        settings: The size bands and the growth measure.
        usd_rate: A US dollar in the run's currency, to put caps in dollars.

    Returns:
        pd.DataFrame: A row a name any portfolio holds and that can be
        sized: ``name``, ``size``, ``style``, then each portfolio's weight (0
        where it does not hold the name, blank for a portfolio without
        weights).
    """
    held = {book: found[found > 0] for book, found in weights.items() if found is not None}
    names = sorted({str(name) for found in held.values() for name in found.index})
    universe = pd.Index([str(name) for name in held.get("index", pd.Series(dtype=float)).index])

    if universe.empty:
        universe = pd.Index(names)

    snapshot = snapshot.reindex(sorted(set(names) | set(universe)))
    caps = snapshot["MARKET_CAP"].astype(float) / usd_rate if "MARKET_CAP" in snapshot else (
        pd.Series(np.nan, index=snapshot.index))
    size = size_positions(caps, settings)
    scores = style_scores(snapshot, universe, settings.growth)
    style = _style_positions(scores, size, caps, universe)
    placed = [name for name in names if pd.notna(size.get(name))]
    table = pd.DataFrame({"name": placed, "size": size.reindex(placed).to_numpy(),
                          "style": style.reindex(placed).fillna(BLEND).to_numpy()})

    for book, found in weights.items():
        table[book] = (None if found is None
                       else found.reindex(placed).fillna(0.0).clip(lower=0.0).to_numpy())

    return table


def coverage(positions: pd.DataFrame,
             weights: Mapping[str, pd.Series | None]) -> dict[str, float | None]:
    """The share of each portfolio's held weight that was placed."""
    found: dict[str, float | None] = {}

    for book, held in weights.items():
        total = float(held[held > 0].sum()) if held is not None else 0.0
        found[book] = (float(positions[book].sum()) / total
                       if held is not None and total > 0 else None)

    return found


def style_box(positions: pd.DataFrame,
              books: tuple[str, ...]) -> pd.DataFrame:
    """Each portfolio's weight in each of the nine cells, over its placed
    names: a row a cell, ``size`` and ``style``, then a column a portfolio,
    each summing to 1 (blank for a portfolio that holds none)."""
    cells = pd.DataFrame([(size, style) for size in SIZES for style in STYLES],
                         columns=["size", "style"])
    size_cells = np.array(SIZES, dtype=object)[_cell(positions["size"])]
    style_cells = np.array(STYLES, dtype=object)[_cell(positions["style"])]

    for book in books:
        shares = _shares(positions, book)

        if shares is None:
            cells[book] = None
            continue

        found = pd.Series(shares).groupby([size_cells, style_cells]).sum()
        cells[book] = [float(found.get((size, style), 0.0))
                       for size, style in zip(cells["size"], cells["style"], strict=True)]

    return cells


def style_classification(positions: pd.DataFrame,
                         books: tuple[str, ...]) -> pd.DataFrame:
    """Each portfolio's centre, the weighted mean of its placed names'
    positions, and the cell it is in: a row a portfolio that holds names,
    ``book``, ``size``, ``style``, ``size_centre`` and ``style_centre``."""
    rows = []

    for book in books:
        shares = _shares(positions, book)

        if shares is None:
            continue

        size = float(np.dot(shares, positions["size"].to_numpy(dtype=float)))
        style = float(np.dot(shares, positions["style"].to_numpy(dtype=float)))
        rows.append({"book": book, "size": SIZES[_cell(pd.Series([size]))[0]],
                     "style": STYLES[_cell(pd.Series([style]))[0]],
                     "size_centre": size, "style_centre": style})

    return pd.DataFrame(rows, columns=["book", "size", "style", "size_centre", "style_centre"])


def _yield(ratios: pd.Series | None) -> pd.Series | None:
    """A ratio's yield, 1 over it; missing for a zero ratio."""
    return None if ratios is None else 1.0 / ratios.astype(float).replace(0.0, np.nan)


def _mean_score(fields: Mapping[str, pd.Series | None],
                universe: pd.Index,
                names: pd.Index) -> pd.Series:
    """The mean of each name's clipped z-scores over the fields it has."""
    scores = []

    for given in fields.values():
        if given is None:
            continue

        values = given.astype(float).replace([np.inf, -np.inf], np.nan)
        among = values.reindex(universe).dropna()
        spread = float(among.std()) if len(among) > 1 else 0.0

        if not spread > 0:
            continue

        scores.append(((values - float(among.mean())) / spread).clip(-CLIP, CLIP))

    if not scores:
        return pd.Series(np.nan, index=names)

    return pd.concat(scores, axis=1).mean(axis=1).reindex(names)


def _style_positions(scores: pd.Series,
                     size: pd.Series,
                     caps: pd.Series,
                     universe: pd.Index) -> pd.Series:
    """Each name's style score as a cap-weighted percentile among the
    universe's names in its size band, times 3: the cap below its score and
    half the cap level with it, over the band's."""
    bands = pd.Series(_cell(size.fillna(BLEND)), index=size.index)
    ranked = pd.DataFrame({"score": scores, "cap": caps, "band": bands}).reindex(universe)
    ranked = ranked.dropna(subset=["score", "cap"])
    ranked = ranked[ranked["cap"] > 0]
    found = pd.Series(np.nan, index=scores.index)
    scored = scores.dropna()

    for band in range(3):
        names = scored.index[bands.reindex(scored.index).to_numpy() == band]
        among = ranked[ranked["band"] == band]
        among = among if not among.empty else ranked

        if names.empty or among.empty:
            continue

        among = among.sort_values("score")
        levels = among["score"].to_numpy(dtype=float)
        running = np.concatenate([[0.0], np.cumsum(among["cap"].to_numpy(dtype=float))])
        mine = scored[names].to_numpy(dtype=float)
        below = running[np.searchsorted(levels, mine, side="left")]
        level = running[np.searchsorted(levels, mine, side="right")] - below
        found[names] = 3.0 * (below + level / 2.0) / running[-1]

    return found


def _cell(positions: pd.Series) -> np.ndarray:
    """Each position's cell, 0 to 2: its floor, with anything past either end
    in the end cell."""
    found: np.ndarray = np.clip(np.floor(positions.to_numpy(dtype=float)), 0, 2).astype(int)

    return found


def _shares(positions: pd.DataFrame,
            book: str) -> np.ndarray | None:
    """*book*'s weights over its placed names, summing to 1; None when it
    has no weights or holds none of them."""
    if book not in positions or positions[book].isna().all():
        return None

    weights = positions[book].to_numpy(dtype=float)
    total = weights.sum()

    return weights / total if total > 0 else None

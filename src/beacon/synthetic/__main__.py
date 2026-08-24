# src/beacon/synthetic/__main__.py
"""
Generate a synthetic data store from the command line.

    python -m beacon.synthetic --assets 512 --start 2019-12-31 --seed 42

With no ``--out`` the store is written where `python -m beacon.server`
auto-loads it, so generating and then spawning the server is all it takes to
have a populated client.

## Why the date defaults differ from the library's

`SyntheticConfig` defaults to a fixed window and a small universe, so
`generate()` returns the same data on any day, in a couple of seconds, and a
test can depend on it. This CLI defaults to **6,000 names over the ten years
ending today**, because its output is the dataset an application is
demonstrated on: 512 names is visibly a toy in a universe pane built for
thousands, and a dataset that stopped eighteen months ago shows up as stale in
every freshness indicator in the client — a true statement about the data and
a misleading one about the application.

Pass both ``--start`` and ``--end`` when reproducibility matters — the seed
fixes the draw, not the calendar.

## The two expansion flags

``--extended-universe`` doubles the universe to 10,000 names.
``--long-history`` reaches back past every crisis the generator models, rather
than the ten years the default covers.

Each roughly doubles the work; together they are about five times the default
and around 10 GB of memory, so the CLI estimates and warns before it starts.
Both defer to an explicit ``--assets`` or ``--start``: they widen a default,
they do not overrule a value somebody named.
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from ..data import store
from .dataset import (
    DEFAULT_EQUITY_PREMIUM,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SEED,
    SyntheticConfig,
    write,
)
from .regimes import CRISES

logger = logging.getLogger(__name__)

# The CLI's defaults are the *client's* dataset, and are deliberately larger
# than `SyntheticConfig`'s. The library default has to stay small enough that
# `generate()` is a couple of seconds inside a test; this produces the store a
# populated application is demonstrated on, and 512 names over five years is
# visibly a toy in a universe pane built for thousands.
#
# 6,000 rather than 5,000 because roughly 27% of a panel is outside its listed
# life at any instant, at the 3% hazards `listings.py` uses. The union is the
# number the universe pane shows; the live subset is what an index can
# actually select, and at 5,000 that subset was 3,650-3,880 -- the pane
# promising a third more than it could deliver. 6,000 puts 4,375-4,656 names
# investable throughout without touching the turnover, which stays at the
# realistic figure rather than being tuned to flatter the count.
DEFAULT_ASSETS = 6_000
DEFAULT_YEARS = 10

# What the two expansion flags open up. Both are a different order of
# magnitude of work rather than a slightly bigger run, so neither is the
# default and the CLI says what it is about to do before it starts.
EXTENDED_ASSETS = 10_000

# `--long-history` is anchored to the regimes rather than set to a round
# number of years, and that is a correction rather than a flourish. The
# crises sit at *fixed dates* while a rolling window moves, so a "25 years
# back from today" flag silently loses them: run at the end of 2025 it starts
# in December 2000 and already clips the first ten months of the dot-com
# unwind, and by 2030 it would miss that episode altogether while cutting into
# the run-up to the financial crisis. The flag exists to deliver the crises,
# so the crises decide where it starts.
#
# The run-in gives a risk model some calm history to estimate on before the
# first episode arrives, rather than starting the panel mid-collapse.
LONG_HISTORY_RUN_IN_YEARS = 1

# Peak memory, per million rows of market data. Measured after the blocking
# work in BN-128, and remarkably linear across the range that matters:
#
#     6,000 x 10y    11.8M rows   2.51 GB    19s   (the default)
#    10,000 x 10y    26.1M rows   3.82 GB    29s
#     5,000 x 25y    32.6M rows   4.77 GB    42s
#
# The default carries fewer rows than the 5,000-name panel it replaced (13.0M)
# despite holding a thousand more names: since BN-130 a delisted instrument
# has its rows removed rather than nulled, so a panel with turnover is smaller
# than the grid its universe size implies. The two expansion figures predate
# that and are upper bounds.
#
# Both flags together is roughly 69M rows and 10 GB, which is why the estimate
# below exists: that combination will exhaust a 16 GB machine, and finding out
# by watching it swap for ten minutes is a bad way to learn it.
GIGABYTES_PER_MILLION_ROWS = 0.146
MEMORY_WARNING_GIGABYTES = 4.0


def default_window(today: date | None = None,
                   years: int = DEFAULT_YEARS) -> tuple[str, str]:
    """The business days ending today, going back `years`.

    Args:
        today: Overridable so a test does not depend on the calendar.
        years: How far back to reach. `--long-history` widens it.

    Returns:
        tuple: ISO start and end dates.
    """
    end = pd.Timestamp(today if today is not None else date.today()).normalize()
    start = end - pd.DateOffset(years=years)

    return start.date().isoformat(), end.date().isoformat()


def long_history_start() -> str:
    """The first date `--long-history` reaches back to.

    Derived from the regimes, so adding an earlier episode widens the flag
    automatically instead of leaving it quietly short of the thing it
    advertises.
    """
    earliest = min(pd.Timestamp(regime.start) for regime in CRISES)

    return str((earliest - pd.DateOffset(years=LONG_HISTORY_RUN_IN_YEARS)
                ).date().isoformat())


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m beacon.synthetic",
        description="Generate a synthetic market-data store.")

    parser.add_argument("--assets", type=int, default=None,
                        help=f"universe size (default: {DEFAULT_ASSETS:,}, "
                             f"or {EXTENDED_ASSETS:,} with --extended-universe)")
    parser.add_argument("--extended-universe", action="store_true",
                        help=f"widen the universe to {EXTENDED_ASSETS:,} names; "
                             f"ignored if --assets is given")
    parser.add_argument("--long-history", action="store_true",
                        help=f"reach back to {long_history_start()} instead of "
                             f"{DEFAULT_YEARS} years, so the panel covers every "
                             f"crisis the generator models; ignored if --start "
                             f"is given")
    parser.add_argument("--start", default=None,
                        help=f"first date (default: {DEFAULT_YEARS} years before --end)")
    parser.add_argument("--end", default=None,
                        help="last date (default: today)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--risk-free-rate", type=float,
                        default=DEFAULT_RISK_FREE_RATE,
                        help=f"annualised (default: {DEFAULT_RISK_FREE_RATE})")
    parser.add_argument("--equity-premium", type=float,
                        default=DEFAULT_EQUITY_PREMIUM,
                        help=f"annualised (default: {DEFAULT_EQUITY_PREMIUM})")
    parser.add_argument("--no-features", action="store_true",
                        help="skip the fundamental ratios and alternative "
                             "data (about 8%% of the rows)")
    parser.add_argument("--out", type=Path, default=None,
                        help="store directory (default: the location "
                             "`python -m beacon.server` auto-loads)")

    return parser


def resolve_window(start: str | None,
                   end: str | None,
                   long_history: bool = False) -> tuple[str, str]:
    """Fill in whichever of the two dates was not given.

    An explicit `--start` always wins over `--long-history`: the flag is a
    convenient span, not an instruction to overrule a date the caller named.
    """
    fallback_start, fallback_end = default_window()
    resolved_end = end if end is not None else fallback_end

    if start is not None:
        return start, resolved_end

    if long_history:
        return long_history_start(), resolved_end

    if end is not None:
        offset = pd.Timestamp(end) - pd.DateOffset(years=DEFAULT_YEARS)

        return offset.date().isoformat(), resolved_end

    return fallback_start, resolved_end


def resolve_assets(assets: int | None,
                   extended_universe: bool) -> int:
    """How many names to generate.

    An explicit `--assets` wins over `--extended-universe`, for the same
    reason `--start` wins over `--long-history`.
    """
    if assets is not None:
        return assets

    return EXTENDED_ASSETS if extended_universe else DEFAULT_ASSETS


def main(argv: list[str] | None = None) -> int:
    """Generate a store and report where it landed.

    Args:
        argv: Argument list, defaulting to sys.argv[1:].

    Returns:
        int: Process exit code. 2 if the arguments do not describe a dataset.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    start, end = resolve_window(args.start, args.end, args.long_history)
    assets = resolve_assets(args.assets, args.extended_universe)

    _announce(assets, start, end)

    try:
        config = SyntheticConfig(assets=assets,
                                 start=start,
                                 end=end,
                                 seed=args.seed,
                                 risk_free_rate=args.risk_free_rate,
                                 equity_premium=args.equity_premium,
                                 features=not args.no_features)
        path = args.out if args.out is not None else store.default_path()

        written = write(config, path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {assets:,} identifiers from {start} to {end} "
          f"(seed {args.seed}) to {written}.")

    return 0


def _announce(assets: int,
              start: str,
              end: str) -> None:
    """Say what is about to happen, before it takes minutes to happen.

    A generator that prints nothing until it finishes is indistinguishable
    from one that has hung, and at the sizes the expansion flags open up the
    wait is long enough for somebody to reasonably conclude the second.
    """
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0.0)
    rows = assets * len(pd.bdate_range(start, end))
    projected = rows / 1e6 * GIGABYTES_PER_MILLION_ROWS

    logger.info("Generating %s names over %.0f years (%s to %s): "
                "%s rows, around %.1f GB peak.",
                f"{assets:,}", years, start, end, f"{rows:,}", projected)

    if projected >= MEMORY_WARNING_GIGABYTES:
        logger.warning(
            "That needs roughly %.0f GB of memory. Narrow the universe with "
            "--assets or the window with --start if this machine has less.",
            projected)


if __name__ == "__main__":
    raise SystemExit(main())

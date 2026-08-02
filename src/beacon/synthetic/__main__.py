# src/beacon/synthetic/__main__.py
"""
Generate a synthetic data store from the command line.

    python -m beacon.synthetic --assets 512 --start 2019-12-31 --seed 42

With no ``--out`` the store is written where `python -m beacon.server`
auto-loads it, so generating and then spawning the server is all it takes to
have a populated client.

## Why the date defaults differ from the library's

`SyntheticConfig` defaults to a fixed window, so `generate()` returns the same
data on any day and a test can depend on it. This CLI defaults to the five
years ending *today*, because its output is demo data: a dataset that stopped
eighteen months ago shows up as stale in every freshness indicator in the
client, which is a true statement about the data and a misleading one about
the application.

Pass both ``--start`` and ``--end`` when reproducibility matters — the seed
fixes the draw, not the calendar.
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from ..data import store
from .dataset import (
    DEFAULT_ASSETS,
    DEFAULT_EQUITY_PREMIUM,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SEED,
    SyntheticConfig,
    write,
)

logger = logging.getLogger(__name__)

# The CLI's default span, in years, ending today.
DEFAULT_YEARS = 5


def default_window(today: date | None = None) -> tuple[str, str]:
    """The five years of business days ending today.

    Args:
        today: Overridable so a test does not depend on the calendar.

    Returns:
        tuple: ISO start and end dates.
    """
    end = pd.Timestamp(today if today is not None else date.today()).normalize()
    start = end - pd.DateOffset(years=DEFAULT_YEARS)

    return start.date().isoformat(), end.date().isoformat()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m beacon.synthetic",
        description="Generate a synthetic market-data store.")

    parser.add_argument("--assets", type=int, default=DEFAULT_ASSETS,
                        help=f"universe size (default: {DEFAULT_ASSETS})")
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
    parser.add_argument("--out", type=Path, default=None,
                        help="store directory (default: the location "
                             "`python -m beacon.server` auto-loads)")

    return parser


def resolve_window(start: str | None,
                   end: str | None) -> tuple[str, str]:
    """Fill in whichever of the two dates was not given."""
    fallback_start, fallback_end = default_window()

    resolved_end = end if end is not None else fallback_end

    if start is not None:
        return start, resolved_end

    if end is not None:
        offset = pd.Timestamp(end) - pd.DateOffset(years=DEFAULT_YEARS)

        return offset.date().isoformat(), resolved_end

    return fallback_start, resolved_end


def main(argv: list[str] | None = None) -> int:
    """Generate a store and report where it landed.

    Args:
        argv: Argument list, defaulting to sys.argv[1:].

    Returns:
        int: Process exit code. 2 if the arguments do not describe a dataset.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    start, end = resolve_window(args.start, args.end)

    try:
        config = SyntheticConfig(assets=args.assets,
                                 start=start,
                                 end=end,
                                 seed=args.seed,
                                 risk_free_rate=args.risk_free_rate,
                                 equity_premium=args.equity_premium)
        path = args.out if args.out is not None else store.default_path()

        written = write(config, path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {args.assets} identifiers from {start} to {end} "
          f"(seed {args.seed}) to {written}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# examples/_shared.py
"""
Setup every example shares: data, an index definition, and printing helpers.

Each script generates its own data rather than expecting a store to exist, so
they run from a clean checkout with no network and nothing prepared. The
defaults are deliberately small — a few dozen names over a few years — so a
script finishes while you are still reading it. `--full` scales that up.
"""
import argparse
import logging

import pandas as pd

from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted, MarketCapWeighted
from beacon.synthetic import SyntheticConfig, generate

# Small enough to run in seconds, large enough that a cap binds and a
# covariance is worth estimating.
#
# The seed is chosen, not arbitrary. The generator has a +6% equity premium, so
# a demo whose index *loses* money over four years reads as a broken library
# rather than as an unlucky path. Across twelve seeds the annualised outcome
# ranged from -7.4% to +11.0%; seed 2 lands at +8.5%, nearest the model's own
# expectation. Nothing else is tuned — change the seed and you get a different,
# equally valid market.
QUICK = SyntheticConfig(assets=40, start="2021-01-04", end="2024-12-31", seed=2)
FULL = SyntheticConfig(assets=250, start="2015-01-02", end="2024-12-31", seed=2)


def parse_arguments(description: str) -> argparse.Namespace:
    """Standard arguments for an example."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--full", action="store_true",
                        help="a larger universe over a longer history")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress library logging")

    arguments = parser.parse_args()

    # Warnings only by default: the library logs a line per rebalance at INFO,
    # which buries the output the example exists to show.
    logging.basicConfig(
        level=logging.ERROR if arguments.quiet else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    return arguments


def market_data(full: bool = False):
    """Generate a dataset and return it with a fetcher over it."""
    config = FULL if full else QUICK
    dataset = generate(config)

    heading(f"Data: {config.assets} names, {config.start} to {config.end}")
    print(f"  identifiers      {len(dataset.universe):,}")
    print(f"  business days    {len(dataset.returns):,}")
    print(f"  corporate actions{len(dataset.actions.data):>6,}")

    return dataset, dataset.fetcher(), config


def index_definition(dataset,
                     config: SyntheticConfig,
                     index_id: str = "DEMO",
                     cap: float | None = 0.10,
                     equal_weighted: bool = False) -> IndexDefinition:
    """A market-cap index over the generated universe, capped by default."""
    return IndexDefinition(
        index_id=index_id,
        index_name=f"{index_id} Index",
        base_date=config.start,
        base_value=1000.0,
        currency=config.currency,
        eligibility_rules=[],
        weighting_scheme=(EqualWeighted() if equal_weighted
                          else MarketCapWeighted(use_free_float=True)),
        rebalancing_frequency="QUARTERLY",
        universe_identifiers=list(dataset.universe.index),
        max_constituent_weight=cap)


def heading(text: str) -> None:
    """A section rule, so a run reads as sections rather than a wall."""
    print(f"\n{text}\n{'-' * len(text)}")


def show_metrics(summary: dict[str, float | None]) -> None:
    """Print a backtest summary, percentages as percentages."""
    as_percent = {"total_return", "annualised_return", "volatility",
                  "max_drawdown", "tracking_error", "tracking_difference"}

    for name, value in summary.items():
        if value is None:
            print(f"  {name:<22} n/a")
        elif name in as_percent:
            print(f"  {name:<22} {value:>8.2%}")
        else:
            print(f"  {name:<22} {value:>8.3f}")


def show_weights(weights: dict[str, float],
                 limit: int = 8,
                 title: str = "Weights") -> None:
    """Print the largest weights, and say how many were not shown."""
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)

    print(f"  {title}:")
    for identifier, weight in ordered[:limit]:
        bar = "#" * max(int(weight * 200), 0)
        print(f"    {identifier:<8} {weight:>7.2%}  {bar}")

    if len(ordered) > limit:
        remainder = sum(weight for _, weight in ordered[limit:])
        print(f"    {'(' + str(len(ordered) - limit) + ' more)':<8} "
              f"{remainder:>7.2%}")


def latest_weights(result) -> dict[str, float]:
    """The weights in force at the end of an index run."""
    return result.weight_snapshots[max(result.weight_snapshots)]


def daily_returns(dataset) -> pd.DataFrame:
    """The return panel a risk model is estimated from."""
    return dataset.returns

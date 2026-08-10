# examples/01_index_and_backtest.py
"""
Define an index, calculate it, and backtest a portfolio that tracks it.

The smallest complete pass through all three layers:

    methodology  ->  calculator  ->  backtest

Run it:

    python examples/01_index_and_backtest.py
    python examples/01_index_and_backtest.py --full

## The one thing to notice

The index level and the backtest NAV are **not the same number**, even with
zero transaction costs. The calculator computes a market-cap-style level
(price x shares / divisor); the engine holds a weight-rebalanced portfolio that
drifts between rebalances and trades in whole amounts of cash. They track
closely and diverge slightly, and that divergence is the thing a tracking
portfolio actually experiences.
"""
from _shared import (
    heading,
    index_definition,
    latest_weights,
    market_data,
    parse_arguments,
    show_metrics,
    show_weights,
)

from beacon.backtest.engine import BacktestEngine
from beacon.index.calculation import IndexCalculator

COSTS_BPS = 10.0
CAPITAL = 10_000_000.0


def main() -> int:
    arguments = parse_arguments(__doc__ or "")
    dataset, fetcher, config = market_data(arguments.full)

    # --- 1. Define -------------------------------------------------------
    definition = index_definition(dataset, config, index_id="DEMO", cap=0.10)

    heading("Methodology")
    print(f"  weighting        {definition.weighting_scheme.scheme_name}")
    print(f"  rebalances       {definition.rebalancing_frequency}")
    print(f"  cap              {definition.max_constituent_weight:.0%}")
    print(f"  universe         {len(definition.universe_identifiers)} names")

    # --- 2. Calculate ----------------------------------------------------
    index = IndexCalculator(definition, fetcher).run(
        start_date=config.start, end_date=config.end)

    levels = index.index_levels

    heading("Index")
    print(f"  base             {levels.iloc[0]:,.2f} on {levels.index[0].date()}")
    print(f"  final            {levels.iloc[-1]:,.2f} on {levels.index[-1].date()}")
    print(f"  total return     {levels.iloc[-1] / levels.iloc[0] - 1:.2%}")
    print(f"  rebalances       {len(index.weight_snapshots)}")
    print()
    show_weights(latest_weights(index), title="Latest composition")

    if index.cap_reports:
        capped = max(index.cap_reports)
        report = index.cap_reports[capped]
        print(f"\n  the cap bound at {len(index.cap_reports)} rebalance(s); "
              f"at {capped.date()} it moved {report.redistributed:.2%} "
              f"off {len(report.capped)} name(s)")

    # --- 3. Backtest -----------------------------------------------------
    backtest = BacktestEngine(start_date=config.start,
                              end_date=config.end,
                              initial_capital=CAPITAL,
                              data_provider=fetcher,
                              target_index_result=index,
                              transaction_cost_bps=COSTS_BPS).run()

    nav = backtest.portfolio_nav

    heading(f"Backtest ({COSTS_BPS:.0f}bps costs)")
    print(f"  initial          {backtest.initial_capital:>14,.2f}")
    print(f"  final NAV        {nav.iloc[-1]:>14,.2f}")
    print(f"  trades           {len(backtest.transactions):>14,}")
    print(f"  total costs      "
          f"{sum(t.transaction_cost for t in backtest.transactions):>14,.2f}")
    print()
    show_metrics(backtest.summary())

    # --- 4. The gap ------------------------------------------------------
    index_return = levels.iloc[-1] / levels.iloc[0] - 1
    portfolio_return = nav.iloc[-1] / backtest.initial_capital - 1

    heading("Index versus portfolio")
    print(f"  index            {index_return:>8.2%}")
    print(f"  portfolio        {portfolio_return:>8.2%}")
    print(f"  difference       {portfolio_return - index_return:>8.2%}")
    print()
    print("  These differ by construction, not by mistake. The index is a")
    print("  divisor-based level; the portfolio holds units, drifts between")
    print("  rebalances and pays to trade. Costs explain part of the gap and")
    print("  the different construction explains the rest.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

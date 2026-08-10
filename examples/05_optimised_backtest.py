# examples/05_optimised_backtest.py
"""
Backtest an optimised portfolio against the index it was built to track.

    python examples/05_optimised_backtest.py

Needs the optimiser extra:

    pip install "py-beacon[optimise]"

## The question this answers

An optimiser reports a tracking error *ex ante* — what its risk model expects.
A backtest reports what actually happened. Those are different numbers, and the
gap between them is the honest measure of whether the risk model was any good.

A constrained portfolio also trades differently: position and sector limits
force trades a cap-weighted index never makes, so turnover and costs rise. The
comparison here is deliberately like-for-like on costs so the difference is
attributable to the constraints rather than to the fee schedule.
"""
from _shared import (
    heading,
    index_definition,
    latest_weights,
    market_data,
    parse_arguments,
    show_metrics,
)

from beacon.backtest.engine import BacktestEngine
from beacon.index.calculation import IndexCalculator
from beacon.optimise import (
    FullInvestment,
    GroupBounds,
    PositionBounds,
    minimise_tracking_error,
)
from beacon.risk import active_risk_contributions, estimate_risk_model

CAPITAL = 10_000_000.0
COSTS_BPS = 10.0
MAX_POSITION = 0.06
MAX_SECTOR = 0.25


def main() -> int:
    arguments = parse_arguments(__doc__ or "")
    dataset, fetcher, config = market_data(arguments.full)

    # --- The index we are tracking ---------------------------------------
    definition = index_definition(dataset, config, index_id="BENCH", cap=0.10)
    index = IndexCalculator(definition, fetcher).run(
        start_date=config.start, end_date=config.end)

    risk = estimate_risk_model(dataset.returns)

    sectors = dataset.universe.groupby("SECTOR").groups
    largest_sector = max(sectors, key=lambda name: len(sectors[name]))

    constraints = [
        FullInvestment(),
        PositionBounds(0.0, MAX_POSITION),
        GroupBounds(largest_sector,
                    [str(name) for name in sectors[largest_sector]],
                    maximum=MAX_SECTOR),
    ]

    # --- Optimise at every rebalance -------------------------------------
    #
    # Optimising once and holding would compare a stale portfolio against a
    # rebalanced index, and attribute to the constraints a difference that was
    # really staleness. Each rebalance gets its own solve against the same
    # constraints.
    schedule = {}
    binding_counts: dict[str, int] = {}

    for date, targets in sorted(index.weight_snapshots.items()):
        solution = minimise_tracking_error(targets, constraints, risk)
        schedule[date] = dict(solution.weights)

        for constraint in solution.binding:
            binding_counts[constraint.kind] = binding_counts.get(
                constraint.kind, 0) + 1

    heading("Optimisation")
    print(f"  rebalances       {len(schedule):>8}")
    print(f"  position limit   {MAX_POSITION:>8.0%}")
    print(f"  sector limit     {MAX_SECTOR:>8.0%}   on {largest_sector}")
    print(f"  binding, by kind {binding_counts}")

    # --- Ex ante versus realised -----------------------------------------
    final_index = latest_weights(index)
    final_optimised = schedule[max(schedule)]

    expected = active_risk_contributions(final_optimised, final_index,
                                         risk.covariance)

    # --- Backtest both ---------------------------------------------------
    plain = BacktestEngine(start_date=config.start, end_date=config.end,
                           initial_capital=CAPITAL, data_provider=fetcher,
                           target_index_result=index,
                           transaction_cost_bps=COSTS_BPS).run()

    optimised = BacktestEngine(start_date=config.start, end_date=config.end,
                               initial_capital=CAPITAL, data_provider=fetcher,
                               target_weights=schedule,
                               transaction_cost_bps=COSTS_BPS).run()

    heading("Tracking the index")
    show_metrics(plain.summary())

    heading("Tracking the optimised weights")
    show_metrics(optimised.summary())

    # --- Compare ---------------------------------------------------------
    plain_costs = sum(t.transaction_cost for t in plain.transactions)
    optimised_costs = sum(t.transaction_cost for t in optimised.transactions)

    heading("Side by side")
    print(f"  {'':<22}{'index':>14}{'optimised':>14}")
    print(f"  {'final NAV':<22}{plain.portfolio_nav.iloc[-1]:>14,.0f}"
          f"{optimised.portfolio_nav.iloc[-1]:>14,.0f}")
    print(f"  {'trades':<22}{len(plain.transactions):>14,}"
          f"{len(optimised.transactions):>14,}")
    print(f"  {'costs paid':<22}{plain_costs:>14,.0f}{optimised_costs:>14,.0f}")

    plain_return = plain.portfolio_nav.iloc[-1] / CAPITAL - 1
    optimised_return = optimised.portfolio_nav.iloc[-1] / CAPITAL - 1

    print(f"  {'total return':<22}{plain_return:>13.2%}"
          f"{optimised_return:>14.2%}")

    heading("Ex ante versus realised")
    realised = _realised_tracking_error(plain, optimised)

    print(f"  ex ante  (risk model)  {expected.volatility:>8.2%}")
    print(f"  realised (backtest)    {realised:>8.2%}")
    print()
    print("  The first is what the covariance predicted at the final")
    print("  rebalance; the second is what the two NAV paths actually did over")
    print("  the whole run. They are computed from different things and over")
    print("  different windows, so they will not agree - the size of the gap")
    print("  is the useful signal, not its existence.")

    print(f"\n  return difference        "
          f"{optimised_return - plain_return:>+8.2%}")
    print(f"  extra trading paid       "
          f"{optimised_costs - plain_costs:>+8,.0f}")
    print()
    print("  Read the return difference carefully. Constraints cost turnover:")
    print("  position and sector limits force trades a cap-weighted index")
    print("  never makes, and that shows up reliably in the costs line.")
    print()
    print("  What they do to *return* is not reliable in either direction. On")
    print("  this seed the constrained portfolio came out ahead, because")
    print("  capping the largest names avoided a drawdown they had. That is")
    print("  one path, not a finding: another seed can go the other way.")
    print("  Constraints are a risk decision, and judging them by a single")
    print("  realised return is exactly how a backtest misleads.")

    return 0


def _realised_tracking_error(plain,
                             optimised) -> float:
    """Annualised standard deviation of the difference in daily returns."""
    import numpy as np

    difference = (optimised.portfolio_nav.pct_change()
                  - plain.portfolio_nav.pct_change()).dropna()

    return float(difference.std() * np.sqrt(252))


if __name__ == "__main__":
    raise SystemExit(main())

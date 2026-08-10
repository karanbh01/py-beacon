# examples/04_optimised_index.py
"""
Optimise an index against real constraints, and read which of them bound.

    python examples/04_optimised_index.py

Needs the optimiser extra:

    pip install "py-beacon[optimise]"

## What to look at

Not the weights. **The binding constraints.** A constraint that bit changed the
answer; one that did not was never relevant, and an optimiser that reports only
weights leaves you unable to tell which was which. The whole reason to look at
a solution is to understand what shaped it.
"""
from _shared import (
    heading,
    index_definition,
    latest_weights,
    market_data,
    parse_arguments,
    show_weights,
)

from beacon.index.calculation import IndexCalculator
from beacon.optimise import (
    FullInvestment,
    GroupBounds,
    PositionBounds,
    minimise_tracking_error,
)
from beacon.risk import estimate_risk_model, risk_contributions

MAX_POSITION = 0.06
MAX_SECTOR = 0.25


def main() -> int:
    arguments = parse_arguments(__doc__ or "")
    dataset, fetcher, config = market_data(arguments.full)

    # --- The target the optimiser tracks ---------------------------------
    definition = index_definition(dataset, config, index_id="TARGET", cap=0.10)
    index = IndexCalculator(definition, fetcher).run(
        start_date=config.start, end_date=config.end)

    target = latest_weights(index)

    heading("Target index")
    show_weights(target, title="Cap-weighted, 10% cap")

    # --- The risk model --------------------------------------------------
    risk = estimate_risk_model(dataset.returns)

    heading("Risk model")
    print(f"  assets           {risk.diagnostics.assets:>8}")
    print(f"  observations     {risk.diagnostics.observations:>8}")
    print(f"  shrinkage        {risk.diagnostics.intensity:>8.3f}"
          f"   toward {risk.diagnostics.target}")
    print(f"  avg correlation  {risk.diagnostics.average_correlation:>8.3f}")
    print(f"  condition number {risk.diagnostics.condition_number:>8.1f}")
    print(f"  positive semi-def{risk.diagnostics.positive_semi_definite!s:>9}")
    print()
    print("  Shrinkage is on by default. A sample covariance is noisiest in")
    print("  its smallest eigenvalues, which is exactly what an optimiser")
    print("  inverts, so the raw estimate produces portfolios that look")
    print("  brilliant in sample and fall apart out of it.")

    # --- Constraints -----------------------------------------------------
    sectors = dataset.universe.groupby("SECTOR").groups
    largest_sector = max(sectors, key=lambda name: len(sectors[name]))
    members = [str(name) for name in sectors[largest_sector]]

    constraints = [
        FullInvestment(),
        PositionBounds(0.0, MAX_POSITION),
        GroupBounds(largest_sector, members, maximum=MAX_SECTOR),
    ]

    heading("Constraints")
    print("  full investment  weights sum to 1")
    print(f"  position bounds  0% to {MAX_POSITION:.0%} per name")
    print(f"  group bounds     {largest_sector} at most {MAX_SECTOR:.0%}"
          f"   ({len(members)} names)")

    # --- Solve -----------------------------------------------------------
    result = minimise_tracking_error(target, constraints, risk)

    heading("Solution")
    print(f"  converged        {result.diagnostics.converged!s:>8}")
    print(f"  status           {result.diagnostics.status:>8}")
    print(f"  iterations       {result.diagnostics.iterations:>8}")
    print(f"  objective        {result.diagnostics.objective:>8.6f}")
    print()
    show_weights(result.weights, title="Optimised")

    # --- What bound ------------------------------------------------------
    heading("Binding constraints")
    if result.binding:
        for constraint in result.binding:
            print(f"  {constraint.label:<34} {constraint.kind:<12}"
                  f" slack {constraint.slack:>.2e}")
        print()
        print("  These are the ones that shaped the answer. Relax any of them")
        print("  and the solution moves; relax the others and nothing happens.")
    else:
        print("  None bound: the unconstrained optimum already satisfied")
        print("  everything asked of it, so the constraints cost nothing here.")

    # --- What it did to the portfolio ------------------------------------
    heading("Target versus optimised")

    largest = sorted(((name, result.weights.get(name, 0.0) - weight)
                      for name, weight in target.items()),
                     key=lambda item: abs(item[1]), reverse=True)

    print("  Largest active positions:")
    for name, delta in largest[:6]:
        direction = "over" if delta > 0 else "under"
        print(f"    {name:<8} {delta:>+7.2%}  {direction}weight")

    before = risk_contributions(target, risk.covariance)
    after = risk_contributions(result.weights, risk.covariance)

    print(f"\n  target volatility     {before.volatility:>7.2%}")
    print(f"  optimised volatility  {after.volatility:>7.2%}")
    print(f"  names held            {sum(1 for w in dict(result.weights).values() if w > 1e-6):>7}"
          f"   of {len(target)}")
    print()
    print("  The objective was tracking error, not volatility, so a lower")
    print("  volatility here is a side effect rather than the goal. Optimising")
    print("  for one thing and reading another is how a result gets")
    print("  misreported.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

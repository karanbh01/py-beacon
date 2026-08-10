# examples/02_backtest_analysis.py
"""
Backtest an index, then analyse it: statistics, charts, weights, attribution.

    python examples/02_backtest_analysis.py
    python examples/02_backtest_analysis.py --full

Writes charts to `examples/output/`. Charting needs the `plot` extra:

    pip install "py-beacon[plot]"

## What is worth reading here

**Held weights are not target weights.** Between rebalances the index holds
fixed units, so weights drift with relative performance. Attributing with
targets would credit a return the index did not earn to a position it did not
hold, so `drifted_weights()` reconstructs what was actually held.

**Contributions sum to the total exactly.** Returns compound while
contributions add, so a naive sum undershoots — Carino linking corrects it, and
the residual printed at the end should sit at machine epsilon. A residual that
is not tiny means an assumption broke upstream.
"""
from pathlib import Path

from _shared import (
    heading,
    index_definition,
    latest_weights,
    market_data,
    parse_arguments,
    show_metrics,
    show_weights,
)

from beacon.analysis import attribute, concentration, drifted_weights
from beacon.backtest.engine import BacktestEngine
from beacon.index.calculation import IndexCalculator

OUTPUT = Path(__file__).resolve().parent / "output"
COSTS_BPS = 10.0


def main() -> int:
    arguments = parse_arguments(__doc__ or "")
    dataset, fetcher, config = market_data(arguments.full)

    definition = index_definition(dataset, config, index_id="ANALYSE", cap=0.10)
    index = IndexCalculator(definition, fetcher).run(
        start_date=config.start, end_date=config.end)

    backtest = BacktestEngine(start_date=config.start, end_date=config.end,
                              initial_capital=10_000_000.0,
                              data_provider=fetcher,
                              target_index_result=index,
                              transaction_cost_bps=COSTS_BPS).run()

    # --- Summary ---------------------------------------------------------
    heading("Summary statistics")
    show_metrics(backtest.summary())

    # --- Concentration ---------------------------------------------------
    targets = latest_weights(index)
    measures = concentration(targets)

    heading("Concentration at the last rebalance")
    print(f"  constituents     {measures.assets:>8}")
    print(f"  herfindahl       {measures.herfindahl_index:>8.4f}")
    print(f"  effective names  {measures.effective_assets:>8.1f}"
          f"   (of {measures.assets})")
    print(f"  largest weight   {measures.largest_weight:>8.2%}")
    print()
    print("  Effective names below the real count is the whole point of the")
    print("  measure: it says how many *equally weighted* names would be as")
    print("  concentrated as this, which is what a raw count cannot.")

    # --- Weights, target versus held -------------------------------------
    prices = dataset.market.data["CLOSE"].unstack("IDENTIFIER")
    held = drifted_weights(index.weight_snapshots, prices)
    final = {name: float(value) for name, value in held.iloc[-1].items()}

    heading("Weights")
    show_weights(targets, title="Target, at the last rebalance")
    print()
    show_weights(final, title="Held, at the end of the run")

    active = sorted(((name, final.get(name, 0.0) - weight)
                     for name, weight in targets.items()),
                    key=lambda item: abs(item[1]), reverse=True)

    print("\n  Largest drifts since the rebalance:")
    for name, delta in active[:5]:
        print(f"    {name:<8} {delta:>+7.2%}")
    print(f"\n  These sum to {sum(d for _, d in active):+.2%} : held weights")
    print("  renormalise to one, so what one name gained another lost.")

    # --- Attribution -----------------------------------------------------
    asset_returns = prices.pct_change().reindex(held.index)
    period_returns = (held.shift(1) * asset_returns).sum(axis=1)

    attribution = attribute(period_returns, held, asset_returns,
                            cap_drag=-0.003, cost_drag=-0.004)

    heading("Attribution")
    contributions = sorted(attribution.contributions,
                           key=lambda row: row.contribution, reverse=True)

    print("  Top contributors:")
    for row in contributions[:5]:
        print(f"    {row.asset_id:<8} {row.contribution:>+7.2%}"
              f"   (avg weight {row.average_weight:>6.2%},"
              f" return {row.total_return:>+8.2%})")

    print("\n  Largest detractors:")
    for row in contributions[-3:]:
        print(f"    {row.asset_id:<8} {row.contribution:>+7.2%}")

    print(f"\n  total return     {attribution.total_return:>+8.2%}")
    print(f"  sum of parts     {sum(r.contribution for r in contributions):>+8.2%}")
    print(f"  cap drag         {attribution.cap_drag:>+8.2%}")
    print(f"  cost drag        {attribution.cost_drag:>+8.2%}")
    print(f"  residual         {attribution.residual:>+8.2e}")
    print()
    print("  The residual is the check. Contributions are Carino-linked, so")
    print("  they sum to the compounded total rather than approximately to")
    print("  it, and anything above machine epsilon means something broke.")

    # --- Charts ----------------------------------------------------------
    heading("Charts")
    try:
        _draw(index, backtest, attribution)
    except Exception as error:
        print(f"  skipped: {error}")
        print('  install the extra with: pip install "py-beacon[plot]"')

    return 0


def _draw(index,
          backtest,
          attribution) -> None:
    """Render the charts this run produced, if matplotlib is available."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from beacon.plot import use

    use("light")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for name, draw in (("level", index.plot.level),
                       ("weights", index.plot.weights),
                       ("performance", backtest.plot.performance),
                       ("annual_returns", backtest.plot.annual_returns),
                       ("contributions", attribution.plot.contributions)):
        draw()
        target = OUTPUT / f"{name}.png"
        plt.savefig(target, dpi=110)
        plt.close("all")
        print(f"  wrote {target.name}")

    print(f"  in {OUTPUT}")


if __name__ == "__main__":
    raise SystemExit(main())

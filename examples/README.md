# Examples

Runnable documentation. Each script generates its own data, so they work from a
clean checkout with no network and nothing prepared:

```bash
pip install -e ".[dev]"
python examples/01_index_and_backtest.py
```

Defaults are small — 40 names over four years — so a script finishes while you
are reading it. `--full` scales to 250 names over ten years. `--quiet`
suppresses library logging.

| Script | Shows |
| --- | --- |
| `01_index_and_backtest.py` | The smallest complete pass: define, calculate, backtest. Why the index level and the portfolio NAV are not the same number. |
| `02_backtest_analysis.py` | Summary statistics, concentration, target versus held weights, attribution with its drags, and the charts. Writes to `output/`. |
| `03_index_futures.py` | Cost-of-carry fair value, basis, implied repo, DV01 by bump-and-revalue, and the cost of rolling. |
| `04_optimised_index.py` | Optimising against real constraints, and reading which of them actually bound. |
| `05_optimised_backtest.py` | Backtesting the optimised weights against the index, and ex-ante versus realised tracking error. |

`02` needs the `plot` extra; `04` and `05` need `optimise`. Both are in `[dev]`.

Every script is executed by `tests/test_examples.py`. An example that stops
working is worse than no example, and it fails silently unless something runs
it.

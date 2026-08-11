# Examples

Runnable documentation, as Jupyter notebooks. Each one generates its own data,
so they work from a clean checkout with no network and nothing prepared:

```bash
pip install -e ".[dev]"
jupyter lab examples/
```

Open any notebook and run the cells top to bottom. They are numbered in a
reading order but **not chained** — no notebook depends on another having been
run, and each can be opened first.

| Notebook | Shows |
| --- | --- |
| `01_index_and_backtest.ipynb` | The smallest complete pass: define, calculate, backtest. Why the index level and the portfolio NAV are not the same number. |
| `02_backtest_analysis.ipynb` | Summary statistics, concentration, target versus held weights, attribution with its drags, and charts. |
| `03_index_futures.ipynb` | Cost-of-carry fair value, basis, implied repo, DV01 by bump-and-revalue, rolling, and convergence at expiry. |
| `04_optimised_index.ipynb` | Optimising against real constraints, and reading which of them actually bound. |
| `05_optimised_backtest.ipynb` | Backtesting the optimised weights against the index, and ex-ante versus realised tracking error. |

`01` and `03` need only the core library. `02` needs the `plot` extra; `04` and
`05` need `optimise`. All are in `[dev]`.

Defaults are small — 40 names over four years — so a notebook finishes while
you are reading it. Change `CONFIG` in the setup cell to scale up.

## Two conventions worth knowing

**Each notebook is standalone, and the setup is duplicated on purpose.** These
were scripts sharing a `_shared.py`, which meant none of them could be opened
and run on its own, or sent to somebody as a single file. A notebook whose
first cell fails on an import of a module you do not have is not runnable
documentation. Duplication is the right trade here.

**They are committed without outputs.** Stored outputs turn every rerun into a
diff of megabytes of base64. Clear them before committing (`jupyter nbconvert
--clear-output --inplace examples/*.ipynb`) — a test enforces it.

## They are tested

`tests/test_examples.py` executes every notebook through a real Jupyter kernel
and checks it runs, keeps its sections, and produces no `nan`. An example that
stops working is worse than no example, and it fails silently unless something
runs it.

# Worklog

Running notes on the packaging/server/plot batch (BN-54 → BN-83). This is the
narrative that does not fit in commit messages: what was found, what was
decided and why, and what is still owed. Newest entries at the top.

Issue numbering: **BN-*n* is GitHub issue #*(n+16)*** — BN-54 is #70, BN-83 is
#99. Commit subjects use the BN number; `Closes #N` uses the GitHub number.

---

## Standing notes

### Owed / not yet done

- **#73 (BN-57) is intentionally still open.** Two of its four invariants
  describe code that does not exist yet. See the BN-57 entry below.
- **Coverage headroom is 0.25pp.** The floor is 80% and the suite sits at
  80.25%. The next uncovered code merged will break the build. Weakest
  modules: `portfolio/reporting.py` (22%), `index/methodology.py` (40%).
- **Issues are not on the project board.** The `gh` token carries `repo`,
  `read:org` and `workflow` but not `project`, so `--project "py-beacon
  development"` could not be applied when the 30 issues were created. Fix with
  `gh auth refresh -s project`, then they can be added in bulk.
- **`us_tickers.csv`** sits untracked in the repo root. Not created by this
  work; left alone deliberately. Decide whether it belongs in the repo or in
  `.gitignore`.

### Reporting module will need both openpyxl and reportlab

Noted 2026-07-29. `openpyxl` is already wired up behind the `excel` extra and
guarded by `beacon._optional.require()`. `reportlab` will be needed alongside
it for PDF output, not instead of it — the two serve different outputs
(spreadsheet vs. paginated document) and should both be reachable.

BN-75 (#91) currently frames this as a timeboxed spike of *weasyprint vs
reportlab*. Treat reportlab as the expected outcome rather than an open
question, and add a `pdf` extra next to `excel` when it lands. The
`EXTRA_FOR_MODULE` table in `src/beacon/_optional.py` is where the new module
gets registered; the guard then produces the right
`pip install "py-beacon[pdf]"` message for free.

### Three tooling omissions that are deliberate

Each of these looks like a bug and will invite a "fix" that breaks something.
All three are commented in place in `pyproject.toml`.

1. **`ruff format` is not adopted.** It cannot express the CLAUDE.md signature
   style and enabling it rewrites ~8,870 lines, reverting BN-53.
2. **`pandas-stubs` is not a dependency.** It converts 68 real mypy findings
   into 137, of which 69 are spurious operator errors on Series arithmetic.
3. **`python_version` is not pinned in `[tool.mypy]`.** Pinning it makes mypy
   parse *third-party* stubs at that version too, and numpy's stubs use PEP 695
   syntax.

---

## BN-59 — CI workflow (#75, closed)

Matrix of 3 Python versions × 3 operating systems, plus a separate build job
that runs `twine check`, installs the built wheel into a clean environment,
asserts `beacon/py.typed` is inside the archive, and uploads wheel + sdist.

**The first run failed all nine test cells.** Four distinct causes, all real,
none of which local testing could have surfaced:

1. `uv pip install --system` cannot target a uv-managed interpreter, so ubuntu
   died with "No system Python installation found for Python 3.13".
2. On macOS/Windows the install proceeded but console scripts were not on
   PATH, so `ruff` was not found. Fixed 1 and 2 by using `actions/setup-python`
   for the version matrix and keeping uv purely for install speed.
3. **A local/CI divergence introduced during BN-56**: `pandas-stubs` was added
   to the dev extra while evaluating it, and removing it from the local
   environment did not remove it from `pyproject.toml`. Locally mypy ran
   without stubs and passed; CI installed `.[dev]` and got them, producing
   exactly the spurious errors BN-56 had documented as the reason to avoid
   them. Worth internalising: *uninstalling a package locally is not the same
   as removing the dependency.*
4. `python_version = "3.11"` in the mypy config made mypy parse numpy's
   `__init__.pyi` as 3.11, which fails on its PEP 695 `type` statements. Every
   3.12/3.13 cell died inside numpy rather than on our code, and the error
   masked all further checking.

Build runs once on ubuntu rather than per matrix cell: the wheel is
`py3-none-any`, so nine builds would produce nine identical artifacts with
colliding names.

## BN-58 — Pre-commit hooks (#74, closed)

Whitespace/merge-conflict/TOML/YAML checks plus `ruff-check --fix` on commit;
`mypy` on the **pre-push** stage only, because it is too slow to run per
commit. mypy is a `local`/`system` hook rather than `mirrors-mypy`: it needs
the `[tool.mypy]` settings from pyproject to scope itself to `src/beacon`, so
`pass_filenames` is false.

`ruff-pre-commit` is pinned to the same version as the `dev` extra so a local
run and CI cannot disagree. That pin needs to move in lockstep — Dependabot's
`dev-tooling` group exists to make that one PR rather than two.

Setup is two commands, both needed:

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

## BN-57 — Pytest config, coverage floor, property tests (#73, still open)

**Delivered:** pytest configuration, an 80% coverage gate, and 16 Hypothesis
properties over invariants that genuinely hold today — weighting schemes,
divisor continuity, portfolio cash accounting, return/level recompounding, and
cost-of-carry pricing.

**Deferred, and why the issue stays open.** Two of the four invariants the
issue names describe features that do not exist:

- *"weights sum to 1.0 after capping"* and *"iterative cap converges"* — there
  is no weight capping anywhere. Only `EqualWeighted` and `MarketCapWeighted`
  exist. The uncapped sum-to-1.0 property *is* tested.
- *"shrunk covariance is PSD"* — there is no covariance estimation in
  `analysis/risk.py`, only scalar volatility/Sharpe metrics. Blocked on BN-73.
- *"attribution reconciles"* — `analysis/attribution.py` is a stub exposing a
  portfolio-vs-benchmark return difference. There are no per-constituent
  contributions and no cap/cost drags to reconcile. Blocked on BN-71, and note
  the CLAUDE.md caveat that the index level is market-cap-based and does not
  equal the weighted sum of constituent returns except when price paths are
  proportional.

**Incidental but significant: the suite went from 12m57s to 13s.** Same 509
tests, no skips. Adding `testpaths = ["tests"]` was the whole cause —
`--collect-only` alone still exceeds three minutes without it, so the cost was
pytest walking the repo tree, not the tests themselves.

**Three properties needed their input domain narrowed rather than their
tolerance loosened**, which is the interesting part and is documented inline in
`tests/test_properties.py`:

- `get_weights()` does *not* sum to 1.0 in general — cash has no entry in the
  mapping, so any uninvested cash leaves the sum short. Only
  `get_holdings_summary()` (which adds a `CASH` row) sums to 1 unconditionally.
- The return/level round-trip is genuinely ill-conditioned when adjacent levels
  differ by orders of magnitude: the return approaches −100%, `1 + r` cancels,
  and the reconstructed level retains only absolute precision. Levels are now
  generated as a bounded geometric walk — the regime the identity is meant for.
- Portfolio cash accumulates by sequential subtraction while the expected value
  is one sum then one subtraction, leaving a residual at machine precision on
  the traded notional.

None masked a library defect; all three were checked against the code.

## BN-56 — ruff + strict mypy + py.typed (#72, closed)

**mypy found a real, shipped bug.** `ReportGenerator.generate_holdings_report_excel`
called `portfolio.get_holdings_summary(data_provider, valuation_date)`, but
that method has taken **zero** arguments since the Portfolio refactor removed
its DataFetcher dependency. Every call raised `TypeError` — the Excel holdings
report had never worked, and no test covered it. Fixed, and the dead
`data_provider` parameter removed (no working caller could have existed). It
now writes a real .xlsx.

Three further annotation defects: derivatives `market_data` was
`dict[str, float]` while callers pass a list of dividend tuples under
`discrete_dividends`; `ETF.get_tracking_performance` declared `dict[str, float]`
but returns an error *string* on one path; and two Optionals were used without
narrowing. 68 errors resolved with **zero** `# type: ignore`.

**The `ruff format` decision was yours to make and you made it:** keep the
CLAUDE.md signature style, adopt the linter only. Worth recording *why* the two
are irreconcilable — ruff/black put the opening paren then indent, whereas the
project style keeps the first parameter on the `def` line and aligns the rest
beneath it. There is no ruff setting that produces the latter.

## BN-55 — Optional extras (#71, closed)

`beacon._optional.require(module, feature)` raises `MissingDependencyError`
naming the extra to install. It subclasses both `BeaconError` and `ImportError`
so existing `except ImportError` handlers keep working, and every guarded
module must be registered in `EXTRA_FOR_MODULE` — an unregistered one raises
`KeyError`, which is a packaging bug rather than a user-facing condition.

Verified the honest way rather than by mocking: built the wheel, installed it
into a venv holding only pandas/numpy/pydantic, and confirmed both that
`import beacon` succeeds and that the Excel path raises the actionable message.

Two judgement calls: added an `excel` extra the issue did not list (openpyxl is
a real optional import that needed a home — see the reporting note above), and
deleted `requirements.txt`, whose pins contradicted the pyproject and which
nothing referenced.

## BN-54 — PEP 621 packaging and src layout (#70, closed)

The previous `pyproject.toml` was the **unedited PyPA sample template** — it
declared `sample = "sample:main"` as a console script and pointed its URLs at
`github.com/pypa/sampleproject`. None of that metadata was ever real.

The src layout prompted a good question worth recording: pandas and numpy are
flat, so why isn't this? Because they are compiled — you *cannot* import them
from a source tree without building the extensions first, so they get
tree-vs-installed isolation for free from the compiler. A pure-Python package
has no such protection: the tree just imports, silently, and you end up testing
something other than what you ship. The distribution is `py-beacon`; the import
package is still `beacon`, and `src` never appears in an import.

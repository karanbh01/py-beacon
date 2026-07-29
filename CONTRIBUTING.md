# Contributing to Beacon

## Getting set up

```bash
git clone https://github.com/karanbh01/py-beacon.git
cd py-beacon
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install                        # lint + whitespace checks on commit
pre-commit install --hook-type pre-push   # strict type check on push
```

Beacon uses a `src/` layout, so the package resolves through the editable
install rather than the working tree. That is deliberate: it means the tests
exercise what a user would actually get from the wheel. After the one-time
install, everything works from any directory.

## The gates

Three things must be clean before anything is committed. CI runs all of them
across Python 3.11–3.13 on Linux, macOS and Windows.

```bash
ruff check .        # lint
mypy                # strict type check of src/beacon
pytest -q           # full suite plus the coverage gate
```

Coverage is gated at 80% and currently sits just above it, so new code
generally needs tests to keep the build green.

**There is no formatter.** `ruff format` is deliberately not adopted: it cannot
produce the signature style described below, and enabling it would rewrite
most of the codebase. Formatting is a review concern, not a tool-enforced one.

## Coding conventions

These are enforced by review rather than by tooling, so they matter.

- Keep files to roughly 500 lines; split a module before it grows well past
  that.
- Nest at most 3 levels deep inside a function. Refactor deeper logic into
  helpers or early returns.
- Separate logical steps within a function with blank lines.
- **Signatures with more than one parameter** (`self` and `cls` count) put each
  parameter on its own line: the first stays on the `def` line, and every
  subsequent one aligns directly beneath it. Zero- and one-parameter
  signatures stay on one line. Multi-line calls use the same alignment.

  ```python
  def __init__(self,
               fund_id: str,
               portfolio: Portfolio,
               management_fee_bps: int = 0):
      ...

  engine = BacktestEngine(start_date=start,
                          end_date=end_date,
                          initial_capital=capital)
  ```

- Docstrings are Google style (`Args:` / `Returns:` / `Raises:`). Types come
  from the annotations, so do not repeat them in the docstring. The API
  reference is generated from these, and `mkdocs build --strict` fails on a
  docstring that stops parsing.

## Tests

Tests use synthetic, deterministic data only — no network, no fixture files.
Follow the existing shape: module-level constants for the universe and dates,
helper builders, and test classes grouped by behaviour (`TestConstruction`,
`TestTracking`, …). Integration tests live in `*_integration.py`; unit tests
stay in the per-module file. Property-based tests go in
`tests/test_properties.py`.

When a property test fails, work out whether the invariant is genuinely wrong
before touching the tolerance. Narrowing the generated input domain to the
regime the invariant actually holds for is usually the honest fix; loosening a
tolerance to make a failure disappear usually hides something.

## Issues, branches and commits

Work is tracked as GitHub issues titled `[BN-##] ...`. The BN number and the
GitHub issue number are not the same — this batch runs BN-*n* = issue
#*(n+16)*.

Commit messages use the `[BN-##]` prefix, a bulleted body describing the
change and its reasoning, and end with `Closes #<github-issue-number>` using
the **GitHub** number. If an issue can only be partly delivered, use `Refs #N`
instead and leave the issue open with a comment explaining what remains.

Labels group the work: `packaging`, `server`, `plot`, `deferred`, plus the
domain labels (`index`, `backtest`, `portfolio`, `fund`, `asset`, `analysis`,
`derivative`, `documentation`, `enhancement`).

Every issue's implementation is verified by tests before committing, and the
suite must stay green.

## Releasing

1. Update `__version__` in `src/beacon/__init__.py`.
2. Move the `[Unreleased]` entries in [CHANGELOG.md](CHANGELOG.md) under the
   new version heading.
3. Tag `vX.Y.Z`. The tag must match `__version__` — the release workflow
   checks this and fails the build otherwise.
4. The workflow builds the wheel and sdist, runs `twine check`, verifies the
   wheel installs and imports, and creates a **draft** GitHub release with both
   artifacts and the changelog section as the body.
5. Review and publish the draft. Publishing triggers the documentation deploy.

PyPI publishing is wired but disabled while the repository is private.

## Versioning and deprecation policy

Beacon follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the major version is `0`, the public API may change in any release — the
project is pre-1.0 and the surface is still settling. Breaking changes are
recorded under **Changed** or **Removed** in the changelog.

### Deprecation policy

Once 1.0 ships, anything exported from a subpackage's `__all__` is public and
covered by the following:

- A deprecated name keeps working for at least one minor release before
  removal, and emits a `DeprecationWarning` naming its replacement.
- The deprecation is recorded under **Deprecated** in the changelog in the
  release that introduces it, and under **Removed** in the release that drops
  it.
- Removals only happen in a major release.

Anything with a leading underscore, and anything not in an `__all__`, is
internal and may change at any time.

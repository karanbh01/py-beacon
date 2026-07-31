# 4. Chart images are compared at zero tolerance, with generated baselines

**Status:** Accepted
**Date:** 2026-07-31
**Issue:** [BN-82] (#98)

## Context

Charts are the one part of this library where a property test cannot say
whether the output is right. `test_plot.py` asserts that a line has the accent
colour and that an annotated total matches the bars beside it; none of that
notices a label that started overlapping, a legend that moved onto the data, or
a colour that shifted by a shade.

## Decision

**pytest-mpl against committed baselines, at zero tolerance, in both styles.**

Eighteen baselines — nine charts, light and dark — generated from
`beacon.testing.dataset` and committed under `tests/baseline/`.

## Why zero tolerance

A generous tolerance passes exactly the changes worth catching. The whole
category of defect here is "something moved slightly and nobody looked", and a
threshold that admits a few pixels of drift admits precisely that. Verified by
changing the series line width from 1.5pt to 1.6pt: six baselines fail.

## Why this can work at all

Two earlier decisions make deterministic images possible, and without either
one a baseline would only be valid on the machine that generated it:

- `beacon.testing.dataset` builds its price paths from `+` and `*` alone, never
  `exp` or `log`, so the numbers are bit-identical across platforms (BN-95).
- The beacon styles pin `DejaVu Sans`, which ships with matplotlib, so text
  metrics do not depend on what fonts a machine happens to have.

## Consequences

- **Without `--mpl` the tests still run** as smoke tests: the figure is built
  and discarded. A contributor without baselines, or on a machine whose font
  stack differs, gets the charts exercised rather than a wall of failures they
  cannot act on. CI passes `--mpl`, which is where drift must fail.
- **Updating a baseline is a deliberate act.** The command is in the test
  module's docstring:

      pytest tests/test_plot_images.py --mpl-generate-path=tests/baseline

  Regenerating shows up as a binary diff in review, which is the point — a
  changed chart should be looked at.
- **The gallery is generated at docs build**, not committed. A gallery that can
  go stale becomes a record of what the charts used to look like, and nobody
  knows which images are still true. Light and dark are shown together rather
  than behind a toggle: the reason for having two styles is that a reader can
  see whether the dark one works, and a toggle hides that comparison.

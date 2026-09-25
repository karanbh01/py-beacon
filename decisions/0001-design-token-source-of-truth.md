# 1. Design tokens are vendored, with a drift check

**Status:** Accepted
**Date:** 2026-07-30
**Issue:** [BN-94] (#113)

## Context

Beacon's charts are meant to sit inside the desktop client without looking like
a different product, which means they have to draw with the client's colours.
Those colours are generated from Figma and live in
[`tokens/colors.json`](https://github.com/karanbh01/beacon-ui/blob/main/tokens/colors.json)
in the **beacon-ui** repository, where a build step turns them into CSS
variables and TypeScript constants.

BN-77 specifies that the `"beacon"` matplotlib style is generated from "the same
file the UI consumes". That file did not exist in this repository, so BN-77 —
and therefore every chart issue after it — had nothing to build on.

The colours are also not only chrome. The file carries a `raw` section of
mode-independent values: the three heatmap stops that BN-81's correlation
colormap needs, and the report-page ink that BN-97's PDF output needs. So this
decision gates more than the style sheet.

## Decision

**Vendor a copy of `colors.json` inside the package, and fail CI when it drifts
from beacon-ui's.**

The copy lives at `src/beacon/tokens/colors.json` — inside the package, not at
the repository root. `beacon.tokens` reads it through `importlib.resources`.

`scripts/check_token_drift.py` compares the two copies and runs as its own CI
job.

## Options considered

**(a) Vendor here, with a drift check — chosen.** The package stays
self-contained: it installs from PyPI and renders charts with no network and no
knowledge of a second repository. Divergence is caught by CI rather than by
someone noticing a chart looks wrong.

**(b) Generate from beacon-ui at build time.** Makes beacon-ui a build
dependency of a Python package, so `pip install py-beacon` from a source
distribution would need to reach GitHub. It also could not work offline, and it
would couple the release of one repository to the availability of another. The
coupling buys nothing that (a)'s drift check does not.

**(c) Define independently and hand-sync.** Rejected explicitly. It is what
happens by default when nobody decides, and its failure mode is silent: the two
drift, charts stop matching the application, and nothing reports it. The issue
called this out as the default to avoid, and it is.

## Consequences

- The values in `src/beacon/tokens/colors.json` are **not editable here**. A
  change made in this repository and not in beacon-ui is drift, and the check
  will reject it. The design system does not live here.
- Updating tokens is a two-step flow: change them in beacon-ui, then copy the
  file across. The drift check tells you when you have done the first and
  forgotten the second.
- The comparison is **semantic, not byte-for-byte**. Both files are parsed and
  their colour values compared, so line endings, key order and whitespace do
  not matter — a byte comparison would fail on a Windows checkout for no useful
  reason. Descriptions and comments are ignored too: rewording prose is not a
  design change and should not fail anyone's build.
- An unreachable upstream is a **failure, not a pass**. A check that could not
  run has not run. It exits with a distinct code so a network blip is
  distinguishable from a real design-system change.
- beacon-ui had to become public for this to work on every run. While it was
  private the check would have needed a personal access token stored as a
  secret, would have been skipped whenever that secret was absent — which was
  its actual state — and would therefore have shipped switched off.

## Notes

The file records where each value came from, per mode: `figma` for a verified
variable, `provisional` for an invented placeholder, `new` for something added
in beacon-ui that has not been mirrored back into Figma. A placeholder renders
exactly as convincingly as a real colour, so `beacon.tokens.unverified()`
surfaces them rather than leaving the distinction buried in the JSON.

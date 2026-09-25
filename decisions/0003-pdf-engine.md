# 3. reportlab, not weasyprint, for PDF rendering

**Status:** Accepted
**Date:** 2026-07-31
**Issue:** [BN-75] (#91), implemented in [BN-97] (#116)

## Context

BN-75 called for a timeboxed spike comparing weasyprint and reportlab for
`POST /reports/render`. The owner directed on 2026-07-29 that `openpyxl` and
`reportlab` are complementary — spreadsheet versus paginated document — and
that the spike should treat **reportlab as the expected outcome** and be
timeboxed accordingly.

BN-97 then built the PDF foundation on reportlab. This ADR records the decision
rather than re-running a spike whose answer was already acted on; writing it
down afterwards is worth doing because the reasoning is not obvious from the
code, and the next person will otherwise wonder why the HTML-to-PDF route was
not taken.

## Decision

**reportlab.**

## Why

**Determinism.** A report is a record. Two people running the same template on
the same data should be able to confirm they got the same thing by comparing
hashes. reportlab has an `invariant` mode that pins the creation timestamp and
document id; weasyprint's output varies between runs, and the usual workaround
is post-processing the PDF to strip metadata, which is a second tool in the
chain and a second thing to get wrong.

**No browser engine.** weasyprint pulls in a CSS layout stack — Pango, cairo,
GDK-PixBuf on some platforms — which is a substantial native dependency for a
local desktop-companion process, and one that behaves differently across the
three operating systems this project's CI matrix covers. reportlab is pure
Python plus a small set of wheels.

**The output is not a web page.** A factsheet is a fixed-size, precisely placed
document. Describing it as HTML and asking a layout engine to paginate the
result adds an indirection whose failure mode is "it moved slightly", and
whose fix is fighting CSS. Drawing to a canvas at known coordinates is more
direct for this kind of artefact, and the block model already describes the
page in terms of stacked blocks rather than flowing content.

**It composes with the existing Excel path.** `beacon.portfolio.reporting`
writes spreadsheets through openpyxl for numbers someone will pick up and work
with. reportlab covers the paginated document meant to be read as laid out.
Neither replaces the other, which was the owner's framing.

## What weasyprint would have been better at

Honestly: reflowing content, and styling from the same CSS the client uses. If
a future report needs long prose that paginates naturally, or needs to share a
stylesheet with the UI, that argument comes back. It does not apply to a
one-page factsheet with fixed blocks.

## Consequences

- The renderer draws at coordinates and refuses when content does not fit
  rather than paginating, which is a real limitation and is documented where it
  lives. Pagination is future work, not a gap this decision created.
- Charts arrive as images from the plotting layer rather than as HTML/SVG the
  engine lays out. That is why `Chart` blocks hold a path.
- The `pdf` extra is `reportlab` alone, so a caller who never renders a PDF
  installs nothing extra.

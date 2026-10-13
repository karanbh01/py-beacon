---
title: Charts
description: "Every chart class and setting: the result charts, the chart classes, Table, interactive charts, Style and its settings groups."
---

# Charts

Every chart class and setting in `beacon.plot`. [Charts](../concepts/charts.md)
explains how to use them, and the [Gallery](../gallery.md) shows each one.

**Chart classes**, each drawn with matplotlib and made interactive with
`.interactive()`:

| Class | Draws |
| --- | --- |
| `Line` | One or more series over dates or numbers, with an optional second axis |
| `Area` | Parts of a whole stacked, or one series filled from zero |
| `Column` | Values by category, upright |
| `Bar` | Values by category, across |
| `Combo` | Columns on the left axis and a line on the right |
| `Scatter` | One measure against another |
| `Histogram` | Counts in equal bins |
| `Heatmap` | A table of values as colour |
| `Grid` | Several charts under one heading |
| `Table` | A DataFrame as a table, as an image or as HTML |

Every chart class is a `Chart`: `.save()`, `.show()`, `.figure`, `.ax`,
`.axes`, `.layout` and `.interactive()`. Results draw through their `.plot`
accessor (`PlotAccessor`), and `compare()` puts several results on one chart.

**Settings.** A `Style` holds every setting, in groups: `SizeSettings`,
`HeadingSettings`, `TextSettings`, `LineSettings`, `AxesSettings`,
`LegendSettings`, `NotesSettings`, `ColorSettings` and `TableSettings`. Each
group's fields are its settings, and `configure()`, `settings()` and `style=`
take them by their short names, which the
[settings table](../concepts/charts.md#styles) lists with their defaults.
`use()` sets the theme, `palette()` returns a theme's named colours and
`reset()` goes back to Beacon's design.

**Interactive charts.** `.interactive()` returns a `BeaconFigure`
(`beacon.plot.interactive.figure`), a plotly figure whose `write_html()` and
`to_html()` carry the chart's fonts and its popup.

::: beacon.plot

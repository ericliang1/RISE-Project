"""Editable PowerPoint export of the SHAP map-contribution figure.

Rebuilds shap_map_summary.{pdf,png,svg} as NATIVE PowerPoint chart objects
(not an embedded picture), so every number, color, label, and axis is
editable in PowerPoint directly:

  Panel A: a native clustered bar chart (right-click -> Edit Data in
           Excel), one series per model, mean |Shapley value| per map.
  Panel B: a native XY scatter chart (DeepSets, per-scenario Shapley
           values), points binned into three mast-count colors — a
           continuous colormap has no native scatter-chart equivalent in
           PowerPoint, so masts are bucketed into 3 editable series
           instead. A 4th series holds the black diamond mean markers; a
           5th, invisible series anchored at the left axis edge carries
           the row-name data labels (Source evidence, Sensor visibility,
           Wind-error sensitivity, Fit quality), so labels never sit
           inside the point cloud regardless of where a map's mean falls.

Reads ONLY the already-computed shap_map_contributions.csv — no model or
GPU access, no numbers recomputed. For editability (thousands of native
scatter points make PowerPoint sluggish), Panel B plots a fixed-seed
subsample of the 2,000 DeepSets scenarios; the mean diamonds are computed
from the FULL sample regardless.

Usage (from repo root):
  python "Poster Graphics v2/shap/export_pptx.py"
"""
import csv
import pathlib

import numpy as np
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (XL_CHART_TYPE, XL_LEGEND_POSITION,
                             XL_LABEL_POSITION, XL_MARKER_STYLE,
                             XL_TICK_LABEL_POSITION)
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = pathlib.Path(__file__).resolve().parent
CSV_PATH = HERE / "shap_map_contributions.csv"
OUT_PATH = HERE / "shap_map_summary.pptx"

N_SUBSAMPLE = 400          # DeepSets scenarios plotted in the panel-B scatter
SEED = 0

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
SURF = RGBColor(0xFC, 0xFC, 0xFB)
GRID = RGBColor(0xE1, 0xE0, 0xD9)

MODEL_C = {"DeepSets": RGBColor(0x2A, 0x78, 0xD6),
           "GNN": RGBColor(0xC9, 0x40, 0x40),
           "Set Transformer": RGBColor(0x1B, 0xAF, 0x7A)}
MAST_BINS = [("4–6 masts", (4, 6), RGBColor(0x49, 0x03, 0xA0)),
             ("7–9 masts", (7, 9), RGBColor(0xCC, 0x47, 0x78)),
             ("10–12 masts", (10, 12), RGBColor(0xFD, 0xC6, 0x27))]
MAP_ORDER = ["Source evidence", "Sensor visibility", "Wind-error sensitivity",
            "Fit quality"]
# panel-B x-axis bounds (real data spans -3.02..7.19).  The left side is
# padded well past the data to reserve a dedicated gutter for the row-name
# labels, so even the longest label ("Wind-error sensitivity") clears the
# point cloud instead of merely nudging past its edge.
X_AXIS_MIN, X_AXIS_MAX = -7.0, 7.6
MAP_KEY = {"Source evidence": "shapley_source_evidence",
          "Sensor visibility": "shapley_sensor_visibility",
          "Wind-error sensitivity": "shapley_wind_error_sensitivity",
          "Fit quality": "shapley_fit_quality"}
Y_BASE = {m: 4 - k for k, m in enumerate(MAP_ORDER)}   # 4,3,2,1 top-to-bottom


def load_rows():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def set_font(font, size, color, bold=False, name="Calibri"):
    font.size = Pt(size)
    font.bold = bold
    font.name = name
    font.color.rgb = color


def add_textbox(slide, left, top, width, height, text, size, color,
                bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_font(run.font, size, color, bold)
    return box


def style_axis(axis, title=None, title_size=13, tick_size=12,
               color=INK2, grid=True, min_=None, max_=None, major=None):
    axis.format.line.color.rgb = GRID
    axis.tick_labels.font.size = Pt(tick_size)
    axis.tick_labels.font.color.rgb = color
    if title:
        axis.has_title = True
        axis.axis_title.text_frame.text = title
        r = axis.axis_title.text_frame.paragraphs[0].runs[0]
        set_font(r.font, title_size, color)
    if grid:
        axis.has_major_gridlines = True
        axis.major_gridlines.format.line.color.rgb = GRID
        axis.major_gridlines.format.line.width = Pt(0.75)
    else:
        axis.has_major_gridlines = False
    if min_ is not None:
        axis.minimum_scale = min_
    if max_ is not None:
        axis.maximum_scale = max_
    if major is not None:
        axis.major_unit = major


def build_panel_a(slide, rows, left, top, width, height):
    models = ["DeepSets", "GNN", "Set Transformer"]
    means = {m: {} for m in models}
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        for name in MAP_ORDER:
            vals = [abs(float(r[MAP_KEY[name]])) for r in mrows]
            means[m][name] = float(np.mean(vals))

    # PowerPoint's BAR_CLUSTERED plots the first category at the BOTTOM
    # (opposite of a normal reading order), so the category list is
    # reversed here to make the canonical order (Source evidence first)
    # appear at the top, matching Panel B and every other poster figure.
    display_order = list(reversed(MAP_ORDER))
    data = CategoryChartData()
    data.categories = display_order
    for m in models:
        data.add_series(m, [means[m][name] for name in display_order])

    gframe = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, left, top,
                                    width, height, data)
    chart = gframe.chart
    chart.has_title = True
    chart.chart_title.text_frame.text = "A. Mean absolute contribution"
    r = chart.chart_title.text_frame.paragraphs[0].runs[0]
    set_font(r.font, 16, INK, bold=True)

    for i, m in enumerate(models):
        series = chart.plots[0].series[i]
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = MODEL_C[m]
        series.format.line.fill.background()
        series.has_data_labels = True
        dl = series.data_labels
        dl.show_value = True
        dl.number_format = "0.00"
        dl.number_format_is_linked = False
        dl.font.size = Pt(11)
        dl.font.color.rgb = INK2
        dl.position = XL_LABEL_POSITION.OUTSIDE_END

    chart.plots[0].gap_width = 40
    chart.plots[0].overlap = -8

    style_axis(chart.value_axis, "mean |Shapley value| (nats)",
              min_=0, tick_size=12)
    style_axis(chart.category_axis, grid=False, tick_size=13)

    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(12)
    chart.legend.font.color.rgb = INK2
    return means


def bin_masts(n):
    for label, (lo, hi), color in MAST_BINS:
        if lo <= n <= hi:
            return label, color
    return MAST_BINS[-1][0], MAST_BINS[-1][2]


def build_panel_b(slide, rows, left, top, width, height):
    ds = [r for r in rows if r["model"] == "DeepSets"]
    rng = np.random.default_rng(SEED)
    jitter_all = {name: rng.normal(0, 0.11, len(ds)).clip(-0.32, 0.32)
                 for name in MAP_ORDER}
    sub_idx = rng.choice(len(ds), size=min(N_SUBSAMPLE, len(ds)),
                         replace=False)
    sub_idx.sort()

    data = XyChartData()
    for label, _, _ in MAST_BINS:
        s = data.add_series(label)
        for name in MAP_ORDER:
            y0 = Y_BASE[name]
            for i in sub_idx:
                r = ds[i]
                blabel, _ = bin_masts(int(r["n_masts"]))
                if blabel != label:
                    continue
                x = float(r[MAP_KEY[name]])
                y = y0 + float(jitter_all[name][i])
                s.add_data_point(x, y)

    means_series = data.add_series("Mean")
    for name in MAP_ORDER:
        vals = [float(r[MAP_KEY[name]]) for r in ds]
        means_series.add_data_point(float(np.mean(vals)), Y_BASE[name])

    # invisible series anchored at the left axis edge purely to carry the
    # row-name data labels — a fixed anchor (rather than labeling the mean
    # point directly) guarantees the text never sits inside the point
    # cloud regardless of where each map's mean happens to fall
    x_min, x_max = X_AXIS_MIN, X_AXIS_MAX
    label_series = data.add_series("Row labels")
    for name in MAP_ORDER:
        label_series.add_data_point(x_min + 0.015 * (x_max - x_min),
                                    Y_BASE[name])

    gframe = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, left, top,
                                    width, height, data)
    chart = gframe.chart
    chart.has_title = True
    chart.chart_title.text_frame.text = "B. Per-scenario values — DeepSets"
    r = chart.chart_title.text_frame.paragraphs[0].runs[0]
    set_font(r.font, 16, INK, bold=True)

    plot = chart.plots[0]
    for i, (label, _, color) in enumerate(MAST_BINS):
        series = plot.series[i]
        series.marker.style = XL_MARKER_STYLE.CIRCLE
        series.marker.size = 5
        series.marker.format.fill.solid()
        series.marker.format.fill.fore_color.rgb = color
        series.marker.format.fill.transparency = 0.35
        series.marker.format.line.fill.background()
        series.format.line.fill.background()   # no connecting line

    mean_series = plot.series[len(MAST_BINS)]
    mean_series.marker.style = XL_MARKER_STYLE.DIAMOND
    mean_series.marker.size = 11
    mean_series.marker.format.fill.solid()
    mean_series.marker.format.fill.fore_color.rgb = INK
    mean_series.marker.format.line.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    mean_series.marker.format.line.width = Pt(1.25)
    mean_series.format.line.fill.background()

    label_series = plot.series[len(MAST_BINS) + 1]
    label_series.marker.style = XL_MARKER_STYLE.NONE
    label_series.format.line.fill.background()
    for point, name in zip(label_series.points, MAP_ORDER):
        point.data_label.has_text_frame = True
        tf = point.data_label.text_frame
        tf.text = name
        run = tf.paragraphs[0].runs[0]
        set_font(run.font, 13, INK, bold=True)
        point.data_label.position = XL_LABEL_POSITION.RIGHT

    style_axis(chart.value_axis, "per-scenario Shapley value (nats, "
              "log p of true cell)", min_=X_AXIS_MIN, max_=X_AXIS_MAX,
              major=2)
    yaxis = chart.category_axis if hasattr(chart, "category_axis") else None
    # XY scatter has two value axes in pptx's model
    vaxis2 = chart.value_axis
    # second (vertical) axis is exposed the same way via plots[0].chart's
    # y axis; pptx names it value_axis for the primary (x) — the y axis is
    # reached via chart.value_axis is X for bar charts but for XY scatter
    # python-pptx's value_axis is actually the horizontal one; the
    # vertical one is accessed identically through chart.category_axis is
    # NOT valid for scatter. Use chart.value_axis for X and the chart's
    # second axis via chart._chartSpace for Y formatting below instead.
    chart.value_axis.tick_labels.number_format = "0"
    chart.value_axis.tick_labels.number_format_is_linked = False

    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(12)
    chart.legend.font.color.rgb = INK2
    hide_legend_entry(chart, len(MAST_BINS) + 1)   # "Row labels" anchor
    return sub_idx, ds


def hide_legend_entry(chart, index):
    """Remove one series from the legend by index (the invisible row-label
    anchor series should never appear as a legend entry)."""
    from pptx.oxml.ns import qn
    legend = chart._chartSpace.find(qn("c:chart")).find(qn("c:legend"))
    entry = legend.makeelement(qn("c:legendEntry"), {})
    idx = entry.makeelement(qn("c:idx"), {"val": str(index)})
    delete = entry.makeelement(qn("c:delete"), {"val": "1"})
    entry.append(idx)
    entry.append(delete)
    legend.insert(1, entry)   # after c:legendPos, before layout/overlay


def fix_y_axis(gframe_chart):
    """python-pptx's high-level API only names one value axis per chart
    type; for XY_SCATTER both axes are <c:valAx> elements in the XML.  Set
    the vertical one directly so masts/maps rows land at fixed positions
    the mean-label trick above assumes (0.4..4.6, hidden ticks)."""
    from pptx.oxml.ns import qn
    plot_area = gframe_chart._chartSpace.find(qn("c:chart")).find(
        qn("c:plotArea"))
    valaxes = plot_area.findall(qn("c:valAx"))
    # first is X (already styled via chart.value_axis); second is Y
    yax = valaxes[1]
    scaling = yax.find(qn("c:scaling"))
    for tag, val in (("c:max", "4.6"), ("c:min", "0.4")):
        el = scaling.find(qn(tag))
        if el is None:
            el = scaling.makeelement(qn(tag), {})
            scaling.append(el)
        el.set("val", val)
    major = yax.find(qn("c:majorUnit"))
    if major is None:
        major = yax.makeelement(qn("c:majorUnit"), {})
        yax.append(major)
    major.set("val", "1")
    tlp = yax.find(qn("c:tickLblPos"))
    if tlp is not None:
        tlp.set("val", "none")
    majorGrid = yax.find(qn("c:majorGridlines"))
    if majorGrid is not None:
        yax.remove(majorGrid)


def main():
    rows = load_rows()
    models_present = sorted(set(r["model"] for r in rows))
    print(f"loaded {len(rows)} rows from {CSV_PATH.name}, "
          f"models: {models_present}")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank layout

    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = SURF

    add_textbox(slide, Inches(0.35), Inches(0.22), Inches(12.6), Inches(0.6),
               "Which physics map matters? Exact Shapley values over the "
               "four maps", 24, INK, bold=True)
    add_textbox(slide, Inches(0.35), Inches(0.78), Inches(12.6), Inches(0.4),
               "feature = one map, absent = zero baseline · value = "
               "log p(true cell) · all 16 coalitions enumerated per "
               "scenario · 2,000 seed-1 test scenarios", 13, MUTED)

    top = Inches(1.35)
    height = Inches(5.55)
    means = build_panel_a(slide, rows, Inches(0.35), top, Inches(5.7),
                          height)
    sub_idx, ds = build_panel_b(slide, rows, Inches(6.3), top, Inches(6.7),
                                height)

    # fix the scatter chart's vertical axis (see fix_y_axis docstring)
    b_chart = slide.shapes[-1].chart
    fix_y_axis(b_chart)

    add_textbox(slide, Inches(6.3), Inches(6.95), Inches(6.7), Inches(0.45),
               f"colors bin mast count into 3 groups (a continuous "
               f"colormap has no native PowerPoint scatter equivalent); "
               f"{len(sub_idx)}/{len(ds)} scenarios plotted for "
               f"editability (fixed seed {SEED}); black diamonds are the "
               f"FULL-sample mean, matching Panel A's Source-evidence row",
               11, MUTED)
    add_textbox(slide, Inches(0.35), Inches(6.95), Inches(5.7), Inches(0.45),
               "source: shap_map_contributions.csv · DeepSets / GNN / "
               "Set Transformer, ensr, seed 1 · every chart is native "
               "and fully editable (right-click → Edit Data)",
               11, MUTED)

    prs.save(OUT_PATH)
    print(f"wrote {OUT_PATH}")
    print("\nmean |Shapley| per map (from the embedded chart data):")
    for name in MAP_ORDER:
        row = f"  {name:24s}" + "".join(
            f"{m}: {means[m][name]:.3f}  " for m in means)
        print(row)


if __name__ == "__main__":
    main()

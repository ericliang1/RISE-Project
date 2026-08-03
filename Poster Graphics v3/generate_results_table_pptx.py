"""Table I of the super-emitter benchmark as an editable PowerPoint table.

Same discipline as the figures: nothing transcribed by hand.  Per-seed
coverage / median equivalent radius / share of regions below the 50 m
facility scale are computed from the frozen measured-wind conformal
audits under csr-data-se48 (min-max across the three training seeds),
and the paired-effect columns with their 95% CIs come from
`results/se48_table_deltas_ci.json` (prespecified seed-1 run, 10,000
paired-bootstrap resamples — the file behind Fig. 06).

Output: `table1_results_se48.pptx` (one 16:9 slide, a real PowerPoint
table object, booktabs-style rules) in this folder.

Usage:
  source scripts/env.sh && python "Poster Graphics v3/generate_results_table_pptx.py"
"""
import copy
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import generate_poster_graphics_v3 as pg
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
SURF = RGBColor(0xFC, 0xFC, 0xFB)
FONT = "Arial"
MINUS = "−"
NETS = [("DeepSets", ""), ("GNN", "_gnn"), ("Set Transformer", "_st")]
CI_KEY = {"DeepSets": "DeepSets", "GNN": "GNN", "Set Transformer": "SetTransf"}


def seed_metrics(tag):
    """(coverage, median radius m, %<50 m) per seed for one audit tag."""
    out = []
    for s in pg.SEEDS:
        z = pg.audit(tag, s)
        r = pg.rad(z["sizes"])
        out.append((float(z["covered"].mean()), float(np.median(r)),
                    float((r < 50).mean() * 100)))
    return np.array(out)


def rng_fmt(vals, fmt):
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return lo if lo == hi else f"{lo}–{hi}"


def sgn(x, fmt):
    return (MINUS if x < 0 else "+") + fmt.format(abs(x))


def build_rows():
    ci = json.load(open(pg.REPO / "results" / "se48_table_deltas_ci.json"))
    rows = []
    for name, suff in NETS:
        cells = [name]
        for cond in ("nomaps", "ens"):
            m = seed_metrics(cond + suff)
            cells += [rng_fmt(m[:, 0], "{:.2f}"), rng_fmt(m[:, 1], "{:.0f}"),
                      rng_fmt(m[:, 2], "{:.0f}") + "%"]
        c = ci[CI_KEY[name]]
        cells.append(f"{sgn(c['d_radius_m'], '{:.1f}')} "
                     f"[{sgn(c['ci'][0], '{:.1f}')}, "
                     f"{sgn(c['ci'][1], '{:.1f}')}]")
        cells.append(f"{sgn(c['d_frac50_pp'], '{:.1f}')} "
                     f"[{sgn(c['frac_ci'][0], '{:.1f}')}, "
                     f"{sgn(c['frac_ci'][1], '{:.1f}')}]")
        rows.append(cells)
        print("  " + " | ".join(cells))
    return rows


def set_cell(cell, text, size=13, bold=False, color=INK, align=PP_ALIGN.CENTER):
    cell.fill.solid()
    cell.fill.fore_color.rgb = SURF
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = cell.margin_right = Inches(0.04)
    cell.margin_top = cell.margin_bottom = Inches(0.02)
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    f = r.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color


def hline(cell, edge, width_pt, color="0B0B0B"):
    """Booktabs-style horizontal rule on one cell edge ('T' or 'B')."""
    tcPr = cell._tc.get_or_add_tcPr()
    tag = qn(f"a:ln{edge}")
    for old in tcPr.findall(tag):
        tcPr.remove(old)
    ln = tcPr.makeelement(tag, {"w": str(Emu(Pt(width_pt)).emu),
                                "cap": "flat"})
    fill = ln.makeelement(qn("a:solidFill"), {})
    clr = ln.makeelement(qn("a:srgbClr"), {"val": color})
    fill.append(clr)
    ln.append(fill)
    # a:lnB must precede a:lnTlToBr etc.; append order is fine for T/B
    tcPr.insert(0, ln) if edge == "T" else tcPr.append(ln)


def main():
    rows = build_rows()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = SURF

    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.42),
                                  Inches(12.2), Inches(0.6))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = ("Localization performance with and without physics-derived "
              "feature maps")
    r.font.name, r.font.size, r.font.bold = FONT, Pt(26), True
    r.font.color.rgb = INK

    cap = slide.shapes.add_textbox(Inches(0.55), Inches(1.08),
                                   Inches(12.2), Inches(0.9))
    ctf = cap.text_frame
    ctf.word_wrap = True
    p = ctf.paragraphs[0]
    r = p.add_run()
    r.text = ("Super-emitter benchmark: 2,000 held-out scenarios, 4–8 "
              "masts, q ~ LogU(100, 500) kg/h.  Metric ranges are the "
              "minimum and maximum across three training seeds.  Paired "
              "effects compare predictions on the same 2,000 scenarios for "
              "the prespecified seed-1 run; 95% confidence intervals from "
              "10,000 paired-bootstrap resamples.  Negative ΔRadius "
              "and positive Δ(<50 m) indicate improvement.")
    r.font.name, r.font.size = FONT, Pt(12.5)
    r.font.color.rgb = INK2

    n_rows, n_cols = 2 + len(rows), 9
    gt = slide.shapes.add_table(n_rows, n_cols, Inches(0.55), Inches(2.25),
                                Inches(12.2), Inches(2.6))
    table = gt.table
    # kill the default banded style
    table.first_row = False
    table.horz_banding = False
    widths = [2.15, 0.95, 1.15, 0.95, 0.95, 1.15, 0.95, 2.05, 1.90]
    for c, w in zip(table.columns, widths):
        c.width = Inches(w)
    for i, row in enumerate(table.rows):
        row.height = Inches(0.42 if i < 2 else 0.5)

    # row 0: grouped headers
    set_cell(table.cell(0, 0), "")
    table.cell(0, 1).merge(table.cell(0, 3))
    set_cell(table.cell(0, 1), "Without feature maps", 13.5, True)
    table.cell(0, 4).merge(table.cell(0, 6))
    set_cell(table.cell(0, 4), "With feature maps", 13.5, True)
    table.cell(0, 7).merge(table.cell(0, 8))
    set_cell(table.cell(0, 7), "Paired effect (95% CI)", 13.5, True)

    heads = ["Base network", "Cov.", "Radius (m)", "< 50 m",
             "Cov.", "Radius (m)", "< 50 m",
             "ΔRadius (m)", "Δ(< 50 m) (pp)"]
    for j, h in enumerate(heads):
        set_cell(table.cell(1, j), h, 13, False, INK,
                 PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)

    for i, cells in enumerate(rows):
        for j, v in enumerate(cells):
            set_cell(table.cell(2 + i, j), v, 13, bold=(j == 5),
                     align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)

    for j in range(n_cols):
        hline(table.cell(0, j), "T", 1.6)          # top rule
        hline(table.cell(1, j), "B", 1.0)          # under column headers
        hline(table.cell(n_rows - 1, j), "B", 1.6)  # bottom rule
    for j in range(1, 7):                           # cmidrules under groups
        hline(table.cell(0, j), "B", 0.75)
    hline(table.cell(0, 7), "B", 0.75)
    hline(table.cell(0, 8), "B", 0.75)

    note = slide.shapes.add_textbox(Inches(0.55), Inches(5.05),
                                    Inches(12.2), Inches(0.5))
    p = note.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = ("Computed from the frozen measured-wind conformal audits "
              "(csr-data-se48) and results/se48_table_deltas_ci.json — "
              "regenerate with generate_results_table_pptx.py.")
    r.font.name, r.font.size = FONT, Pt(10.5)
    r.font.color.rgb = MUTED

    out = pg.OUT / "table1_results_se48.pptx"
    prs.save(out)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()

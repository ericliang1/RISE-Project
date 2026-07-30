"""Shared poster style for the XAI figures — the single source of truth.

Both xai_fig1.py and xai_fig23.py import everything visual from here so the
figures cannot drift apart: one rcParams block, one canonical feature-map
order and naming, one colormap per quantity, and one set of glyphs
(white triangle = mast, gold star = true source, red X = argmax).
"""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURF = "#fcfcfb"
GOLD, REDX = "#f2c230", "#e04343"
BLUE, ORANGE = "#2a78d6", "#eb6834"

# Canonical feature-map presentation: display order -> ensr channel index.
# 1) Source evidence  2) Sensor visibility  3) Wind-error sensitivity
# 4) Fit quality  — used verbatim by every figure that shows the maps.
MAP_ORDER = [("Source evidence", 0), ("Sensor visibility", 2),
             ("Wind-error sensitivity", 1), ("Fit quality", 3)]

# One colormap per quantity class.
CMAP_MAG = "magma"        # magnitude: probability, evidence, feature maps
CMAP_MAG_R = "magma_r"    # reversed magnitude (bright = good / small)
CMAP_DIV = "coolwarm"     # diverging attribution, centred at 0 (white)
CMAP_COUNT = "plasma"     # ordinal counts (masts)


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 16, "axes.titlesize": 18, "figure.titlesize": 22,
        "axes.titleweight": "bold",
        "lines.linewidth": 2.4,
        "figure.constrained_layout.use": True,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
    })


def draw_masts(ax, sx, hero=False):
    """White triangles with ink edges; one size for small panels, one for
    the hero panel."""
    ax.scatter(sx[:, 0], sx[:, 1], marker="^", s=170 if hero else 64,
               c="white", edgecolors=INK, linewidths=1.4 if hero else 1.1,
               zorder=6)


def draw_source(ax, xy, hero=False):
    """Gold star with ink edge."""
    ax.plot(*xy, marker="*", color=GOLD, ms=30 if hero else 16, mec=INK,
            mew=1.5 if hero else 1.0, lw=0, zorder=7)


def draw_argmax(ax, xy, hero=False):
    """Red X with a white halo so it survives any background."""
    ax.plot(*xy, marker="x", color=REDX, ms=20 if hero else 11,
            mew=4.0 if hero else 2.6, lw=0, zorder=7,
            path_effects=[pe.withStroke(linewidth=6 if hero else 4.2,
                                        foreground="white")])


def frame(ax, lw=1.2):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
        sp.set_linewidth(lw)


def header(fig, title, subtitle, ty=1.05, sy=1.02, x=0.02):
    """Figure title + one-line gray subtitle as unmanaged fig.text (placed
    above the axes area; bbox_inches='tight' includes them).  Used instead
    of fig.suptitle because constrained layout repositions suptitles into
    the subtitle line.

    The subtitle is auto-shrunk if it would overflow the canvas width —
    otherwise bbox_inches='tight' silently widens the saved image past the
    axes block, leaving a dead whitespace band.  The title is never shrunk
    (the brief fixes it at 22 pt); an overflowing title is reported so the
    wording can be shortened instead."""
    t1 = fig.text(x, ty, title, fontsize=22, fontweight="bold", ha="left",
                  va="bottom")
    t2 = fig.text(x, sy, subtitle, fontsize=13.5, color=MUTED, ha="left",
                  va="bottom")
    fig.canvas.draw()
    wpx = fig.get_size_inches()[0] * fig.dpi
    for t, shrink in ((t1, False), (t2, True)):
        ext = t.get_window_extent()
        avail = wpx - ext.x0 - 0.008 * wpx
        if ext.width > avail:
            if shrink:
                t.set_fontsize(max(11.0,
                                   t.get_fontsize() * avail / ext.width))
            else:
                print(f"WARNING: title overflows figure width by "
                      f"{(ext.width - avail) / fig.dpi:.2f} in — shorten "
                      f"the wording")


def legend_handles(n_masts):
    """Legend proxies for the shared glyphs, defined once so they cannot
    drift from draw_masts/draw_source/draw_argmax."""
    return [plt.Line2D([], [], marker="^", lw=0, ms=12, mfc="white",
                       mec=INK, label=f"sensor masts ({n_masts})"),
            plt.Line2D([], [], marker="*", lw=0, ms=17, mfc=GOLD, mec=INK,
                       label="true source"),
            plt.Line2D([], [], marker="x", lw=0, ms=12, mew=3.2,
                       color=REDX, label="argmax prediction")]


def save_all(fig, base):
    """Vector PDF + 300-dpi PNG + SVG, same base filename.  Warns when the
    tight bbox is meaningfully wider than the declared figsize (i.e. some
    text overflows and would leave a dead band in the export)."""
    fig.canvas.draw()
    tb = fig.get_tightbbox()
    fw, fh = fig.get_size_inches()
    if tb.width > fw * 1.03:
        print(f"WARNING: tight bbox {tb.width:.1f} in wide vs figsize "
              f"{fw:.1f} in — text overflows the axes block")
    for ext in ("pdf", "png", "svg"):
        p = f"{base}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")

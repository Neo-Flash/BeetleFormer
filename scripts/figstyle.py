"""Shared publication figure style for the dsRNATransformer manuscript.

Unified 6-color palette (user-specified), seaborn-based, 600 dpi, editable-text
SVG + PDF + PNG export. Import this from every plotting script so all figures
share one visual language.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---- unified 6-color palette (RGB -> hex) ---------------------------------- #
NAVY   = "#0F2B5F"   # rgb(15,43,95)    primary / hero
BLUE   = "#5991C7"   # rgb(89,145,199)  secondary
SKY    = "#8EC1E7"   # rgb(142,193,231) light
PALE   = "#B9DBF1"   # rgb(185,219,241) pale fill
TAN    = "#D5A370"   # rgb(213,163,112) accent / contrast
BROWN  = "#7B4F37"   # rgb(123,79,55)   dark accent
PALETTE = [NAVY, BLUE, SKY, PALE, TAN, BROWN]

# semantic roles — deep navy and deep brown are the two PRIMARY colors;
# the lighter blues / tan are only used as supplementary shades when a third+
# category is needed.
HERO      = NAVY     # our main model (BeetleFormer)
BASELINE  = BROWN    # baseline / comparison
LOWER     = BLUE     # third category (lower bound / variant)
POS       = NAVY     # positive class
NEG       = BROWN    # negative class
THIRD     = BLUE     # supplementary third series

# custom colormaps built from the unified palette (no off-palette teal/pink)
from matplotlib.colors import LinearSegmentedColormap
DIVERGE = LinearSegmentedColormap.from_list("beetle_div", [BROWN, TAN, "#ffffff", SKY, NAVY])
SEQ_CMAP = LinearSegmentedColormap.from_list("beetle_seq", ["#ffffff", PALE, SKY, BLUE, NAVY])


def set_style(context="paper", font_scale=1.0):
    """Apply the unified seaborn/matplotlib publication style. Call once."""
    sns.set_theme(context=context, style="ticks", font_scale=font_scale,
                  rc={
                      "figure.dpi": 150,
                      "savefig.dpi": 600,
                      "font.family": "sans-serif",
                      "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                      "svg.fonttype": "none",     # editable SVG text
                      "pdf.fonttype": 42,         # editable PDF text
                      "axes.linewidth": 0.8,
                      "axes.grid": False,
                      "legend.frameon": False,
                      "axes.titleweight": "bold",
                      "axes.titlesize": 10,
                      "figure.constrained_layout.use": True,  # no overlap
                  })


def panel_label(ax, label, x=-0.10, y=1.05, size=12):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=size,
            fontweight="bold", ha="left", va="bottom")


def save(fig, path_noext, dpi=600):
    """Save SVG (editable) + PDF + high-dpi PNG."""
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{path_noext}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

"""Shared publication figure style for the paper's figures.

The paper body is typeset in NimbusRomNo9L (a Times clone) by neurips_2026.sty. A
matplotlib figure left on its defaults embeds DejaVu Sans, and the mismatch is visible
at a glance -- it is the usual tell of a figure that does not belong to its paper. This
module matches the body text instead:

  - Times/STIX serif for text, STIX for math (metric-compatible with Times)
  - Type 42 fonts, since NeurIPS-style venues reject Type 3
  - figure sizes in inches that match \\linewidth at NeurIPS margins, so no rescaling
    happens at \\includegraphics time and the point sizes below are the point sizes
    the reader sees

Import and call `apply()` before creating any figure.

    from figstyle import apply, LINEWIDTH, INK, BLUE, WARM, GREY
    apply()
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# NeurIPS 2026 text block is 5.5in wide. Draw at exactly that so
# \includegraphics[width=\linewidth] applies no scale factor.
LINEWIDTH = 5.5

# Palette: one accent for the condition under test, one for its control, grey for
# the reference. Distinguishable in greyscale by lightness as well as hue.
INK = "#1a1a1a"
BLUE = "#1f4e79"   # the condition being argued for
WARM = "#b8622f"   # the control that matches it
GREY = "#8e8e8e"   # baseline / reference


def apply():
    """Set rcParams so figures match the paper's typography."""
    matplotlib.rcParams.update({
        # --- typography: match the body text ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # Body text is 10pt. Captions are 9pt. Figure text should sit at or just
        # below caption size so it never competes with the prose.
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        # --- vector output, no Type 3 ---
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        # --- restrained chrome ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#4a4a4a",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#4a4a4a",
        "ytick.color": "#4a4a4a",
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "figure.dpi": 200,
    })

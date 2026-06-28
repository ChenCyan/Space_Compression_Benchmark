"""Top-journal-quality plots — CVPR/TIP/JSTARS style.

Key conventions:
  - Times New Roman, larger fonts, clean axes
  - Professional color palette (Tableau 10 + extensions)
  - Thin lines (1.0–1.5 pt), clear markers, no chartjunk
  - Legend inside plot area (not outside), subtle grid
  - 300 DPI PDF output, proper column-width sizing
  - Consistent styling across all figures
  - Serif math fonts for labels
"""

import csv, math, os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ====================================================================
# Font & style setup — match journal standards
# ====================================================================
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": False,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
})

# ====================================================================
# Data
# ====================================================================
DSETS = [
    ("Berlin-Urban-Gradient", "results/berlin_full_final.csv", "berlin"),
    ("Indian Pines", "results/indian_new_codecs.csv", "indian"),
]
OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

# ====================================================================
# Color palette — Tableau-inspired, designed for colorblind safety
# ====================================================================
# Learned methods get warm/orange tones, classic get cool/blue tones
FAMILIES = [
    # prefix          color        marker   label            z  ls   connect
    # ---- classic (cool spectrum) ----
    ("ccsds123",      "#2166ac",   "o",     "CCSDS-123",     6, "-",  True),
    ("jpegls",        "#b2182b",   "p",     "JPEG-LS$_{NEAR}$", 5, "-",  True),
    ("jpeg2000",      "#4daf4a",   "D",     "JPEG2000$_{MCT}$", 5, "--", True),
    ("klt_dwt_nc28",  "#1b9e77",   "s",     "KLT+DWT$_{28}$",  4, "-",  True),
    ("klt_dwt_nc56",  "#66c2a5",   "s",     "KLT+DWT$_{56}$",  4, "--", True),
    ("klt_dwt_nc111", "#a6d854",   "s",     "KLT+DWT$_{111}$", 4, ":",  True),
    ("lz4",           "#7570b3",   "*",     "LZ4",             3, ":",  False),
    ("zlib",          "#d95f02",   "*",     "zlib",            3, ":",  False),
    # ---- learned (warm spectrum) ----
    ("cae1d",         "#e41a1c",   "P",     "CAE1D$_{1D}$",   3, "-",  True),
    ("cae3d",         "#f781bf",   "X",     "CAE3D$_{3D}$",   3, "-",  True),
    ("sscnet",        "#999999",   "v",     "SSCNet",          3, "-",  True),
    ("hycot",         "#ff7f00",   "<",     "HYCOT",           3, "-",  True),
    ("hycass",        "#377eb8",   "^",     "HyCASS",          3, "-.", True),
]

FS = {
    "lossless":      {"ec": "#2ca02c", "lw": 1.3, "s": 100},
    "near-lossless": {"ec": "#ffbb00", "lw": 0.8, "s": 85},
    "lossy":         {"ec": "#aaaaaa", "lw": 0.5, "s": 70},
}

XTICKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# ====================================================================
# Helpers
# ====================================================================
def load(path):
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            for k in ["cr","bpppc","mse","mae","psnr","pae","sa","throughput_mbs"]:
                try: d[k] = float(d[k])
                except: d[k] = None
            rows.append(d)
    return rows

def family_key(name):
    if name.startswith("klt_dwt"):
        for tok in name.split("_"):
            if tok.startswith("nc"): return "klt_dwt_"+tok
    for prefix, *_ in FAMILIES:
        if name.startswith(prefix): return prefix
    return name

def style_for(fk):
    for prefix, color, marker, label, z, ls, _ in FAMILIES:
        if fk.startswith(prefix):
            return {"color":color,"marker":marker,"label":label,"zorder":z,"ls":ls}
    return {"color":"#333333","marker":"x","label":fk,"zorder":2,"ls":"-"}

def connect_for(fk):
    for prefix, *_ in FAMILIES:
        if fk.startswith(prefix):
            return FAMILIES[[f[0] for f in FAMILIES].index(prefix)][6]
    return False

def f_psnr(v, ceil=100.0):
    if v is None: return None
    if math.isinf(v) or not np.isfinite(v): return ceil
    return float(v)

def group_rows(rows):
    groups = {}
    for r in rows:
        fk = family_key(r["method"]); groups.setdefault(fk,[]).append(r)
    for g in groups.values(): g.sort(key=lambda x: (x.get("cr")or 0))
    return groups

# ====================================================================
# Plot panel builder
# ====================================================================
def plot_panel(ax, groups, y_key, ylabel, log_y=False, title=""):
    seen = set()
    for fk, group in sorted(groups.items()):
        s = style_for(fk)

        pts = []
        for r in group:
            cr, val = r["cr"], r.get(y_key)
            if cr is None or val is None: continue
            if val == 0:
                if y_key == "throughput_mbs": continue
                val = 1e-6
            pts.append((cr, val))
        if not pts: continue
        crs, vals = zip(*pts)

        lbl = s["label"] if s["label"] not in seen else None
        seen.add(s["label"])

        # Line for ALL families (connect all points) — carries legend label
        ax.plot(crs, vals, color=s["color"], linestyle=s["ls"],
                linewidth=1.2, alpha=0.5, zorder=1,
                label=lbl if lbl else None)

        # Markers (no legend — line carries the label)
        for r in group:
            cr, val = r["cr"], r.get(y_key)
            if cr is None or val is None: continue
            if val == 0:
                if y_key == "throughput_mbs": continue
                val = 1e-6
            fs = FS.get(r.get("family",""),{})
            ax.scatter(cr, val, c=s["color"], marker=s["marker"], s=fs.get("s",75),
                       edgecolors=fs.get("ec","#333333"), linewidths=fs.get("lw",0.4),
                       zorder=s["zorder"])

    if log_y: ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS], fontsize=8)
    ax.set_xlabel("Compression Ratio", fontsize=12, fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="medium")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.25, which="major", linestyle="-", linewidth=0.3)
    ax.grid(False, which="minor")
    ax.tick_params(axis="x", which="minor", bottom=False)
    # Subtle x-minor ticks
    ax.tick_params(axis="both", which="major", length=4, width=0.6)
    ax.tick_params(axis="both", which="minor", length=2, width=0.4)

def add_legend(ax, groups_dict, ncol=3, **kw):
    """Build clean legend with one entry per family (line+marker combo proxy)."""
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate by label
    seen = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    leg = ax.legend(seen.values(), seen.keys(), ncol=ncol, loc="best", **kw)
    leg.get_frame().set_linewidth(0.5)
    return leg

# ====================================================================
# Generate figures for each dataset
# ====================================================================
for dset_name, csv_path, prefix in DSETS:
    print(f"\n{'='*50}\n  {dset_name}\n{'='*50}")
    rows = load(csv_path)
    groups = group_rows(rows)
    print(f"  {len(rows)} codecs, {len(groups)} families")

    # Lossy-only view
    groups_lossy = {}
    for fk, g in groups.items():
        fg = [r for r in g if (r.get("mse") or 0) > 0]
        if fg: groups_lossy[fk] = fg

    # ---- Throughput (all) ----
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    plot_panel(ax, groups, "throughput_mbs", "Throughput (MB/s)", log_y=True,
               title=f"Compression Ratio vs Throughput")
    add_legend(ax, groups, ncol=2, fontsize=7)
    fig.tight_layout(pad=0.5)
    fname = f"{OUT}/{prefix}_throughput.pdf"
    fig.savefig(fname); plt.close(fig)
    print(f"  [1/4] {fname}")

    # ---- PSNR ----
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_panel(ax, groups_lossy, "psnr", "PSNR (dB)",
               title=f"Rate-Distortion (CR vs PSNR)")
    add_legend(ax, groups_lossy, ncol=3, fontsize=7)
    fig.tight_layout(pad=0.5)
    fname = f"{OUT}/{prefix}_psnr.pdf"
    fig.savefig(fname); plt.close(fig)
    print(f"  [2/4] {fname}")

    # ---- PAE ----
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_panel(ax, groups_lossy, "pae", "PAE (DN)", log_y=True,
               title=f"Rate-Distortion (CR vs PAE)")
    add_legend(ax, groups_lossy, ncol=3, fontsize=7)
    fig.tight_layout(pad=0.5)
    fname = f"{OUT}/{prefix}_pae.pdf"
    fig.savefig(fname); plt.close(fig)
    print(f"  [3/4] {fname}")

    # ---- MSE ----
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_panel(ax, groups_lossy, "mse", "MSE (DN$^2$)", log_y=True,
               title=f"Rate-Distortion (CR vs MSE)")
    add_legend(ax, groups_lossy, ncol=3, fontsize=7)
    fig.tight_layout(pad=0.5)
    fname = f"{OUT}/{prefix}_mse.pdf"
    fig.savefig(fname); plt.close(fig)
    print(f"  [4/4] {fname}")

print(f"\nAll plots saved to {OUT}/")

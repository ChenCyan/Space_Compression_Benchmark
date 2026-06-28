"""Generate per-dataset plots with same-family points connected by lines.

Berlin (51 codecs, all methods) → 4 figures
Indian Pines (12 codecs, classic only) → 4 figures
"""

import csv, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DSETS = [
    ("Berlin-Urban-Gradient (111 bands)", "results/berlin_full_final.csv", "berlin"),
    ("Indian Pines (200 bands)", "results/indian_new_codecs.csv", "indian"),
]
OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

# ---- Family config ----
# (prefix, color, marker, label, zorder, linestyle, connect_line: bool)
FAMILIES = [
    ("ccsds123",     "#2ca02c", "o", "CCSDS-123",             5, "-",   True),
    ("jpegls",       "#e377c2", "p", "JPEG-LS$_{NEAR}$",      6, "-",   True),
    ("jpeg2000",     "#ff7f0e", "D", "JPEG2000$_{MCT}$",      4, "--",  True),
    ("klt_dwt_nc28", "#1f77b4", "s", "KLT+DWT$_{28}$",        4, "-",   True),
    ("klt_dwt_nc56", "#17becf", "s", "KLT+DWT$_{56}$",        4, "--",  True),
    ("klt_dwt_nc111","#9467bd", "s", "KLT+DWT$_{111}$",       4, ":",   True),
    ("lz4",          "#bcbd22", "*", "LZ4",                    3, ":",   False),
    ("zlib",         "#8c564b", "*", "zlib",                   3, ":",   False),
    ("cae1d",        "#e6550d", "P", "CAE1D$_{1D}$",           4, "-",   False),
    ("cae3d",        "#fdae6b", "X", "CAE3D$_{3D}$",           4, "-",   False),
    ("sscnet",       "#bdbdbd", "v", "SSCNet",                 4, "-",   False),
    ("hycot",        "#31a354", "<", "HYCOT",                   4, "-",   False),
    ("hycass",       "#3182bd", "^", "HyCASS",                 4, "-.",  False),
]

FS = {
    "lossless":      {"ec": "limegreen", "lw": 1.5, "s": 130},
    "near-lossless": {"ec": "gold",      "lw": 1.0, "s": 110},
    "lossy":         {"ec": "gray",      "lw": 0.5, "s": 90},
}

XTICKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def load(path):
    rows = []
    with open(path) as f:
        for d in csv.DictReader(f):
            for k in ["cr", "bpppc", "mse", "mae", "psnr", "pae", "sa", "throughput_mbs"]:
                try: d[k] = float(d[k])
                except: d[k] = None
            rows.append(d)
    return rows


def family_key(name):
    if name.startswith("klt_dwt"):
        for tok in name.split("_"):
            if tok.startswith("nc"):
                return "klt_dwt_" + tok
    for prefix, *_ in FAMILIES:
        if name.startswith(prefix):
            return prefix
    return name


def style_for(fk):
    for prefix, color, marker, label, z, ls, _ in FAMILIES:
        if fk.startswith(prefix):
            return {"color": color, "marker": marker, "label": label, "zorder": z, "ls": ls}
    return {"color": "#7f7f7f", "marker": "x", "label": fk, "zorder": 3, "ls": "-"}


def connect_for(fk):
    """Should points of this family be connected by lines?"""
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
        fk = family_key(r["method"])
        groups.setdefault(fk, []).append(r)
    for g in groups.values():
        g.sort(key=lambda x: (x.get("cr") or 0))
    return groups


def plot_panel(ax, groups, y_key, ylabel, log_y=False, title=""):
    seen = set()
    for fk, group in sorted(groups.items()):
        s = style_for(fk)
        do_line = connect_for(fk)

        # Collect valid points
        pts = []
        for r in group:
            cr, val = r["cr"], r.get(y_key)
            if cr is None or val is None: continue
            if val == 0:
                if y_key == "throughput_mbs": continue
                val = 1e-6  # tiny marker for lossless points
            pts.append((cr, val))
        if not pts: continue
        crs, vals = zip(*pts)

        lbl = s["label"] if s["label"] not in seen else None
        seen.add(s["label"])

        # Connect line (same-family points) — line carries legend for this family
        if do_line:
            ax.plot(crs, vals, color=s["color"], linestyle=s["ls"],
                    linewidth=1.4, alpha=0.45, zorder=1,
                    label=lbl if lbl else None)

        # Scatter markers — first point carries the legend label
        first = True
        for r in group:
            cr, val = r["cr"], r.get(y_key)
            if cr is None or val is None: continue
            if val == 0:
                if y_key == "throughput_mbs": continue
                val = 1e-6
            fs = FS.get(r.get("family", ""), {})
            pt_label = lbl if first and not do_line else None
            first = False
            ax.scatter(cr, val, c=s["color"], marker=s["marker"],
                       s=fs.get("s", 80), edgecolors=fs.get("ec", "k"),
                       linewidth=fs.get("lw", 0.5), zorder=s["zorder"],
                       label=pt_label)

    if log_y:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS], fontsize=7)
    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(axis="x", which="minor", bottom=False)


def make_four_figures(groups, dset_name, prefix):
    """Generate 4 figures for one dataset."""

    # ---- Throughput (all methods including lossless) ----
    fig, ax = plt.subplots(figsize=(11, 7))
    plot_panel(ax, groups, "throughput_mbs", "Throughput (MB/s)", log_y=True,
               title=f"CR vs Throughput — {dset_name}")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7.5, loc="upper left", ncol=2)
    fig.tight_layout()
    fname = f"{OUT}/{prefix}_throughput.pdf"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [1/4] {fname}")

    # ---- Lossy-only groups for PSNR/PAE/MSE ----
    lossy_groups = {}
    for fk, g in groups.items():
        filtered = [r for r in g if (r.get("mse") or 0) > 0]
        if filtered:
            lossy_groups[fk] = filtered

    for y_key, ylabel, title_prefix in [
        ("psnr", "PSNR (dB)", "Rate-Distortion: CR vs PSNR"),
        ("pae", "PAE (DN)", "Rate-Distortion: CR vs PAE"),
        ("mse", "MSE (DN²)", "Rate-Distortion: CR vs MSE"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 7.5))
        plot_panel(ax, lossy_groups, y_key, ylabel, log_y=(y_key != "psnr"),
                   title=f"{title_prefix} — {dset_name}")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ncol = 4
        ax.legend(by_label.values(), by_label.keys(), fontsize=7.5,
                  loc="lower left" if y_key == "psnr" else "upper left", ncol=ncol)
        fig.tight_layout()
        fname = f"{OUT}/{prefix}_{y_key}.pdf"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        idx = {"psnr": 2, "pae": 3, "mse": 4}[y_key]
        print(f"  [{idx}/4] {fname}")


# ========== Main ==========
for dset_name, csv_path, prefix in DSETS:
    print(f"\n{'='*60}")
    print(f"  {dset_name}")
    print("=" * 60)
    rows = load(csv_path)
    groups = group_rows(rows)
    print(f"  {len(rows)} codecs, {len(groups)} families")
    make_four_figures(groups, dset_name, prefix)

print(f"\nAll plots saved to {OUT}/")

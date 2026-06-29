"""Plot benchmark results for berlin-dense dataset with newly trained HyCASS models."""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "legend.framealpha": 0.85, "legend.edgecolor": "#ccc",
    "grid.alpha": 0.25, "grid.linewidth": 0.4,
})

CSV = "/data/cyl/space_compression/hycass/results/berlin_dense_bench.csv"
OUT = "/data/cyl/space_compression/hycass/results/plots_berlin_dense"
os.makedirs(OUT, exist_ok=True)

STYLE = {
    "ccsds123":     dict(c="#2166ac", m="o", label="CCSDS-123",       ls="-"),
    "jpeg2000":     dict(c="#4daf4a", m="D", label="JPEG2000$_{MCT}$",ls="--"),
    "klt_dwt_nc28": dict(c="#984ea3", m="s", label="KLT+DWT$_{28}$",  ls="-"),
    "klt_dwt_nc56": dict(c="#e78ac3", m="s", label="KLT+DWT$_{56}$",  ls="--"),
    "lz4":          dict(c="#7570b3", m="*", label="LZ4",             ls=":"),
    "zlib":         dict(c="#d95f02", m="*", label="zlib",            ls=":"),
    "hycass":       dict(c="#e41a1c", m="^", label="HyCASS (retrained)", ls="-."),
}

def family(name):
    for p in ["ccsds123","jpeg2000","klt_dwt_nc28","klt_dwt_nc56","lz4","zlib","hycass"]:
        if name.startswith(p): return p
    return name

def load():
    rows = []
    with open(CSV, newline="") as f:
        for d in csv.DictReader(f):
            for k in ["cr","psnr","pae","mse","throughput_mbs"]:
                try: d[k] = float(d[k])
                except: d[k] = None
            d["_fam"] = family(d["method"])
            rows.append(d)
    return rows

rows = load()
print(f"Loaded {len(rows)} rows")

# group by family, sort by CR
groups = {}
for r in rows:
    groups.setdefault(r["_fam"], []).append(r)
for g in groups.values():
    g.sort(key=lambda x: x["cr"] or 0)

XTICKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

def plot_metric(metric, ylabel, title, fname, log_y=False, psnr_ceil=100):
    fig, ax = plt.subplots(figsize=(7, 5))
    seen = set()
    for fk, grp in groups.items():
        s = STYLE.get(fk, dict(c="#888", m=".", label=fk, ls="-"))
        pts = []
        for r in grp:
            v = r.get(metric)
            if v is None or r["cr"] is None: continue
            if metric == "psnr":
                v = psnr_ceil if (math.isinf(v) or not np.isfinite(v)) else v
            if v <= 0 and metric != "throughput_mbs": continue
            pts.append((r["cr"], v))
        if not pts: continue
        crs, vals = zip(*pts)
        lbl = s["label"] if s["label"] not in seen else None
        seen.add(s["label"])
        ax.plot(crs, vals, color=s["c"], linestyle=s["ls"], linewidth=1.3,
                alpha=0.55, zorder=1, label=lbl)
        for cr, v in pts:
            ax.scatter(cr, v, c=s["c"], marker=s["m"], s=65,
                       edgecolors="#333", linewidths=0.4, zorder=2)
        # annotate JPEG2000 CR values
        if fk == "jpeg2000":
            for r in grp:
                v = r.get(metric)
                if v is None or r["cr"] is None: continue
                if metric == "psnr":
                    v = psnr_ceil if (math.isinf(v) or not np.isfinite(v)) else v
                op = r.get("operating_point", "")
                tag = op.replace("cr=", "") if "cr=" in op else ("L" if "lossless" in op else "")
                if tag:
                    ax.annotate(tag, (r["cr"], v), textcoords="offset points",
                                xytext=(6, -6), fontsize=6.5, color=s["c"], alpha=0.85)

    if log_y: ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS], fontsize=8)
    ax.set_xlim(1.2, 1100)
    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.25, which="major", ls="-", lw=0.3)
    h, l = ax.get_legend_handles_labels()
    seen2 = {}
    for hh, ll in zip(h, l):
        if ll and ll not in seen2: seen2[ll] = hh
    if seen2:
        ax.legend(seen2.values(), seen2.keys(), ncol=2, loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"Saved: {fname}")

plot_metric("psnr",         "PSNR (dB)",        "Berlin-Dense — CR vs PSNR",       "bd_psnr.png")
plot_metric("pae",          "PAE (DN)",          "Berlin-Dense — CR vs PAE",        "bd_pae.png",  log_y=True)
plot_metric("throughput_mbs","Throughput (MB/s)","Berlin-Dense — CR vs Throughput", "bd_throughput.png", log_y=True)
plot_metric("mse",          "MSE",               "Berlin-Dense — CR vs MSE",        "bd_mse.png",  log_y=True)

print(f"\nAll plots saved to {OUT}/")

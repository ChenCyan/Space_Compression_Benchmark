"""Multi-dataset comparison plots — Berlin vs Indian Pines, side by side."""
import csv, os, math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BERLIN_CSV = "results/berlin_new_codecs.csv"
INDIAN_CSV = "results/indian_new_codecs.csv"
OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for d in csv.DictReader(f):
            for k in ["cr","bpppc","mse","mae","psnr","pae","sa","throughput_mbs"]:
                try: d[k] = float(d[k])
                except: d[k] = None
            rows.append(d)
    return rows

berlin_rows = load(BERLIN_CSV)
indian_rows = load(INDIAN_CSV)
datasets = [("Berlin-Urban-Gradient (111 bands)", berlin_rows),
            ("Indian Pines (200 bands)", indian_rows)]

# Only classic methods: CCSDS-123, JPEG2000, KLT+DWT
def is_classic(name):
    return any(name.startswith(p) for p in ["ccsds123","jpeg2000","klt_dwt"])

for r in berlin_rows + indian_rows:
    r["_classic"] = is_classic(r["method"])

FAMILIES = {
    "ccsds123":   {"color":"#2ca02c","marker":"o","label":"CCSDS-123","ls":"-","lw":1.5,"z":5},
    "jpegls":     {"color":"#e377c2","marker":"p","label":"JPEG-LS","ls":"-","lw":1.8,"z":6},
    "jpeg2000":   {"color":"#ff7f0e","marker":"D","label":"JPEG2000","ls":"--","lw":1.5,"z":4},
    "klt_dwt_nc28": {"color":"#1f77b4","marker":"s","label":"KLT+DWT nc=28","ls":"-","lw":1.5,"z":4},
    "klt_dwt_nc56": {"color":"#17becf","marker":"s","label":"KLT+DWT nc=56","ls":"--","lw":1.2,"z":4},
    "lz4":        {"color":"#bcbd22","marker":"*","label":"LZ4","ls":":","lw":1.0,"z":3},
    "zlib":       {"color":"#8c564b","marker":"*","label":"zlib","ls":":","lw":1.0,"z":3},
}

FS = {"lossless":{"ec":"limegreen","lw":1.5,"s":130},
      "near-lossless":{"ec":"gold","lw":1.0,"s":110},
      "lossy":{"ec":"gray","lw":0.5,"s":90}}

def family_key(name):
    if "klt_dwt" in name:
        for tok in name.split("_"):
            if tok.startswith("nc"): return "klt_dwt_" + tok
    for p in FAMILIES:
        if name.startswith(p): return p
    return name

def f_psnr(v, ceil=100.0):
    if v is None: return None
    if math.isinf(v) or not np.isfinite(v): return ceil
    return float(v)

XTICKS = [1,2,4,8,16,32,64]

def setup_ax(ax, title):
    ax.set_xlabel("Compression Ratio", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS], fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_title(title, fontsize=13, fontweight="bold")

def plot_one(ax, rows, y_key, y_label, is_log=False):
    """Plot one panel: all classic methods from a dataset."""
    groups = {}
    for r in rows:
        if not r.get("_classic"): continue
        fk = family_key(r["method"])
        groups.setdefault(fk, []).append(r)
    for g in groups.values():
        g.sort(key=lambda x: (x.get("cr") or 0))

    seen = set()
    for fk, group in sorted(groups.items()):
        s = FAMILIES.get(fk, {"color":"#7f7f7f","marker":"x","label":fk,"ls":"-","lw":1.0,"z":3})
        pts = []
        for r in group:
            cr, val = r["cr"], r.get(y_key)
            if cr is None or val is None: continue
            val = val if val > 0 or y_key == "throughput_mbs" else 0.03
            pts.append((cr, val))
        if not pts: continue
        crs, vals = zip(*pts)
        lbl = s["label"] if s["label"] not in seen else None; seen.add(s["label"])
        ax.plot(crs, vals, color=s["color"], linestyle=s["ls"], linewidth=s["lw"], alpha=0.5, zorder=1)
        for r in group:
            cr = r["cr"]
            val = r.get(y_key)
            if cr is None or val is None: continue
            vm = 0.03 if (val == 0 and y_key != "throughput_mbs") else val
            fs = FS.get(r.get("family",""),{})
            ax.scatter(cr, vm, c=s["color"], marker=s["marker"], s=fs.get("s",80),
                       edgecolors=fs.get("ec","k"), linewidth=fs.get("lw",0.5), zorder=s["z"])
    if is_log:
        ax.set_yscale("log")

# =====================================================================
# Figure 1: CR vs PSNR (2 panels)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for ax, (title, rows) in zip(axes, datasets):
    plot_one(ax, rows, "psnr", "PSNR (dB)")
    setup_ax(ax, title)
axes[0].set_ylabel("PSNR (dB)", fontsize=12)
axes[1].set_ylabel("")
# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), fontsize=9, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Rate-Distortion: CR vs PSNR", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/cr_vs_psnr_comparison.pdf", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[1/3] cr_vs_psnr_comparison.pdf")

# =====================================================================
# Figure 2: CR vs PAE (2 panels)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for ax, (title, rows) in zip(axes, datasets):
    plot_one(ax, rows, "pae", "PAE (DN)", is_log=True)
    setup_ax(ax, title)
axes[0].set_ylabel("PAE (DN)", fontsize=12)
axes[1].set_ylabel("")
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), fontsize=9, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Rate-Distortion: CR vs PAE", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/cr_vs_pae_comparison.pdf", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[2/3] cr_vs_pae_comparison.pdf")

# =====================================================================
# Figure 3: CR vs Throughput (2 panels)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for ax, (title, rows) in zip(axes, datasets):
    plot_one(ax, rows, "throughput_mbs", "Throughput (MB/s)", is_log=True)
    setup_ax(ax, title)
axes[0].set_ylabel("Throughput (MB/s)", fontsize=12)
axes[1].set_ylabel("")
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), fontsize=9, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Throughput vs Compression Ratio", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/cr_vs_throughput_comparison.pdf", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[3/3] cr_vs_throughput_comparison.pdf")

print(f"\nAll comparison plots saved to {OUT}/")

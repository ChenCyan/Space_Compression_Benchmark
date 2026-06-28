"""Final 4-plot suite — all Berlin data with full legend."""
import csv, os, math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/berlin_full_final.csv"
OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

rows = []
with open(CSV) as f:
    for d in csv.DictReader(f):
        for k in ["cr","bpppc","mse","mae","psnr","pae","sa","throughput_mbs"]:
            try: d[k] = float(d[k])
            except: d[k] = None
        rows.append(d)

# ---- Families ----
# Classic
FAMILIES = [
    ("ccsds123",     "#2ca02c","o","CCSDS-123",       5,"-"),
    ("jpegls",       "#e377c2","p","JPEG-LS",         6,"-"),
    ("jpeg2000",     "#ff7f0e","D","JPEG2000",        4,"--"),
    ("klt_dwt_nc28", "#1f77b4","s","KLT+DWT nc=28",  4,"-"),
    ("klt_dwt_nc56", "#17becf","s","KLT+DWT nc=56",  4,"--"),
    ("klt_dwt_nc111","#9467bd","s","KLT+DWT nc=111", 4,":"),
    ("lz4",          "#bcbd22","*","LZ4",            3,":"),
    ("zlib",         "#8c564b","*","zlib",           3,":"),
    # Learned
    ("cae1d",  "#e6550d","P","CAE1D",  4,"-"),
    ("cae3d",  "#fdae6b","X","CAE3D",  4,"-"),
    ("sscnet", "#bdbdbd","v","SSCNet", 4,"-"),
    ("hycot",  "#31a354","<","HYCOT",  4,"-"),
    ("hycass", "#3182bd","^","HyCASS", 4,"-."),
]

FS = {"lossless":{"ec":"limegreen","lw":1.5,"s":130},
      "near-lossless":{"ec":"gold","lw":1.0,"s":110},
      "lossy":{"ec":"gray","lw":0.5,"s":90}}

def family_key(name):
    if name.startswith("klt_dwt"):
        for tok in name.split("_"):
            if tok.startswith("nc"): return "klt_dwt_"+tok
    for prefix, *_ in FAMILIES:
        if name.startswith(prefix): return prefix
    return name

def style_for(fk):
    for prefix, color, marker, label, z, ls in FAMILIES:
        if fk.startswith(prefix):
            return {"color":color,"marker":marker,"label":label,"zorder":z,"ls":ls}
    return {"color":"#7f7f7f","marker":"x","label":fk,"zorder":3,"ls":"-"}

def f_psnr(v, ceil=100.0):
    if v is None: return None
    if math.isinf(v) or not np.isfinite(v): return ceil
    return float(v)

# Group rows
groups = {}
for r in rows:
    fk = family_key(r["method"])
    groups.setdefault(fk, []).append(r)
for g in groups.values():
    g.sort(key=lambda x: (x.get("cr") or 0))

XTICKS = [1,2,4,8,16,32,64,128,256,512,1024]

# ---- Lossless: only include in throughput plot ----
def lossy_filter(groups):
    """Only codecs with real error (MSE > 0)."""
    result = {}
    for fk, g in groups.items():
        filtered = [r for r in g if (r.get("mse") or 0) > 0]
        if filtered:
            result[fk] = filtered
    return result

groups_lossy = lossy_filter(groups)

def plot_panel(ax, groups_dict, y_key, ylabel, log_y=False):
    seen = set()
    for fk, group in sorted(groups_dict.items()):
        s = style_for(fk)
        pts = []
        for r in group:
            cr, val = r["cr"], r.get(y_key)
            if cr is None or val is None: continue
            val = val if val > 0 or y_key == "throughput_mbs" else 1e-6
            pts.append((cr, val))
        if not pts: continue
        crs, vals = zip(*pts)
        # Lines: only for classic methods; learned = scatter only
        use_line = not any(fk.startswith(p) for p in ["cae","ssc","hyc","klt_dwt_nc"])
        # Actually, KLT lines are useful. Only learned = scatter.
        is_learned = any(fk.startswith(p) for p in ["cae","ssc","hyco","hyca"])
        lbl = s["label"] if s["label"] not in seen else None
        seen.add(s["label"])
        if not is_learned:
            ax.plot(crs, vals, color=s["color"], linestyle=s["ls"], linewidth=1.2, alpha=0.4, zorder=1)
        for r in group:
            cr, val = r["cr"], r.get(y_key)
            if cr is None or val is None: continue
            vm = 1e-6 if (val == 0 and y_key != "throughput_mbs") else val 
            fs = FS.get(r.get("family",""),{})
            ax.scatter(cr, vm, c=s["color"], marker=s["marker"], s=fs.get("s",80),
                       edgecolors=fs.get("ec","k"), linewidth=fs.get("lw",0.5), zorder=s["zorder"])
    if log_y: ax.set_yscale("log")
    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS], fontsize=7)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(axis="x", which="minor", bottom=False)

# Build legend from first plot
fig, ax = plt.subplots(figsize=(16, 9))
seen = set()
for fk, group in sorted(groups_lossy.items()):
    s = style_for(fk)
    if s["label"] not in seen:
        seen.add(s["label"])
        ax.scatter([],[],c=s["color"],marker=s["marker"],s=70,edgecolors="k",linewidth=0.5,label=s["label"])
ax.legend(fontsize=8, ncol=7, loc="center")
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
leg_path = f"{OUT}/legend.pdf"
fig.savefig(leg_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("[0] legend.pdf")

# ---- Figure 1: Throughput (all methods) ----
fig, ax = plt.subplots(figsize=(10,7))
plot_panel(ax, groups, "throughput_mbs", "Throughput (MB/s)", log_y=True)
ax.set_title("CR vs Throughput (Berlin-Urban-Gradient, 111 bands)", fontsize=14, fontweight="bold")
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/final_cr_vs_throughput.pdf", dpi=150)
plt.close(fig)
print("[1/4] cr_vs_throughput.pdf")

# ---- Figure 2: CR vs PSNR (lossy only) ----
fig, ax = plt.subplots(figsize=(12, 7.5))
plot_panel(ax, groups_lossy, "psnr", "PSNR (dB)")
ax.set_title("Rate-Distortion: CR vs PSNR", fontsize=15, fontweight="bold")
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=7.5, loc="lower left", ncol=4)
fig.tight_layout()
fig.savefig(f"{OUT}/final_cr_vs_psnr.pdf", dpi=150)
plt.close(fig)
print("[2/4] cr_vs_psnr.pdf")

# ---- Figure 3: CR vs PAE (lossy only) ----
fig, ax = plt.subplots(figsize=(12, 7.5))
plot_panel(ax, groups_lossy, "pae", "PAE (DN)", log_y=True)
ax.set_title("Rate-Distortion: CR vs PAE", fontsize=15, fontweight="bold")
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=7.5, loc="upper left", ncol=4)
fig.tight_layout()
fig.savefig(f"{OUT}/final_cr_vs_pae.pdf", dpi=150)
plt.close(fig)
print("[3/4] cr_vs_pae.pdf")

# ---- Figure 4: CR vs MSE (lossy only) ----
fig, ax = plt.subplots(figsize=(12, 7.5))
plot_panel(ax, groups_lossy, "mse", "MSE (DN²)", log_y=True)
ax.set_title("Rate-Distortion: CR vs MSE", fontsize=15, fontweight="bold")
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=7.5, loc="upper left", ncol=4)
fig.tight_layout()
fig.savefig(f"{OUT}/final_cr_vs_mse.pdf", dpi=150)
plt.close(fig)
print("[4/4] cr_vs_mse.pdf")

print(f"\nAll plots saved to {OUT}/")

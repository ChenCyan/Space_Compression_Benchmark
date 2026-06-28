"""Generate 4 benchmark plots — KLT+DWT split into sub-lines by n_components."""
import csv, os, math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/benchmark_all.csv"
OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

rows = []
with open(CSV) as f:
    for d in csv.DictReader(f):
        for k in ["cr","bpppc","mse","mae","psnr","pae","sa","throughput_mbs"]:
            try: d[k] = float(d[k])
            except: d[k] = None
        rows.append(d)

# Split KLT+DWT into sub-families to keep each RD curve monotonic.
# Same method but different n_components = different spectral dimension, so
# connecting them by CR alone produces non-monotonic zigzags.
FAMILY_GROUPS = [
    # (prefix, color, marker, label, zorder, linestyle, lw)
    ("ccsds123",       "#2ca02c", "o", "CCSDS-123",        5, "-",  1.5),
    ("jpeg2000",       "#ff7f0e", "D", "JPEG2000",         4, "--", 1.5),
    ("klt_dwt_nc28",   "#1f77b4", "s", "KLT+DWT nc=28",   4, "-",  1.5),
    ("klt_dwt_nc56",   "#17becf", "s", "KLT+DWT nc=56",   4, "--", 1.2),
    ("klt_dwt_nc111",  "#9467bd", "s", "KLT+DWT nc=111",  4, ":",  1.2),
    ("cae1d",          "#8c564b", "P", "CAE1D",            4, "-",  1.2),
    ("cae3d",          "#e377c2", "X", "CAE3D",            4, "-",  1.2),
    ("sscnet",         "#7f7f7f", "v", "SSCNet",           4, "-",  1.2),
    ("hycot",          "#bcbd22", "h", "HYCOT",            4, "-",  1.2),
    ("hycass",         "#d62728", "^", "HyCASS",           4, "-.", 1.5),
]

FS = {
    "lossless":      {"ec":"limegreen","lw":1.5,"s":130},
    "near-lossless": {"ec":"gold","lw":1.0,"s":110},
    "lossy":         {"ec":"gray","lw":0.5,"s":90},
}

def family_key(name):
    """Map method name to a sub-family key."""
    if "klt_dwt" in name:
        for token in name.split("_"):
            if token.startswith("nc"):
                return "klt_dwt_" + token  # e.g. klt_dwt_nc28
    for prefix, *_ in FAMILY_GROUPS:
        if name.startswith(prefix):
            return prefix
    return name

def style_for(key):
    for prefix, color, marker, label, z, ls, lw in FAMILY_GROUPS:
        if key.startswith(prefix):
            return {"color":color, "marker":marker, "label":label, "zorder":z, "ls":ls, "lw":lw}
    return {"color":"#7f7f7f","marker":"x","label":key,"zorder":3,"ls":"-","lw":1.0}

# Group rows
groups = {}
for row in rows:
    fk = family_key(row["method"])
    groups.setdefault(fk, []).append(row)
for g in groups.values():
    g.sort(key=lambda r: (r.get("cr") or 0))

def f_psnr(v, ceil=100.0):
    if v is None: return None
    if math.isinf(v) or not np.isfinite(v): return ceil
    return float(v)

XTICKS = [1,2,4,8,16,32,64,128,256,512,1024,2048]

def setup_axes(ax, ylabel, title):
    ax.set_xlabel("Compression Ratio", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS], fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(axis="x", which="minor", bottom=False)

# ====================================================================
# Helper: split data into lossless (throughput-only) and lossy (all plots)
def lossy_groups():
    """Yield (fk, group) only for codecs that actually produce errors (MSE>0)."""
    for fk, g in sorted(groups.items()):
        lossy = [r for r in g if r.get("family") != "lossless" and (r.get("mse") or 0) > 0]
        if lossy:
            lossy.sort(key=lambda r: (r.get("cr") or 0))
            yield fk, lossy

# ====================================================================
# Figure 1: CR vs Throughput (ALL methods, including lossless)
# ====================================================================
fig, ax = plt.subplots(figsize=(9,6))
seen = set()
for fk, group in sorted(groups.items()):
    s = style_for(fk)
    pts = [(r["cr"], r["throughput_mbs"]) for r in group if r["cr"] and (r["throughput_mbs"] or 0) > 0]
    if not pts: continue
    crs, tps = zip(*pts)
    lbl = s["label"] if s["label"] not in seen else None; seen.add(s["label"])
    ax.plot(crs, tps, color=s["color"], linestyle=s["ls"], linewidth=s["lw"], alpha=0.5, zorder=1)
    for r in group:
        cr, tp = r["cr"], r["throughput_mbs"]
        if cr is None or tp is None or tp <= 0: continue
        fs = FS.get(r.get("family",""),{})
        ax.scatter(cr, tp, c=s["color"], marker=s["marker"], s=fs.get("s",90),
                   edgecolors=fs.get("ec","k"), linewidth=fs.get("lw",0.5), zorder=s["zorder"])
    if lbl: ax.scatter([],[], c=s["color"], marker=s["marker"], s=70, edgecolors="k", linewidth=0.5, label=lbl)
setup_axes(ax, "Throughput (MB/s)", "Compression Ratio vs Throughput")
ax.set_yscale("log")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/cr_vs_throughput.pdf", dpi=150)
plt.close(fig); print("[1/4] cr_vs_throughput.pdf")

# ====================================================================
# Figure 2: CR vs PSNR (lossy/near-lossless only)
# ====================================================================
fig, ax = plt.subplots(figsize=(9,6))
seen = set()
for fk, group in lossy_groups():
    s = style_for(fk)
    pts = [(r["cr"], f_psnr(r["psnr"],100)) for r in group if r["cr"] and r["psnr"] is not None]
    if not pts: continue
    crs, pvals = zip(*pts)
    lbl = s["label"] if s["label"] not in seen else None; seen.add(s["label"])
    ax.plot(crs, pvals, color=s["color"], linestyle=s["ls"], linewidth=s["lw"], alpha=0.5, zorder=1)
    for r in group:
        cr, psnr, tp = r["cr"], f_psnr(r["psnr"],100), r.get("throughput_mbs",0)
        if cr is None or psnr is None: continue
        fs = FS.get(r.get("family",""),{})
        ax.scatter(cr, psnr, c=s["color"], marker=s["marker"], s=fs.get("s",90),
                   edgecolors=fs.get("ec","k"), linewidth=fs.get("lw",0.5), zorder=s["zorder"])
        if tp > 0: ax.annotate(f"{tp:.1f}", (cr, psnr+1.2), ha="center", fontsize=6, color=s["color"], alpha=0.6)
    if lbl: ax.scatter([],[], c=s["color"], marker=s["marker"], s=70, edgecolors="k", linewidth=0.5, label=lbl)
setup_axes(ax, "PSNR (dB)", "Rate-Distortion: CR vs PSNR")
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/cr_vs_psnr.pdf", dpi=150)
plt.close(fig); print("[2/4] cr_vs_psnr.pdf")

# ====================================================================
# Figure 3: CR vs PAE (lossy/near-lossless only)
# ====================================================================
fig, ax = plt.subplots(figsize=(9,6))
seen = set()
for fk, group in lossy_groups():
    s = style_for(fk)
    pts = [(r["cr"], r["pae"] if r["pae"] and r["pae"]>0 else 0.03)
           for r in group if r["cr"] and r["pae"] is not None]
    if not pts: continue
    crs, pvals = zip(*pts)
    lbl = s["label"] if s["label"] not in seen else None; seen.add(s["label"])
    ax.plot(crs, pvals, color=s["color"], linestyle=s["ls"], linewidth=s["lw"], alpha=0.5, zorder=1)
    for r in group:
        cr, pae = r["cr"], r["pae"]
        if cr is None or pae is None: continue
        pm = 0.03 if pae == 0 else pae
        fs = FS.get(r.get("family",""),{})
        ax.scatter(cr, pm, c=s["color"], marker=s["marker"], s=fs.get("s",90),
                   edgecolors=fs.get("ec","k"), linewidth=fs.get("lw",0.5), zorder=s["zorder"])
    if lbl: ax.scatter([],[], c=s["color"], marker=s["marker"], s=70, edgecolors="k", linewidth=0.5, label=lbl)
setup_axes(ax, "PAE (DN)", "Rate-Distortion: CR vs PAE")
ax.set_yscale("log")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/cr_vs_pae.pdf", dpi=150)
plt.close(fig); print("[3/4] cr_vs_pae.pdf")

# ====================================================================
# Figure 4: CR vs MSE (lossy/near-lossless only)
# ====================================================================
fig, ax = plt.subplots(figsize=(9,6))
seen = set()
for fk, group in lossy_groups():
    s = style_for(fk)
    pts = [(r["cr"], r["mse"] if r["mse"] and r["mse"]>0 else 1e-4)
           for r in group if r["cr"] and r["mse"] is not None]
    if not pts: continue
    crs, pvals = zip(*pts)
    lbl = s["label"] if s["label"] not in seen else None; seen.add(s["label"])
    ax.plot(crs, pvals, color=s["color"], linestyle=s["ls"], linewidth=s["lw"], alpha=0.5, zorder=1)
    for r in group:
        cr, mse = r["cr"], r["mse"]
        if cr is None or mse is None: continue
        mm = 1e-4 if mse == 0 else mse
        fs = FS.get(r.get("family",""),{})
        ax.scatter(cr, mm, c=s["color"], marker=s["marker"], s=fs.get("s",90),
                   edgecolors=fs.get("ec","k"), linewidth=fs.get("lw",0.5), zorder=s["zorder"])
    if lbl: ax.scatter([],[], c=s["color"], marker=s["marker"], s=70, edgecolors="k", linewidth=0.5, label=lbl)
setup_axes(ax, "MSE (DN²)", "Rate-Distortion: CR vs MSE")
ax.set_yscale("log")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/cr_vs_mse.pdf", dpi=150)
plt.close(fig); print("[4/4] cr_vs_mse.pdf")

print(f"\nAll plots saved to {OUT}/")

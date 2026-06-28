"""Generate 4 benchmark comparison plots from CSV data."""
import csv, os, math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CSV = "results/benchmark_full_v2.csv"
OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

# Read data
rows = []
with open(CSV) as f:
    for d in csv.DictReader(f):
        for k in ["cr","bpppc","mse","mae","psnr","pae","sa","throughput_mbs"]:
            try:
                d[k] = float(d[k])
            except:
                d[k] = None
        rows.append(d)

# ---- Style config ----
STYLE = {
    "ccsds123":    {"color":"#2ca02c", "marker":"o", "label":"CCSDS-123 (lossless)", "zorder":5},
    "jpeg2000":    {"color":"#ff7f0e", "marker":"D", "label":"JPEG2000",        "zorder":4},
    "klt_dwt":     {"color":"#1f77b4", "marker":"s", "label":"KLT+DWT",         "zorder":4},
    "hycass":      {"color":"#d62728", "marker":"^", "label":"HyCASS (learned)", "zorder":4},
}
FAMILY_STYLE = {
    "lossless":      {"edgecolors":"limegreen", "linewidth":1.5, "s":120},
    "near-lossless": {"edgecolors":"gold",      "linewidth":1.0, "s":100},
    "lossy":         {"edgecolors":"gray",       "linewidth":0.5, "s":80},
}

def style_for(name):
    for prefix, s in STYLE.items():
        if name.startswith(prefix):
            return s.copy()
    return {"color":"#7f7f7f", "marker":"x", "label":name, "zorder":3}

def finite_psnr(v, ceil=100.0):
    if v is None: return None
    if math.isinf(v) or not np.isfinite(v):
        return ceil
    return float(v)

def sort_key(r):
    # Sort: lossless first, then by CR
    fm = r.get("family","")
    return (0 if fm=="lossless" else 1 if fm=="near-lossless" else 2, r.get("cr",0))

rows.sort(key=sort_key)

# ====================================================================
# Figure 1: CR vs Throughput
# ====================================================================
fig, ax = plt.subplots(figsize=(9,6))
seen = set()
for row in rows:
    s = style_for(row["method"])
    cr = row["cr"]; tp = row["throughput_mbs"]
    if cr is None or tp is None or tp <= 0: continue
    lbl = s["label"] if s["label"] not in seen else None
    seen.add(s["label"])
    fs = FAMILY_STYLE.get(row.get("family",""), {})
    ax.scatter(cr, tp, c=s["color"], marker=s["marker"], s=fs.get("s",90),
               edgecolors=fs.get("edgecolors","k"),
               linewidth=fs.get("linewidth",0.5), zorder=s.get("zorder",3),
               label=lbl)
ax.set_xlabel("Compression Ratio", fontsize=13)
ax.set_ylabel("Throughput (MB/s)", fontsize=13)
ax.set_title("Compression Ratio vs Throughput", fontsize=15, fontweight="bold")
ax.set_xscale("log"); ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=10, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/cr_vs_throughput.pdf", dpi=150)
plt.close(fig); print("[1/4] cr_vs_throughput.pdf")

# ====================================================================
# Figure 2: CR vs PSNR
# ====================================================================
fig, ax = plt.subplots(figsize=(9,6))
seen = set()
for row in rows:
    s = style_for(row["method"])
    cr = row["cr"]; psnr = finite_psnr(row["psnr"], ceil=100)
    if cr is None or psnr is None: continue
    lbl = s["label"] if s["label"] not in seen else None
    seen.add(s["label"])
    fs = FAMILY_STYLE.get(row.get("family",""), {})
    tp = row.get("throughput_mbs",0)
    ax.scatter(cr, psnr, c=s["color"], marker=s["marker"], s=fs.get("s",90),
               edgecolors=fs.get("edgecolors","k"),
               linewidth=fs.get("linewidth",0.5), zorder=s.get("zorder",3),
               label=lbl)
    # annotate throughput
    if tp > 0:
        ax.annotate(f"{tp:.1f}", (cr, psnr+0.8), ha="center", fontsize=6.5,
                    color="gray", alpha=0.8)
ax.set_xlabel("Compression Ratio", fontsize=13)
ax.set_ylabel("PSNR (dB)", fontsize=13)
ax.set_title("Rate-Distortion: CR vs PSNR", fontsize=15, fontweight="bold")
ax.set_xscale("log")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=10, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/cr_vs_psnr.pdf", dpi=150)
plt.close(fig); print("[2/4] cr_vs_psnr.pdf")

# ====================================================================
# Figure 3: CR vs PAE
# ====================================================================
fig, ax = plt.subplots(figsize=(9,6))
seen = set()
for row in rows:
    s = style_for(row["method"])
    cr = row["cr"]; pae = row["pae"]
    if cr is None or pae is None: continue
    if pae == 0: pae = 0.05  # marker for lossless
    lbl = s["label"] if s["label"] not in seen else None
    seen.add(s["label"])
    fs = FAMILY_STYLE.get(row.get("family",""), {})
    ax.scatter(cr, pae, c=s["color"], marker=s["marker"], s=fs.get("s",90),
               edgecolors=fs.get("edgecolors","k"),
               linewidth=fs.get("linewidth",0.5), zorder=s.get("zorder",3),
               label=lbl)
ax.set_xlabel("Compression Ratio", fontsize=13)
ax.set_ylabel("PAE (DN)", fontsize=13)
ax.set_title("Rate-Distortion: CR vs PAE", fontsize=15, fontweight="bold")
ax.set_xscale("log"); ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=10, loc="upper right")
# Mark lossless zone
ax.axhline(y=0.1, color="limegreen", linestyle="--", alpha=0.4, linewidth=1)
ax.annotate("Lossless (PAE=0)", (1.5, 0.15), fontsize=8, color="limegreen", alpha=0.8)
fig.tight_layout(); fig.savefig(f"{OUT}/cr_vs_pae.pdf", dpi=150)
plt.close(fig); print("[3/4] cr_vs_pae.pdf")

# ====================================================================
# Figure 4: CR vs MSE
# ====================================================================
fig, ax = plt.subplots(figsize=(9,6))
seen = set()
for row in rows:
    s = style_for(row["method"])
    cr = row["cr"]; mse = row["mse"]
    if cr is None or mse is None: continue
    if mse == 0: mse = 1e-6  # tiny marker for lossless
    lbl = s["label"] if s["label"] not in seen else None
    seen.add(s["label"])
    fs = FAMILY_STYLE.get(row.get("family",""), {})
    ax.scatter(cr, mse, c=s["color"], marker=s["marker"], s=fs.get("s",90),
               edgecolors=fs.get("edgecolors","k"),
               linewidth=fs.get("linewidth",0.5), zorder=s.get("zorder",3),
               label=lbl)
ax.set_xlabel("Compression Ratio", fontsize=13)
ax.set_ylabel("MSE (DN²)", fontsize=13)
ax.set_title("Rate-Distortion: CR vs MSE", fontsize=15, fontweight="bold")
ax.set_xscale("log"); ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=10, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/cr_vs_mse.pdf", dpi=150)
plt.close(fig); print("[4/4] cr_vs_mse.pdf")

print(f"\nAll plots saved to {OUT}/")

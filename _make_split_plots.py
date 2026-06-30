"""Split-plot suite — lossless / near-lossless / lossy categories."""
import csv, math, os, numpy as np, re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman","DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "legend.framealpha": 0.85, "legend.edgecolor": "#ccc",
    "grid.alpha": 0.25, "grid.linewidth": 0.4,
})

DATA_DIR = "/data/cyl/space_compression/hycass"
CSV_LOSSY    = f"{DATA_DIR}/results/berlin_dense.csv"
CSV_DENSE_BENCH = f"{DATA_DIR}/results/berlin_dense_bench.csv"
CSV_NEAR     = f"{DATA_DIR}/results/berlin_near_dense2.csv"
CSV_LL       = f"{DATA_DIR}/results/berlin_full_final.csv"
CSV_JP2_HIGH = f"{DATA_DIR}/results/jpeg2000_high_cr.csv"
OUT = f"{DATA_DIR}/results/plots"
os.makedirs(OUT, exist_ok=True)

STYLE = {
    "ccsds123":     dict(c="#2166ac",m="o",label="CCSDS-123",        ls="-"),
    "jpegls":       dict(c="#b2182b",m="p",label="JPEG-LS$_{NEAR}$", ls="-"),
    "jpeg2000":     dict(c="#4daf4a",m="D",label="JPEG2000$_{MCT}$", ls="--"),
    "klt_dwt_nc28": dict(c="#984ea3",m="s",label="KLT+DWT$_{28}$",   ls="-"),
    "klt_dwt_nc56": dict(c="#e78ac3",m="s",label="KLT+DWT$_{56}$",   ls="--"),
    "klt_dwt_nc111":dict(c="#a6d854",m="s",label="KLT+DWT$_{111}$",  ls=":"),
    "lz4":          dict(c="#7570b3",m="*",label="LZ4",              ls=":"),
    "zlib":         dict(c="#d95f02",m="*",label="zlib",             ls=":"),
    "cae1d":        dict(c="#e41a1c",m="P",label="CAE1D$_{1D}$",     ls="-"),
    "cae3d":        dict(c="#f781bf",m="X",label="CAE3D$_{3D}$",     ls="-"),
    "sscnet":       dict(c="#999999",m="v",label="SSCNet",           ls="-"),
    "hycot":        dict(c="#ff7f00",m="<",label="HYCOT",            ls="-"),
    "hycass":       dict(c="#377eb8",m="^",label="HyCASS",           ls="-."),
}

XTICKS = [1,2,4,8,16,32,64,128,256,512,1024]

def load_csv(path, rows=None):
    if rows is None: rows = []
    existing = {r["method"] for r in rows}
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            if d["method"] in existing: continue
            for k in ["cr","bpppc","mse","mae","psnr","pae","sa","throughput_mbs"]:
                try: d[k] = float(d[k])
                except: d[k] = None
            rows.append(d); existing.add(d["method"])
    return rows

def load_data():
    rows = []
    rows = load_csv(CSV_NEAR, rows)
    rows = load_csv(CSV_LOSSY, rows)
    if os.path.exists(CSV_DENSE_BENCH):
        rows = load_csv(CSV_DENSE_BENCH, rows)
    rows = load_csv(CSV_LL, rows)
    if os.path.exists(CSV_JP2_HIGH):
        rows = load_csv(CSV_JP2_HIGH, rows)
    for r in rows: r["_fam"] = family(r["method"])
    return rows

def family(name):
    if name.startswith("klt_dwt"):
        for t in name.split("_"):
            if t.startswith("nc"): return "klt_dwt_"+t
    for p in ["ccsds123","jpegls","jpeg2000","cae1d","cae3d","sscnet","hycot","hycass","lz4","zlib"]:
        if name.startswith(p): return p
    return name

def f_psnr(v, ceil=100.0):
    if v is None: return None
    if math.isinf(v) or not np.isfinite(v): return ceil
    return float(v)

def group_rows(rows, families, restrict_to_rows=None):
    allowed = set(r["method"] for r in restrict_to_rows) if restrict_to_rows else None
    groups = {}
    for r in rows:
        if allowed is not None and r["method"] not in allowed: continue
        fk = r["_fam"]
        if fk in families:
            groups.setdefault(fk,[]).append(r)
    for g in groups.values(): g.sort(key=lambda x: (x.get("cr")or 0))
    return {k: groups[k] for k in [fam for fam in families if fam in groups]}

def plot_panel(ax, groups, y_key, ylabel, log_y=False, title="", map_fn=None, label_override=None):
    seen = set()
    for fk, group in groups.items():
        s = STYLE[fk].copy()
        if label_override and fk in label_override: s["label"] = label_override[fk]
        pts = [(r["cr"], (map_fn(r[y_key]) if map_fn else r[y_key]))
               for r in group if r["cr"] is not None and r.get(y_key) is not None
               and (isinstance(r[y_key],(int,float)) and r[y_key]>=0)]
        if not pts: continue
        crs, vals = zip(*pts)
        lbl = s["label"] if s["label"] not in seen else None; seen.add(s["label"])
        ax.plot(crs, vals, color=s["c"], linestyle=s["ls"], linewidth=1.2, alpha=0.5, zorder=1, label=lbl)
        for r in group:
            cr, val = r["cr"], r.get(y_key)
            if cr is None or val is None: continue
            if isinstance(val,str): continue
            vv = (map_fn(val) if map_fn else val)
            if vv <= 0 and y_key != "throughput_mbs": vv = 1e-6
            ax.scatter(cr, vv, c=s["c"], marker=s["m"], s=70,
                       edgecolors="#333", linewidths=0.4, zorder=2)
            if fk == "jpeg2000":
                op = r.get("operating_point","")
                tag = op.replace("cr=","") if "cr=" in op else ("L" if "lossless" in op else "")
                if tag: ax.annotate(tag, (cr, vv), textcoords="offset points",
                                    xytext=(8,-6), fontsize=6.5, color=s["c"], alpha=0.85)
    if log_y: ax.set_yscale("log")
    ax.set_xscale("log"); ax.set_xticks(XTICKS); ax.set_xticklabels([str(t) for t in XTICKS], fontsize=8)
    ax.set_xlabel("Compression Ratio", fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.25, which="major", ls="-", lw=0.3)

def add_legend(ax, ncol=2):
    h,l = ax.get_legend_handles_labels(); seen={}
    for hh,ll in zip(h,l):
        if ll and ll not in seen: seen[ll]=hh
    if seen:
        leg=ax.legend(seen.values(),seen.keys(),ncol=ncol,loc="best",fontsize=8)
        leg.get_frame().set_linewidth(0.5)

# ============ Classify & plot ============

rows = load_data()
print(f"Total codecs: {len(rows)}")

LL_M = ["lz4","zlib","ccsds123","jpegls_lossless","jpeg2000_lossless","klt_dwt_nc111_cr1"]
def classify(m):
    if any(m == x or m.startswith(x) for x in LL_M): return "ll"
    if m.startswith("jpegls_ne"): return "near"
    if m in ("jpeg2000_cr1.5","jpeg2000_cr2","jpeg2000_cr2.5","jpeg2000_cr3"): return "near"
    if m.startswith("klt_dwt") and "nc111" not in m:
        match = re.search(r"cr([\d.]+)", m)
        if match and float(match.group(1)) <= 2.0: return "near"
    return "lossy"

for r in rows: r["_cat"] = classify(r["method"])
ll = [r for r in rows if r["_cat"]=="ll"]
near = [r for r in rows if r["_cat"]=="near"]
lossy = [r for r in rows if r["_cat"]=="lossy"]
print(f"Lossless: {len(ll)}, Near: {len(near)}, Lossy: {len(lossy)}")

# --- 1. Lossless throughput ---
gl = group_rows(rows, ["lz4","zlib","ccsds123","jpegls","jpeg2000","klt_dwt_nc111"], restrict_to_rows=ll)
fig,ax=plt.subplots(figsize=(5.5,4.0))
plot_panel(ax,gl,"throughput_mbs","Throughput (MB/s)",log_y=True,
           title="Lossless — CR vs Throughput",
           label_override={"jpeg2000":"JPEG2000$_{5/3}$","klt_dwt_nc111":"KLT+DWT$_{111}$"})
ax.set_xscale("linear"); ax.set_xlim(0.8,2.6); ax.set_xticks([1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4])
ax.set_xticklabels(["1.0","1.2","1.4","1.6","1.8","2.0","2.2","2.4"], fontsize=8)
ax.set_ylim(1,2000); add_legend(ax,2); fig.tight_layout()
fig.savefig(f"{OUT}/plt_lossless_throughput.pdf"); plt.close(fig)
print("[1/7]")

# --- 2-4. Near-lossless (linear X, tight range) ---
nf = ["jpegls","jpeg2000","klt_dwt_nc28","klt_dwt_nc56"]
gn = group_rows(rows, nf, restrict_to_rows=near)
for idx,(key,ylab,logy,title) in enumerate([
    ("throughput_mbs","Throughput (MB/s)",True,"Near-Lossless — CR vs Throughput"),
    ("psnr","PSNR (dB)",False,"Near-Lossless — CR vs PSNR"),
    ("pae","PAE (DN)",True,"Near-Lossless — CR vs PAE"),
], start=2):
    fig,ax=plt.subplots(figsize=(5.5,4.3))
    fn = {"psnr":f_psnr}.get(key)
    plot_panel(ax,gn,key,ylab,log_y=logy,title=title,map_fn=fn,
               label_override={"jpeg2000":"JPEG2000$_{cr=1.5-4}$",
                               "klt_dwt_nc28":"KLT+DWT$_{28}^{cr \\leq 2}$",
                               "klt_dwt_nc56":"KLT+DWT$_{56}^{cr \\leq 2}$"})
    # Linear X for near-lossless (data ranges 1.5–7.4 CR)
    ax.set_xscale("linear")
    ax.set_xlim(1.2, 7.8)
    ax.set_xticks([1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7])
    ax.set_xticklabels(["1.5","2","2.5","3","3.5","4","5","6","7"], fontsize=8)
    # Throughput: linear Y, 0-10
    if key == "throughput_mbs":
        ax.set_yscale("linear")
        ax.set_yticks([0,2,4,6,8,10,12,14,16,18])
        ax.set_yticklabels(["0","2","4","6","8","10","12","14","16","18"], fontsize=8)
        ax.set_ylim(0, 18.5)
    add_legend(ax,2); fig.tight_layout()
    tag = key.replace("throughput_mbs","throughput")
    fig.savefig(f"{OUT}/plt_near_{tag}.pdf"); plt.close(fig)
    print(f"[{idx}/7]")

# --- 5-7. Lossy ---
lf = ["jpeg2000","klt_dwt_nc28","klt_dwt_nc56","klt_dwt_nc111","cae1d","cae3d","sscnet","hycot","hycass"]
gy = group_rows(rows, lf, restrict_to_rows=lossy)
for idx,(key,ylab,logy,title) in enumerate([
    ("psnr","PSNR (dB)",False,"Lossy — CR vs PSNR"),
    ("pae","PAE (DN)",True,"Lossy — CR vs PAE"),
    ("throughput_mbs","Throughput (MB/s)",True,"Lossy — CR vs Throughput"),
], start=5):
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    fn = {"psnr":f_psnr}.get(key)
    plot_panel(ax,gy,key,ylab,log_y=logy,title=title,map_fn=fn,
               label_override={"jpeg2000":"JPEG2000$_{MCT}$ (cr=4–500)",
                               "klt_dwt_nc28":"KLT+DWT$_{28}^{cr \\geq 4}$",
                               "klt_dwt_nc56":"KLT+DWT$_{56}^{cr \\geq 4}$",
                               "klt_dwt_nc111":"KLT+DWT$_{111}^{cr \\geq 2}$"})
    ax.set_xlim(3, 1100)
    add_legend(ax,2); fig.tight_layout()
    tag = key.replace("throughput_mbs","throughput")
    fig.savefig(f"{OUT}/plt_lossy_{tag}.pdf"); plt.close(fig)
    print(f"[{idx}/7]")

print(f"\n7 plots saved to {OUT}/")

"""Plotting utilities for compression benchmark results.

Generates:
  1. Rate-distortion: CR vs PSNR  (colour by method, annotate throughput)
  2. Rate-distortion: CR vs PAE
  3. Throughput-scatter: throughput vs CR (bubble size = PAE)
  4. Same-throughput-band comparison table
"""

from __future__ import annotations

import csv
import math
import os
from typing import Sequence

import numpy as np

# Lazy matplotlib import
try:
    import matplotlib
    matplotlib.use("Agg")                                      # non-interactive
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# Per-method colour/ marker config (extensible)
STYLE = {
    "klt_dwt":     {"color": "#1f77b4", "marker": "s", "label": "KLT+DWT"},
    "jpeg2000":    {"color": "#ff7f0e", "marker": "D", "label": "JPEG2000"},
    "ccsds123":    {"color": "#2ca02c", "marker": "o", "label": "CCSDS-123"},
    "hycass":      {"color": "#d62728", "marker": "^", "label": "HyCASS"},
}


def _style_for(name: str) -> dict:
    for prefix, s in STYLE.items():
        if name.startswith(prefix):
            return s
    return {"color": "#7f7f7f", "marker": "x", "label": name}


def _finite_psnr(psnr_val: float, ceil: float = 100.0) -> float:
    """Map inf PSNR (lossless) to a finite ceiling for plotting."""
    if psnr_val is None:
        return None
    if math.isinf(psnr_val) or not np.isfinite(psnr_val):
        return ceil
    return float(psnr_val)


# ---------------------------------------------------------------------------
# Figure 1: CR vs PSNR
# ---------------------------------------------------------------------------

def plot_cr_vs_psnr(
    rows: Sequence[dict],
    out_path: str = "results/cr_vs_psnr.pdf",
    figsize=(9, 6),
):
    if not HAVE_MPL:
        print("[plots] matplotlib not available, skipping CR-vs-PSNR")
        return

    fig, ax = plt.subplots(figsize=figsize)

    seen_labels = set()
    for row in rows:
        s = _style_for(row.get("method", ""))
        cr = row.get("cr", 0)
        psnr = _finite_psnr(row.get("psnr"), ceil=120)
        tp = row.get("throughput_mbs", 0)
        lbl = s["label"] if s["label"] not in seen_labels else None
        seen_labels.add(s["label"])
        ax.scatter(cr, psnr, c=s["color"], marker=s["marker"], s=80,
                   edgecolors="k", linewidth=0.3, zorder=3,
                   label=lbl)

        # throughput annotation
        if tp > 0:
            ax.annotate(f"{tp:.0f}", (cr, psnr), textcoords="offset points",
                        xytext=(4, -10), fontsize=6, color="gray")

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Rate-Distortion: CR vs PSNR", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Unique legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] CR-vs-PSNR saved to {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: CR vs PAE
# ---------------------------------------------------------------------------

def plot_cr_vs_pae(
    rows: Sequence[dict],
    out_path: str = "results/cr_vs_pae.pdf",
    figsize=(9, 6),
):
    if not HAVE_MPL:
        print("[plots] matplotlib not available, skipping CR-vs-PAE")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for row in rows:
        s = _style_for(row.get("method", ""))
        cr = row.get("cr", 0)
        pae = row.get("pae", 0)
        if pae == 0:
            pae = 0.05                                            # tiny marker for lossless
        ax.scatter(cr, pae, c=s["color"], marker=s["marker"], s=80,
                   edgecolors="k", linewidth=0.3, zorder=3)

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("PAE (DN)", fontsize=12)
    ax.set_title("Rate-Distortion: CR vs PAE", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] CR-vs-PAE saved to {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Throughput vs CR (scatter with PAE colour)
# ---------------------------------------------------------------------------

def plot_throughput_vs_cr(
    rows: Sequence[dict],
    out_path: str = "results/throughput_vs_cr.pdf",
    figsize=(9, 6),
):
    if not HAVE_MPL:
        print("[plots] matplotlib not available, skipping throughput-vs-CR")
        return

    fig, ax = plt.subplots(figsize=figsize)

    pae_vals = [r.get("pae", 1e-9) for r in rows]
    pae_norm = matplotlib.colors.LogNorm(vmin=max(min(pae_vals), 1e-3), vmax=max(pae_vals))

    for row in rows:
        s = _style_for(row.get("method", ""))
        cr = row.get("cr", 0)
        tp = row.get("throughput_mbs", 0)
        pae = max(row.get("pae", 0), 1e-3)
        ax.scatter(cr, tp, c=[pae], cmap="RdYlGn_r", norm=pae_norm,
                   marker=s["marker"], s=100, edgecolors="k", linewidth=0.3,
                   zorder=3)

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("Throughput (MB/s)", fontsize=12)
    ax.set_title("Throughput vs Compression Ratio (colour = PAE)", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] Throughput-vs-CR saved to {out_path}")


# ---------------------------------------------------------------------------
# Table: same-throughput-band comparison
# ---------------------------------------------------------------------------

def throughput_band_table(
    rows: Sequence[dict],
    bands_mbs: Sequence[float] = (10, 50, 100, 200, 500),
    tolerance: float = 0.3,
    out_path: str = "results/throughput_band_table.csv",
):
    """Group results by throughput band and produce a comparison CSV."""

    table = []
    for row in rows:
        tp = row.get("throughput_mbs", 0)
        if tp <= 0:
            continue
        for lo, hi in zip(bands_mbs[:-1], bands_mbs[1:]):
            if lo * (1 - tolerance) <= tp <= hi * (1 + tolerance):
                table.append({
                    "throughput_band": f"{lo}-{hi} MB/s",
                    **{k: row[k] for k in ["method", "family", "operating_point",
                                            "cr", "psnr", "pae", "throughput_mbs"]},
                })
                break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()) if table else ["throughput_band"])
        w.writeheader()
        w.writerows(table)
    print(f"[plots] Throughput-band table saved to {out_path} ({len(table)} entries)")
    return table


# ---------------------------------------------------------------------------
# Generate all plots from a results CSV
# ---------------------------------------------------------------------------

def make_all_plots(csv_path: str, out_dir: str = "results/"):
    """Read a benchmark CSV and produce all figures."""
    rows = []
    with open(csv_path, newline="") as f:
        for d in csv.DictReader(f):
            for key in ["cr", "bpppc", "mse", "mae", "psnr", "pae",
                         "enc_time_s", "dec_time_s", "throughput_mbs"]:
                try:
                    d[key] = float(d[key])
                except (ValueError, TypeError):
                    d[key] = None
            rows.append(d)

    if not rows:
        print("[plots] No data in CSV, skipping all plots")
        return

    plot_cr_vs_psnr(rows, os.path.join(out_dir, "cr_vs_psnr.pdf"))
    plot_cr_vs_pae(rows, os.path.join(out_dir, "cr_vs_pae.pdf"))
    plot_throughput_vs_cr(rows, os.path.join(out_dir, "throughput_vs_cr.pdf"))
    throughput_band_table(rows, out_path=os.path.join(out_dir, "throughput_band_table.csv"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Benchmark results CSV")
    ap.add_argument("--out-dir", default="results/")
    args = ap.parse_args()
    make_all_plots(args.csv, args.out_dir)

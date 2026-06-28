"""Unified quality metrics, all computed in the integer DN domain.

These mirror the repo's existing metrics (metrics/mse.py, psnr.py, sa.py) but
operate on numpy DN arrays so they apply identically to *every* codec
(learned or classical). PAE is added -- it is the defining metric for
near-lossless coding (CCSDS-123.0-B-2) and is 0 for truly lossless codecs.

All functions take (original_dn, reconstructed_dn) as numpy arrays of shape
(C, H, W) in the same integer domain and return python floats.
"""

from __future__ import annotations

import numpy as np


def mse(a_dn: np.ndarray, b_dn: np.ndarray) -> float:
    """Mean squared error in DN^2."""
    a = np.asarray(a_dn, dtype=np.float64)
    b = np.asarray(b_dn, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a_dn: np.ndarray, b_dn: np.ndarray, peak: float) -> float:
    """Peak signal-to-noise ratio in dB, relative to `peak` (e.g. 65535).

    Returns +inf for a perfect (lossless) reconstruction.
    """
    err = mse(a_dn, b_dn)
    if err == 0.0:
        return float("inf")
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(err))


def pae(a_dn: np.ndarray, b_dn: np.ndarray) -> float:
    """Peak absolute error (max |a-b|) in DN. 0 iff lossless.

    This is the quantity CCSDS-123.0-B-2 bounds directly; reporting it lets us
    place near-lossless and lossy codecs on the same error axis.
    """
    a = np.asarray(a_dn, dtype=np.int64)
    b = np.asarray(b_dn, dtype=np.int64)
    return float(np.max(np.abs(a - b)))


def mae(a_dn: np.ndarray, b_dn: np.ndarray) -> float:
    """Mean absolute error in DN."""
    a = np.asarray(a_dn, dtype=np.float64)
    b = np.asarray(b_dn, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def spectral_angle(a_dn: np.ndarray, b_dn: np.ndarray) -> float:
    """Mean spectral angle (degrees) over all pixels.

    Matches metrics/sa.py: angle between original and reconstructed spectral
    vectors (axis 0 = bands). Returns nan-safe mean.
    """
    a = np.asarray(a_dn, dtype=np.float64)
    b = np.asarray(b_dn, dtype=np.float64)
    num = np.sum(a * b, axis=0)
    den = np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.clip(num / den, -1.0, 1.0)
    ang = np.degrees(np.arccos(frac))
    return float(np.nanmean(ang))


def all_metrics(a_dn: np.ndarray, b_dn: np.ndarray, peak: float) -> dict:
    """Compute the full metric set at once."""
    return {
        "mse": mse(a_dn, b_dn),
        "mae": mae(a_dn, b_dn),
        "psnr": psnr(a_dn, b_dn, peak),
        "pae": pae(a_dn, b_dn),
        "sa": spectral_angle(a_dn, b_dn),
    }

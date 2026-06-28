"""Unified data domain for fair cross-codec comparison.

THE PROBLEM
-----------
Different codecs naturally live in different numeric domains:
  - HyCASS expects float32 in [0, 1] (the dataset is global min-max normalized).
  - CCSDS-123 and JPEG2000 operate on integer samples (here: 16-bit DN).
  - PAE (peak absolute error) and PSNR depend entirely on the value range,
    so comparing a PAE computed in [0,1] against one in [0,65535] is meaningless.

THE SOLUTION
------------
We pick ONE reference integer domain -- unsigned `bit_depth`-bit DN -- and define
every metric there. The Berlin / HySpecNet patches are stored as float [0,1];
the original sensor data was integer. We reconstruct an integer DN domain by
scaling float [0,1] by `dn_scale` and rounding. `dn_scale` defaults to 10000,
matching what the repo's own JPEG2000 baseline (models/_jpeg2000.py) uses
(`x = 10_000 * x; x.astype(uint16)`), so our numbers stay consistent with the
existing baselines.

Pipeline per sample:
    float[0,1]  --to_dn-->  uint16 DN  --(codec encode/decode)-->  uint16 DN_hat
    metrics (MSE/PSNR/PAE) are computed in the DN domain.
    PSNR uses peak = dn_peak (default 2**bit_depth - 1).

`bitdepth_native` is the bit depth used to define the *uncompressed reference
size* for the compression ratio:  raw_bits = C*H*W*bitdepth_native.
Set it to the sensor's true bit depth (HyMap is 16-bit) so CR is reported
against a realistic uncompressed size rather than against float32.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Domain:
    """Defines the shared numeric domain in which all codecs and metrics operate.

    Attributes:
        bit_depth: bit depth of the integer DN samples fed to the codecs (e.g. 16).
        dn_scale: multiplier mapping float [0,1] -> DN integer range.
        bitdepth_native: bit depth defining the uncompressed reference size used
            for compression-ratio computation. Usually equals bit_depth.
    """

    bit_depth: int = 16
    dn_scale: float = 10_000.0
    bitdepth_native: int = 16

    @property
    def dn_peak(self) -> int:
        """Maximum representable DN value (used as PSNR peak).

        Uses the actual data range (dn_scale) rather than the container's
        theoretical max (2^bit_depth - 1).  For Berlin-Urban-Gradient the
        global min-max normalization maps values to [0, 10000].
        """
        return int(self.dn_scale)

    @classmethod
    def from_data_dir(cls, data_dir: str) -> "Domain":
        """Create a Domain from a patchified dataset directory.

        Reads metadata.json if present to get vmax (actual DN peak).
        Otherwise falls back to DEFAULT_DOMAIN.
        """
        import json, os
        meta_path = os.path.join(data_dir, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            vmax = meta.get("vmax", 10000)
            return cls(
                bit_depth=int(meta.get("bit_depth", 16)),
                dn_scale=float(vmax),
                bitdepth_native=int(meta.get("bit_depth", 16)),
            )
        return DEFAULT_DOMAIN

    @property
    def np_dtype(self):
        if self.bit_depth <= 8:
            return np.uint8
        if self.bit_depth <= 16:
            return np.uint16
        return np.uint32

    def to_dn(self, x_float: np.ndarray) -> np.ndarray:
        """float [0,1] -> integer DN, clipped to [0, dn_peak]."""
        dn = np.rint(np.asarray(x_float, dtype=np.float64) * self.dn_scale)
        dn = np.clip(dn, 0, self.dn_peak)
        return dn.astype(self.np_dtype)

    def to_float(self, x_dn: np.ndarray) -> np.ndarray:
        """integer DN -> float [0,1] (inverse of to_dn, up to rounding)."""
        return (np.asarray(x_dn, dtype=np.float64) / self.dn_scale).astype(np.float32)

    def raw_bits(self, shape) -> int:
        """Uncompressed reference size in bits for an array of `shape` (C,H,W)."""
        n = int(np.prod(shape))
        return n * self.bitdepth_native


# Sensible default shared by the whole benchmark.
DEFAULT_DOMAIN = Domain(bit_depth=16, dn_scale=10_000.0, bitdepth_native=16)

"""JPEG2000 with global KLT spectral pre-decorrelation.

Pipeline (encode):
  1. Apply pre-computed global PCA basis to decorrelate spectral bands.
  2. Scale float PCA coordinates to uint16 (per-component, lossless rescale).
  3. Encode with standard JPEG2000.
  4. Store: scale factors (2×C float32) + JP2 blob.

Pipeline (decode):
  1. Decode JPEG2000.
  2. Rescale uint16 back to float PCA coordinates.
  3. Apply inverse PCA → original DN space.

The PCA basis is NOT stored in the payload — it must be provided at construction
time (fit on a representative training set).  This avoids sideinfo overhead and
gives the JPEG2000 encoder a fully decorrelated signal to work with.
"""

from __future__ import annotations

import struct
import numpy as np

from benchmark.codecs.base import Codec, EncodeResult
from benchmark.codecs import jp2_util
from benchmark.domain import Domain, DEFAULT_DOMAIN

_MAGIC = b"KJ2G"  # Global-KLT variant


class JPEG2000KLTCodec(Codec):
    """JPEG2000 with global-PCA spectral pre-decorrelation (no sideinfo overhead)."""

    device = "cpu"

    def __init__(
        self,
        cratio: float = 1.0,
        *,
        pca_basis: np.ndarray | None = None,   # (C, C) float64, eigenvectors
        pca_mean:  np.ndarray | None = None,   # (C,)   float64
        domain: Domain = DEFAULT_DOMAIN,
    ):
        super().__init__(domain)
        self.cratio   = float(cratio)
        self.lossless = self.cratio <= 1.0
        self.family   = "lossless" if self.lossless else "lossy"
        self.name     = "jpeg2000_klt_lossless" if self.lossless else f"jpeg2000_klt_cr{self.cratio:g}"

        self._basis = pca_basis   # (C, C): each row is one principal component
        self._mean  = pca_mean    # (C,)

    # ------------------------------------------------------------------
    # Fit PCA on a list of DN patches (call once before benchmarking)
    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, patches_dn: list[np.ndarray], cratio: float = 1.0,
            domain: Domain = DEFAULT_DOMAIN) -> "JPEG2000KLTCodec":
        from sklearn.decomposition import PCA
        C = patches_dn[0].shape[0]
        X = np.concatenate([p.reshape(C, -1).T for p in patches_dn], axis=0)
        pca = PCA(n_components=C, whiten=False)
        pca.fit(X)
        return cls(cratio=cratio,
                   pca_basis=pca.components_.astype(np.float64),
                   pca_mean=pca.mean_.astype(np.float64),
                   domain=domain)

    def operating_point(self) -> str:
        return "lossless" if self.lossless else f"cr={self.cratio:g}"

    def _forward(self, x_dn: np.ndarray):
        """DN (C,H,W) → decorrelated, rescaled to uint16 (C,H,W) + scale params.

        Uses the same glymur-style axis layout as JPEG2000CCodec for optimal
        compression: treat spectral dimension as spatial height.
        The decorrelated signal is rescaled per-band to [0, 65535] uint16.
        """
        C, H, W = x_dn.shape
        X = x_dn.reshape(C, H * W).T.astype(np.float64)   # (N, C)
        if self._mean is not None:
            X -= self._mean
        coords = X @ self._basis.T                          # (N, C) decorrelated

        coords = coords.T.reshape(C, H, W)                  # (C, H, W)

        # Per-band rescale: map [min, max] → [0, 65535]
        lo     = coords.reshape(C, -1).min(axis=1).astype(np.float32)
        hi     = coords.reshape(C, -1).max(axis=1).astype(np.float32)
        scales = np.where(hi - lo > 1e-9,
                          65535.0 / (hi - lo), 1.0).astype(np.float32)

        cube = np.zeros((C, H, W), dtype=np.uint16)
        for k in range(C):
            q = np.rint((coords[k] - lo[k]) * scales[k]).astype(np.int32)
            cube[k] = np.clip(q, 0, 65535).astype(np.uint16)
        return cube, lo, scales

    def _backward(self, cube: np.ndarray, lo: np.ndarray, scales: np.ndarray,
                  C: int, H: int, W: int) -> np.ndarray:
        """Decorrelated uint16 + scale params → DN (C,H,W)."""
        coords = np.zeros((C, H * W), dtype=np.float64)
        for k in range(C):
            coords[k] = cube[k].reshape(-1).astype(np.float64) / scales[k] + lo[k]
        coords = coords.T                                    # (N, C)
        X_hat = coords @ self._basis                        # (N, C)
        if self._mean is not None:
            X_hat += self._mean
        x = X_hat.T.reshape(C, H, W)
        x = np.rint(np.clip(x, 0, self.domain.dn_peak))
        return x.astype(self.domain.np_dtype)

    def encode(self, x_dn: np.ndarray) -> EncodeResult:
        C, H, W = x_dn.shape
        if self._basis is None:
            raise RuntimeError("PCA basis not set — call JPEG2000KLTCodec.fit() first")

        cube, lo, scales = self._forward(x_dn)

        # Use C API codec with uint16 domain (cube values are in [0,65535])
        from benchmark.codecs.jpeg2000_c import JPEG2000CCodec as _C
        from benchmark.domain import Domain
        _uint16_domain = Domain(bit_depth=16, dn_scale=65535.0, bitdepth_native=16)
        _inner = _C(cratio=self.cratio, domain=_uint16_domain)
        inner_enc = _inner.encode(cube)

        payload = bytearray()
        payload += _MAGIC
        payload += struct.pack("<III", C, H, W)
        payload += lo.astype(np.float32).tobytes()      # C × 4 bytes
        payload += scales.tobytes()                     # C × 4 bytes
        payload += struct.pack("<I", len(inner_enc.payload))
        payload += inner_enc.payload
        return EncodeResult(payload=bytes(payload), meta={"C": C, "H": H, "W": W})

    def decode(self, enc: EncodeResult) -> np.ndarray:
        blob = enc.payload
        assert blob[:4] == _MAGIC, f"bad magic: {blob[:4]}"
        off = 4
        C, H, W = struct.unpack_from("<III", blob, off); off += 12
        lo     = np.frombuffer(blob, dtype=np.float32, count=C, offset=off).copy(); off += C * 4
        scales = np.frombuffer(blob, dtype=np.float32, count=C, offset=off).copy(); off += C * 4
        (ln,)  = struct.unpack_from("<I", blob, off); off += 4
        inner_payload = blob[off:off + ln]

        from benchmark.codecs.jpeg2000_c import JPEG2000CCodec as _C, EncodeResult as _ER
        from benchmark.domain import Domain
        _uint16_domain = Domain(bit_depth=16, dn_scale=65535.0, bitdepth_native=16)
        _inner = _C(cratio=self.cratio, domain=_uint16_domain)
        cube = _inner.decode(_ER(payload=inner_payload, meta={"C": C, "H": H, "W": W}))
        return self._backward(cube, lo, scales, C, H, W)

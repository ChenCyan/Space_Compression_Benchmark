"""KLT + DWT codec (spectral PCA + multi-component spatial JPEG2000).

Uses sklearn.decomposition.PCA for the spectral transform and glymur for
multi-component JPEG2000 spatial coding. All scaled components are stacked
and encoded as ONE multi-component JP2, eliminating per-component temp-file I/O.
"""

from __future__ import annotations

import struct

import numpy as np
from sklearn.decomposition import PCA

from benchmark.codecs.base import Codec, EncodeResult
from benchmark.codecs import jp2_util
from benchmark.domain import Domain, DEFAULT_DOMAIN

_MAGIC = b"KLTD"


class KLTDWTCodec(Codec):
    family = "near-lossless"
    device = "cpu"

    def __init__(
        self,
        n_components: int,
        cratio: float = 1.0,
        *,
        domain: Domain = DEFAULT_DOMAIN,
    ):
        super().__init__(domain)
        self.n_components = int(n_components)
        self.cratio = float(cratio)
        self.name = f"klt_dwt_nc{self.n_components}_cr{self.cratio:g}"

    def operating_point(self) -> str:
        return f"nc={self.n_components},jp2cr={self.cratio:g}"

    def encode(self, x_dn: np.ndarray) -> EncodeResult:
        C, H, W = x_dn.shape
        nc = min(self.n_components, C)

        X = x_dn.reshape(C, H * W).T.astype(np.float64)
        pca = PCA(n_components=nc, whiten=False)
        comps = pca.fit_transform(X)
        comps = comps.T.reshape(nc, H, W)

        basis = pca.components_.astype(np.float32)
        mean = pca.mean_.astype(np.float32)

        # Scale each component to uint16 and stack into a cube.
        cube = np.zeros((nc, H, W), dtype=np.uint16)
        scales = np.zeros(nc, dtype=np.float32)
        for k in range(nc):
            c = comps[k]
            amax = float(np.max(np.abs(c)))
            if amax < 1e-12:
                scales[k] = 1.0
            else:
                scale = 32000.0 / amax
                scales[k] = np.float32(scale)
                q_int16 = np.rint(c * scale).astype(np.int16)
                cube[k] = (q_int16.astype(np.int32) + 32768).astype(np.uint16)

        lossless = self.cratio <= 1.0
        pl = jp2_util.encode_multicomponent(cube, lossless=lossless,
                                            cratio=self.cratio if not lossless else None)

        payload = self._pack(C, H, W, nc, basis, mean, scales, pl)
        return EncodeResult(payload=payload, meta={"C": C, "H": H, "W": W, "nc": nc})

    def decode(self, enc: EncodeResult) -> np.ndarray:
        C, H, W, nc, basis, mean, scales, pl = self._unpack(enc.payload)
        cube = jp2_util.decode_multicomponent(pl)                     # (nc, H, W)

        comps = np.zeros((nc, H * W), dtype=np.float64)
        for k in range(nc):
            q = cube[k].astype(np.int32) - 32768
            comps[k] = (q.astype(np.float64) / scales[k]).reshape(H * W)

        X = comps.T @ basis + mean
        x = X.T.reshape(C, H, W)
        x = np.rint(np.clip(x, 0, self.domain.dn_peak))
        return x.astype(self.domain.np_dtype)

    @staticmethod
    def _pack(C, H, W, nc, basis, mean, scales, pl) -> bytes:
        out = bytearray()
        out += _MAGIC
        out += struct.pack("<IIII", C, H, W, nc)
        out += basis.tobytes()
        out += mean.tobytes()
        out += scales.tobytes()
        out += struct.pack("<I", len(pl))
        out += pl
        return bytes(out)

    @staticmethod
    def _unpack(blob: bytes):
        assert blob[:4] == _MAGIC, "bad KLTDWT payload"
        off = 4
        C, H, W, nc = struct.unpack_from("<IIII", blob, off); off += 16
        bsz = nc * C * 4
        basis = np.frombuffer(blob, dtype=np.float32, count=nc*C, offset=off).reshape(nc, C); off += bsz
        mean = np.frombuffer(blob, dtype=np.float32, count=C, offset=off); off += C*4
        scales = np.frombuffer(blob, dtype=np.float32, count=nc, offset=off); off += nc*4
        (ln,) = struct.unpack_from("<I", blob, off); off += 4
        pl = blob[off:off+ln]
        return C, H, W, nc, basis, mean, scales, pl

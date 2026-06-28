"""Standalone JPEG2000 codec with multi-component encoding for high throughput.

All spectral bands are encoded as ONE multi-component JP2 (not C separate JP2s).
This eliminates per-band temp-file I/O and lets OpenJPEG exploit spectral
redundancy via its internal multi-component transform (MCT).

  - lossless: 5/3 reversible DWT, PAE=0.
  - lossy:    9/7 irreversible DWT at the given compression ratio.

When ``num_threads > 1``, the codec enables OpenJPEG's internal thread pool
(via ``glymur.set_option("lib.num_threads", N)``) for code-block-level
parallelism.  Requires OpenJPEG >= 2.4.0 (the default Ubuntu 20.04 package
is 2.3.1 — compile 2.5.2 from source to unlock this).
"""

from __future__ import annotations

import struct

import numpy as np

from benchmark.codecs.base import Codec, EncodeResult
from benchmark.codecs import jp2_util
from benchmark.domain import Domain, DEFAULT_DOMAIN

_MAGIC = b"JP2K"


class JPEG2000Codec(Codec):
    device = "cpu"

    def __init__(self, cratio: float = 1.0, *, num_threads: int = 1,
                 domain: Domain = DEFAULT_DOMAIN):
        super().__init__(domain)
        self.cratio = float(cratio)
        self.lossless = self.cratio <= 1.0
        self.num_threads = int(num_threads)
        self.family = "lossless" if self.lossless else "lossy"
        base = "jpeg2000_lossless" if self.lossless else f"jpeg2000_cr{self.cratio:g}"
        self.name = f"{base}_th{self.num_threads}" if self.num_threads > 1 else base

    def operating_point(self) -> str:
        suffix = f"+thr{self.num_threads}" if self.num_threads > 1 else ""
        return ("lossless" if self.lossless else f"cr={self.cratio:g}") + suffix

    def encode(self, x_dn: np.ndarray) -> EncodeResult:
        C, H, W = x_dn.shape
        cube = x_dn.astype(self.domain.np_dtype)          # (C, H, W)

        # Enable OpenJPEG internal thread pool (requires OpenJPEG >= 2.4.0).
        if self.num_threads > 1:
            import glymur
            glymur.set_option("lib.num_threads", self.num_threads)

        pl = jp2_util.encode_multicomponent(cube, lossless=self.lossless,
                                             cratio=self.cratio if not self.lossless else None)

        if self.num_threads > 1:
            import glymur
            glymur.set_option("lib.num_threads", 1)

        out = bytearray()
        out += _MAGIC
        out += struct.pack("<III", C, H, W)
        out += struct.pack("<I", len(pl))
        out += pl
        return EncodeResult(payload=bytes(out), meta={"C": C, "H": H, "W": W})

    def decode(self, enc: EncodeResult) -> np.ndarray:
        blob = enc.payload
        assert blob[:4] == _MAGIC
        C, H, W = struct.unpack_from("<III", blob, 4)
        (ln,) = struct.unpack_from("<I", blob, 16)

        # Also enable threads for decode (supported since OpenJPEG 2.2.0).
        if self.num_threads > 1:
            import glymur
            glymur.set_option("lib.num_threads", self.num_threads)

        result = jp2_util.decode_multicomponent(blob[20:20 + ln])

        if self.num_threads > 1:
            import glymur
            glymur.set_option("lib.num_threads", 1)

        return result

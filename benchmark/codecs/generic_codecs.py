"""General-purpose lossless codecs: LZ4 (fast) and zlib/Deflate (standard).

These serve as "no-signal-model" baselines — they treat the image as raw bytes
with no spectral or spatial structure, showing the compression floor that any
specialised codec should beat.

zlib supports multi-threaded band-split compression (pigz-style).
"""

from __future__ import annotations

import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from benchmark.codecs.base import Codec, EncodeResult
from benchmark.domain import Domain, DEFAULT_DOMAIN

_MAGIC_LZ4 = b"LZ4B"
_MAGIC_ZLIB = b"ZLIB"
_MAGIC_ZLIB_MT = b"ZLMT"  # multi-threaded format marker


class LZ4Codec(Codec):
    """LZ4 compression (very fast, general-purpose)."""
    device = "cpu"
    family = "lossless"

    def __init__(self, *, domain: Domain = DEFAULT_DOMAIN):
        super().__init__(domain)
        self.name = "lz4"

    def operating_point(self) -> str:
        return "lossless"

    def encode(self, x_dn: np.ndarray) -> EncodeResult:
        import lz4.frame
        C, H, W = x_dn.shape
        raw = x_dn.astype(self.domain.np_dtype).tobytes()
        compressed = lz4.frame.compress(raw, compression_level=1)
        out = _MAGIC_LZ4 + struct.pack("<III", C, H, W) + compressed
        return EncodeResult(payload=out, meta={"C": C, "H": H, "W": W})

    def decode(self, enc: EncodeResult) -> np.ndarray:
        import lz4.frame
        assert enc.payload[:4] == _MAGIC_LZ4
        C, H, W = struct.unpack_from("<III", enc.payload, 4)
        raw = lz4.frame.decompress(enc.payload[16:])
        return np.frombuffer(raw, dtype=self.domain.np_dtype).reshape(C, H, W)


class ZlibCodec(Codec):
    """zlib/Deflate compression (standard, moderate speed).
    
    Multi-threaded mode splits the cube by band, compresses each independently,
    and concatenates. This enables parallelism at the cost of ~5-15% CR loss
    (cross-band Deflate matches are lost).
    """
    device = "cpu"
    family = "lossless"

    def __init__(self, level: int = 6, *, domain: Domain = DEFAULT_DOMAIN, num_threads: int = 1):
        super().__init__(domain)
        self.level = int(level)
        self.num_threads = max(1, int(num_threads))
        self.name = "zlib"

    def operating_point(self) -> str:
        return f"level={self.level}"

    def encode(self, x_dn: np.ndarray) -> EncodeResult:
        C, H, W = x_dn.shape
        
        if self.num_threads == 1:
            raw = x_dn.astype(self.domain.np_dtype).tobytes()
            compressed = zlib.compress(raw, level=self.level)
            out = _MAGIC_ZLIB + struct.pack("<IIII", C, H, W, self.level) + compressed
            return EncodeResult(payload=out, meta={"C": C, "H": H, "W": W})
        
        # Multi-threaded: compress each band independently
        def _compress_band(c):
            band = x_dn[c].astype(self.domain.np_dtype).tobytes()
            return zlib.compress(band, level=self.level)
        
        n_workers = min(self.num_threads, C)
        compressed = [None] * C
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_compress_band, c): c for c in range(C)}
            for future in as_completed(futures):
                c = futures[future]
                compressed[c] = future.result()
        
        # Format: MAGIC + C + H + W + level + [len_band0, len_band1, ...] + [band0, band1, ...]
        header = _MAGIC_ZLIB_MT + struct.pack("<IIII", C, H, W, self.level)
        lengths = struct.pack("<" + "I" * C, *[len(b) for b in compressed])
        out = header + lengths + b"".join(compressed)
        return EncodeResult(payload=out, meta={"C": C, "H": H, "W": W, "mt": True})

    def decode(self, enc: EncodeResult) -> np.ndarray:
        blob = enc.payload
        is_mt = blob[:4] == _MAGIC_ZLIB_MT
        
        if not is_mt:
            assert blob[:4] == _MAGIC_ZLIB
            C, H, W, _ = struct.unpack_from("<IIII", blob, 4)
            raw = zlib.decompress(blob[20:])
            return np.frombuffer(raw, dtype=self.domain.np_dtype).reshape(C, H, W)
        
        C, H, W, _ = struct.unpack_from("<IIII", blob, 4)
        off = 20
        lengths = list(struct.unpack_from("<" + "I" * C, blob, off))
        off += 4 * C
        
        def _decompress_band(start, ln):
            return np.frombuffer(zlib.decompress(blob[start:start + ln]), dtype=self.domain.np_dtype)
        
        if self.num_threads == 1:
            bands = []
            for ln in lengths:
                bands.append(_decompress_band(off, ln))
                off += ln
        else:
            bands = [None] * C
            n_workers = min(self.num_threads, C)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {}
                for c, ln in enumerate(lengths):
                    fut = pool.submit(_decompress_band, off, ln)
                    futures[fut] = c
                    off += ln
                for future in as_completed(futures):
                    c = futures[future]
                    bands[c] = future.result()
        
        out = np.zeros((C, H, W), dtype=self.domain.np_dtype)
        for c, band in enumerate(bands):
            out[c] = band.reshape(H, W)
        return out

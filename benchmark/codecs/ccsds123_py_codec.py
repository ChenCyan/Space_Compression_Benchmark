"""CCSDS-123.0-B-1 (lossless) codec using numba-accelerated predictor.

Lazy-imports compression_fast to avoid module-level crash when numba is
not pre-installed in the Docker image (installed at container startup).

Multi-threaded: encode bands in parallel via ThreadPoolExecutor.
Each band predictor is independent (weight vector reset at first pixel of each band).
"""

from __future__ import annotations

import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from benchmark.codecs.base import Codec, EncodeResult
from benchmark.domain import Domain, DEFAULT_DOMAIN

_FAST_DIR = os.path.join(os.path.dirname(__file__), "ccsds123_py")
_FAST = None


def _get_fast():
    """Lazy import of compression_fast (numba JIT only compiled once per process)."""
    global _FAST
    if _FAST is None:
        if _FAST_DIR not in sys.path:
            sys.path.insert(0, _FAST_DIR)
        import compression_fast as _mod
        _FAST = _mod
    return _FAST


class CCSDS123PyCodec(Codec):
    device = "cpu"
    family = "lossless"

    def __init__(self, lossless=True, pae_bound=0, *, domain=DEFAULT_DOMAIN, num_threads=1):
        super().__init__(domain)
        self.lossless = bool(lossless)
        self.pae_bound = int(pae_bound) if not self.lossless else 0
        self.family = "lossless" if self.lossless else "near-lossless"
        self.name = "ccsds123_lossless" if self.lossless else f"ccsds123_pae{self.pae_bound}"
        self.num_threads = max(1, int(num_threads))

    def operating_point(self):
        return "lossless" if self.lossless else f"PAE<={self.pae_bound}"

    def encode(self, x_dn: np.ndarray) -> EncodeResult:
        fast = _get_fast()
        C, H, W = x_dn.shape
        D_int = self.domain.bit_depth

        if self.num_threads == 1:
            data_bip = np.asarray(x_dn, dtype=np.int64).transpose(2, 1, 0).copy()
            mapped = fast.fast_predictor(data_bip, D=D_int, P=2, W_RES=4, R=32)
            bitdata, num_bits = fast.fast_encoder(mapped, D=D_int)
            return EncodeResult(
                payload=struct.pack("<IIIQ", C, H, W, num_bits) + bitdata,
                meta={"C": C, "H": H, "W": W, "num_bits": num_bits},
            )
        
        # Multi-threaded: encode each band independently
        data_bip = np.asarray(x_dn, dtype=np.int64).transpose(2, 1, 0).copy()  # (W, H, C)

        def _encode_band(z):
            """Predict + encode a single band, returns (bit_offset, bit_data, num_bits)."""
            band_3d = data_bip[:, :, z:z+1].copy()  # (W, H, 1)
            mapped = fast.fast_predictor(band_3d, D=D_int, P=2, W_RES=4, R=32)
            bitdata, num_bits = fast.fast_encoder(mapped, D=D_int)
            return (z, bitdata, num_bits)

        n_workers = min(self.num_threads, C)
        results = [None] * C
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_encode_band, z): z for z in range(C)}
            for future in as_completed(futures):
                z, bitdata, num_bits = future.result()
                results[z] = (bitdata, num_bits)

        # Concatenate per-band bitstreams, store per-band lengths
        total_bits = 0
        header = struct.pack("<III", C, H, W)
        header += struct.pack("<" + "I" * C, *[r[1] for r in results])
        bit_parts = []
        for bitdata, num_bits in results:
            bit_parts.append(bitdata)
            total_bits += num_bits

        return EncodeResult(
            payload=header + b"".join(bit_parts),
            meta={"C": C, "H": H, "W": W, "num_bits": total_bits, "mt": True},
        )

    def decode(self, enc: EncodeResult) -> np.ndarray:
        fast = _get_fast()
        D_int = self.domain.bit_depth
        
        # Check if multi-threaded encoded (has per-band bit lengths in header)
        mt = enc.meta.get("mt", False)
        
        if not mt:
            # Legacy single-threaded format
            C, H, W, num_bits = struct.unpack_from("<IIIQ", enc.payload, 0)
            bit_data = enc.payload[20:]
            decoded_mapped = fast.fast_decoder(bit_data, num_bits, W, H, C, D=D_int)
            data_bip = fast.fast_unpredictor(decoded_mapped, D=D_int, P=2, W_RES=4, R=32)
            x_dn = data_bip.transpose(2, 1, 0)
            x_dn = np.clip(x_dn, 0, self.domain.dn_peak)
            return x_dn.astype(self.domain.np_dtype)
        
        # Multi-threaded decode
        C, H, W = struct.unpack_from("<III", enc.payload, 0)
        off = 12
        band_nbits = list(struct.unpack_from("<" + "I" * C, enc.payload, off))
        off += 4 * C
        
        total_bits = sum(band_nbits)
        bit_data = enc.payload[off:]
        
        if self.num_threads == 1:
            decoded_mapped = fast.fast_decoder(bit_data, total_bits, W, H, C, D=D_int)
            data_bip = fast.fast_unpredictor(decoded_mapped, D=D_int, P=2, W_RES=4, R=32)
        else:
            # Decode each band independently
            def _decode_band(z, bit_offset, nbits):
                # Extract this band's bits from the concatenated stream
                byte_start = bit_offset // 8
                byte_end = (bit_offset + nbits + 7) // 8
                band_bitdata = bit_data[byte_start:byte_end]
                decoded = fast.fast_decoder(band_bitdata, nbits, W, H, 1, D=D_int)
                unpred = fast.fast_unpredictor(decoded, D=D_int, P=2, W_RES=4, R=32)
                return (z, unpred[:, :, 0])
            
            bit_offset = 0
            futures_map = {}
            n_workers = min(self.num_threads, C)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for z, nbits in enumerate(band_nbits):
                    fut = pool.submit(_decode_band, z, bit_offset, nbits)
                    futures_map[fut] = z
                    bit_offset += nbits
            
            data_bip = np.zeros((W, H, C), dtype=np.int64)
            for future in as_completed(futures_map):
                z, band = future.result()
                data_bip[:, :, z] = band
        
        x_dn = data_bip.transpose(2, 1, 0)
        x_dn = np.clip(x_dn, 0, self.domain.dn_peak)
        return x_dn.astype(self.domain.np_dtype)

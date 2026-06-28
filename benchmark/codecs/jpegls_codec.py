"""JPEG-LS codec via pyjpegls (CharLS C++ backend).
Multi-threaded: bands divided into num_threads chunks, each chunk processed serially.
"""
from __future__ import annotations
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from benchmark.codecs.base import Codec, EncodeResult
from benchmark.domain import Domain, DEFAULT_DOMAIN

_MAGIC = b"JPLS"

class JPEGLSCodec(Codec):
    device = "cpu"

    def __init__(self, lossless=True, lossy_error=0, *, domain=DEFAULT_DOMAIN, num_threads=1):
        super().__init__(domain)
        self.lossless = bool(lossless)
        self.lossy_error = int(lossy_error) if not self.lossless else 0
        self.family = "lossless" if self.lossless else "near-lossless"
        self.name = "jpegls_lossless" if self.lossless else f"jpegls_ne{self.lossy_error}"
        self.num_threads = max(1, int(num_threads))

    def operating_point(self):
        return "lossless" if self.lossless else f"NEAR={self.lossy_error}"

    def encode(self, x_dn):
        import jpeg_ls
        C, H, W = x_dn.shape

        def _encode_chunk(start, end):
            return [jpeg_ls.encode(x_dn[c].astype(np.uint16),
                    lossy_error=self.lossy_error).tobytes() for c in range(start, end)]

        if self.num_threads == 1:
            payloads = _encode_chunk(0, C)
        else:
            chunk_size = max(1, C // self.num_threads)
            chunks = []
            for i in range(0, C, chunk_size):
                chunks.append((i, min(i + chunk_size, C)))
            n_workers = min(self.num_threads, len(chunks))
            results = [None] * len(chunks)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_encode_chunk, s, e): idx for idx, (s, e) in enumerate(chunks)}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            payloads = []
            for r in results:
                payloads.extend(r)

        out = bytearray(_MAGIC)
        out += struct.pack("<IIII", C, H, W, int(self.lossy_error))
        for pl in payloads:
            out += struct.pack("<I", len(pl)) + pl
        return EncodeResult(payload=bytes(out), meta={"C": C, "H": H, "W": W})

    def decode(self, enc):
        import jpeg_ls
        blob = enc.payload
        assert blob[:4] == _MAGIC
        C, H, W, _ = struct.unpack_from("<IIII", blob, 4)
        off = 20
        band_info = []
        for c in range(C):
            (ln,) = struct.unpack_from("<I", blob, off)
            band_info.append((off + 4, ln))
            off = off + 4 + ln

        def _decode_chunk(indices):
            return [(c, jpeg_ls.decode(np.frombuffer(
                blob[band_info[c][0]:band_info[c][0]+band_info[c][1]], dtype=np.uint8)))
                    for c in indices]

        if self.num_threads == 1:
            bands_list = _decode_chunk(list(range(C)))
        else:
            chunk_size = max(1, C // self.num_threads)
            idx_chunks = [list(range(i, min(i + chunk_size, C))) for i in range(0, C, chunk_size)]
            n_workers = min(self.num_threads, len(idx_chunks))
            results = [None] * len(idx_chunks)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_decode_chunk, ic): idx for idx, ic in enumerate(idx_chunks)}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            bands_list = []
            for r in results:
                bands_list.extend(r)

        out = np.zeros((C, H, W), dtype=np.uint16)
        for c, band in bands_list:
            out[c] = band
        return out.astype(self.domain.np_dtype)

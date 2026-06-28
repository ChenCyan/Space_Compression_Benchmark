"""Tiled JPEG2000 codec with ThreadPoolExecutor for multi-threaded compression.

Splits the input (C, H, W) image into spatial tiles, compresses each tile
independently in parallel, then stitches results back during decode.

This achieves tile-level parallelism (independent codec instances per tile),
which scales better than OpenJPEG's internal code-block-level threading.
Works with any OpenJPEG version — no library upgrade needed.

Payload format::

    ┌──────────────────────────────────────────────────────────────┐
    │  Magic "JPTL"                               4 bytes          │
    │  C, H, W, tile_size                         uint32 × 4      │ header
    │  payload_len[0] ... payload_len[N-1]         uint32 × N      │
    ├──────────────────────────────────────────────────────────────┤
    │  tile_payload[0]                                             │ tiles
    │  tile_payload[1]                                             │
    │  ...                                                         │
    └──────────────────────────────────────────────────────────────┘

Tiles are in row-major order. Edge tiles may be smaller than tile_size.
"""

from __future__ import annotations

import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from benchmark.codecs.base import Codec, EncodeResult
from benchmark.codecs import jp2_util
from benchmark.domain import Domain, DEFAULT_DOMAIN

_MAGIC = b"JPTL"


def _tile_grid(H: int, W: int, tile_size: int) -> list[tuple[int, int, int, int]]:
    """Return list of (y_start, x_start, tile_h, tile_w) for all tiles, row-major."""
    tiles = []
    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            th = min(tile_size, H - y)
            tw = min(tile_size, W - x)
            tiles.append((y, x, th, tw))
    return tiles


def _encode_tile(args: tuple[np.ndarray, bool, float | None]) -> bytes:
    """Encode a single spatial tile → JP2 bytes.  (Picklable for ThreadPoolExecutor.)"""
    tile, lossless, cratio = args
    return jp2_util.encode_multicomponent(tile, lossless=lossless, cratio=cratio)


def _decode_tile(payload: bytes) -> np.ndarray:
    """Decode a single tile payload → (C, tile_h, tile_w).  (Picklable for ThreadPoolExecutor.)"""
    return jp2_util.decode_multicomponent(payload)


class JPEG2000TileCodec(Codec):
    """JPEG2000 with manual spatial tiling + ThreadPoolExecutor parallelism.

    Parameters
    ----------
    cratio : float
        Target compression ratio.  ``<= 1.0`` → lossless.
    tile_size : int
        Spatial tile size in pixels (square tiles).
    num_threads : int
        Number of threads for parallel tile encode / decode.
    domain : Domain
        DN domain for de-/normalisation.
    """

    device = "cpu"

    def __init__(
        self,
        cratio: float = 1.0,
        *,
        tile_size: int = 256,
        num_threads: int = 1,
        domain: Domain = DEFAULT_DOMAIN,
    ):
        super().__init__(domain)
        self.cratio = float(cratio)
        self.lossless = self.cratio <= 1.0
        self.tile_size = int(tile_size)
        self.num_threads = int(num_threads)
        self.family = "lossless" if self.lossless else "lossy"
        self.name = (
            f"jpeg2000_tiled_cr{self.cratio:g}_t{self.tile_size}_th{self.num_threads}"
        )

    def operating_point(self) -> str:
        return f"cr={self.cratio:g}_tiles={self.tile_size}_thr={self.num_threads}"

    # ------------------------------------------------------------------
    def encode(self, x_dn: np.ndarray) -> EncodeResult:
        C, H, W = x_dn.shape
        tiles = _tile_grid(H, W, self.tile_size)
        n_tiles = len(tiles)

        t0 = time.perf_counter()

        if self.num_threads <= 1 or n_tiles <= 1:
            # ---- single-threaded path (no executor overhead) ----
            payloads = []
            for y, x, th, tw in tiles:
                tile = x_dn[:, y:y + th, x:x + tw]
                payloads.append(
                    jp2_util.encode_multicomponent(
                        tile, lossless=self.lossless,
                        cratio=None if self.lossless else self.cratio,
                    )
                )
        else:
            # ---- multi-threaded path ----
            tasks = []
            for y, x, th, tw in tiles:
                tile = x_dn[:, y:y + th, x:x + tw].copy()  # copy so thread owns its data
                tasks.append((tile, self.lossless, None if self.lossless else self.cratio))
            payloads = [None] * n_tiles
            with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
                futures = {ex.submit(_encode_tile, t): idx for idx, t in enumerate(tasks)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    payloads[idx] = fut.result()

        enc_time = time.perf_counter() - t0

        # ---- pack payload ----
        header = bytearray()
        header += _MAGIC
        header += struct.pack("<IIII", C, H, W, self.tile_size)
        for pl in payloads:
            header += struct.pack("<I", len(pl))
        total_payload = bytes(header) + b"".join(payloads)

        meta = {
            "C": C, "H": H, "W": W,
            "tile_size": self.tile_size,
            "n_tiles": n_tiles,
            "enc_time_internal": enc_time,
        }
        return EncodeResult(payload=total_payload, meta=meta)

    # ------------------------------------------------------------------
    def decode(self, enc: EncodeResult) -> np.ndarray:
        blob = enc.payload
        assert blob[:4] == _MAGIC, f"Bad magic: {blob[:4]!r}"

        off = 4
        C, H, W, tile_size = struct.unpack_from("<IIII", blob, off)
        off += 16

        tiles = _tile_grid(H, W, tile_size)
        n_tiles = len(tiles)

        # Read payload lengths
        payload_lens = []
        for _ in range(n_tiles):
            (ln,) = struct.unpack_from("<I", blob, off)
            off += 4
            payload_lens.append(ln)

        # Slice out tile payloads
        tile_payloads = []
        for ln in payload_lens:
            tile_payloads.append(blob[off:off + ln])
            off += ln

        # Decode tiles (parallel if threaded)
        if self.num_threads <= 1 or n_tiles <= 1:
            tile_arrays = [_decode_tile(pl) for pl in tile_payloads]
        else:
            tile_arrays = [None] * n_tiles
            with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
                futures = {ex.submit(_decode_tile, pl): idx
                           for idx, pl in enumerate(tile_payloads)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    tile_arrays[idx] = fut.result()

        # Stitch
        out = np.zeros((C, H, W), dtype=self.domain.np_dtype)
        for (y, x, th, tw), arr in zip(tiles, tile_arrays):
            out[:, y:y + th, x:x + tw] = arr
        return out

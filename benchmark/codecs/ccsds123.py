"""CCSDS-123.0-B-1 / B-2 codec via NTNU verification model.

Thin adapter around the open-source NTNU CCSDS-123.0-B-2 High-Level Model
(https://github.com/NTNU-SmallSat-Lab/ccsds123_issue_2_verification_model).
MIT license, verified against CCSDS official test vectors.
"""

from __future__ import annotations

import importlib.util
import os
import struct
import sys
import tempfile

import numpy as np

from benchmark.codecs.base import Codec, EncodeResult
from benchmark.domain import Domain, DEFAULT_DOMAIN

_NTNU_PATHS = ["/opt/ccsds123_i2_hlm", "/data/cyl/space_compression/ccsds123_i2_hlm"]


def _load_ntnu_module(mod_name: str):
    """Load a module from the NTNU repo by absolute path (avoids import clash
    with our own ccsds123.py)."""
    for base in _NTNU_PATHS:
        pkg = os.path.join(base, "ccsds123_i2_hlm")
        mod_file = os.path.join(pkg, f"{mod_name}.py")
        if os.path.isfile(mod_file):
            if base not in sys.path:
                sys.path.insert(0, base)
            spec = importlib.util.spec_from_file_location(
                f"ccsds123_i2_hlm.{mod_name}", mod_file,
                submodule_search_locations=[pkg],
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        f"NTNU CCSDS-123 module '{mod_name}' not found. Tried: {_NTNU_PATHS}"
    )


class CCSDS123Codec(Codec):
    device = "cpu"

    def __init__(
        self,
        lossless: bool = True,
        pae_bound: int = 0,
        *,
        domain: Domain = DEFAULT_DOMAIN,
    ):
        super().__init__(domain)
        self.lossless = bool(lossless)
        self.pae_bound = int(pae_bound) if not self.lossless else 0
        self.family = "lossless" if self.lossless else "near-lossless"
        self.name = "ccsds123_lossless" if self.lossless else f"ccsds123_pae{self.pae_bound}"

    def operating_point(self) -> str:
        return "lossless" if self.lossless else f"PAE<={self.pae_bound}"

    def encode(self, x_dn: np.ndarray) -> EncodeResult:
        ccsds_mod = _load_ntnu_module("ccsds123")
        header_mod = _load_ntnu_module("header")
        CCSDS123 = ccsds_mod.CCSDS123
        Header = header_mod.Header
        QuantizerFidelityControlMethod = header_mod.QuantizerFidelityControlMethod

        C, H, W = x_dn.shape

        # Write temp raw file (BSQ, uint16 big-endian).
        raw = np.asarray(x_dn, dtype=self.domain.np_dtype)
        raw_be = raw.astype(">u2")
        # NTNU parses filename with regex x(.+)x → must be name-u16be-CxHxW.raw
        raw_name = f"bench-u16be-{C}x{H}x{W}.raw"
        raw_path = os.path.join("/tmp", raw_name)
        raw_be.tofile(raw_path)

        ccsds = CCSDS123(raw_path)
        ccsds.set_header()          # creates header with correct dims from raw_path
        ccsds.output_folder = tempfile.mkdtemp(prefix="ccsds123_out_", dir="/tmp")

        # Now override compression mode on the parsed header.
        hdr = ccsds.header
        if self.lossless:
            hdr.quantizer_fidelity_control_method = QuantizerFidelityControlMethod.LOSSLESS
            hdr.absolute_error_limit_value = 0
            hdr.relative_error_limit_value = 0
            hdr.absolute_error_limit_bit_depth = 0
            hdr.relative_error_limit_bit_depth = 0
        else:
            hdr.quantizer_fidelity_control_method = QuantizerFidelityControlMethod.ABSOLUTE_ONLY
            hdr.absolute_error_limit_value = self.pae_bound
            hdr.absolute_error_limit_bit_depth = max(1, self.pae_bound.bit_length())

        # compress_image() recreates Header from image_file; monkey-patch to
        # return our configured header instead.
        _orig_header = header_mod.Header
        header_mod.Header = lambda *a, **kw: hdr
        try:
            ccsds.compress_image()
        finally:
            header_mod.Header = _orig_header

        try:
            bitstream_path = os.path.join(ccsds.output_folder, "z-output-bitstream.bin")
            num_bits = os.path.getsize(bitstream_path) * 8

            rec = ccsds.predictor.sample_representative  # (H, W, C)
            rec = np.rint(rec).clip(0, self.domain.dn_peak).astype(self.domain.np_dtype)
            rec = rec.transpose(2, 0, 1)

            meta = {"C": C, "H": H, "W": W, "num_bits": num_bits}
            payload = struct.pack("<IIIQ", C, H, W, num_bits)
            return EncodeResult(payload=payload, meta=meta, reconstruction=rec)

        finally:
            for p in [raw_path, bitstream_path]:
                if os.path.exists(p):
                    os.remove(p)
            od = ccsds.output_folder
            if os.path.isdir(od):
                import shutil
                shutil.rmtree(od, ignore_errors=True)

    def decode(self, enc: EncodeResult) -> np.ndarray:
        if enc.reconstruction is not None:
            return enc.reconstruction
        raise NotImplementedError("CCSDS123Codec: standalone decode not available.")

"""Unified codec interface.

Every benchmarked method -- learned (HyCASS) or classical (CCSDS-123, KLT+DWT,
JPEG2000) -- implements this interface so the runner can treat them
identically and measure CR / throughput / error the same way.

Contract:
    encode(x_dn) -> bytes            # the actual compressed payload
    decode(payload, meta) -> x_dn    # reconstruction in the SAME DN domain

`encode` returns a `bytes` object whose length (in bits) is the real coded
size -- this is what compression ratio is computed from, NOT a theoretical
estimate. For codecs that need shape/side-info to decode, return it via
`encode_with_meta`, which packs everything decode needs.

This avoids the trap in the original repo where CR is a fixed architectural
constant (bpppc = 32 / CR); here CR is measured from actual payload bytes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmark.domain import Domain, DEFAULT_DOMAIN


@dataclass
class EncodeResult:
    """Output of an encode: the payload plus everything decode needs."""

    payload: bytes
    meta: dict = field(default_factory=dict)
    reconstruction: np.ndarray | None = None
    """Optional reconstruction produced during encode (used by verification-model
    codecs like CCSDS-123 that lack a standalone decoder)."""

    @property
    def num_bits(self) -> int:
        return len(self.payload) * 8


class Codec(abc.ABC):
    """Base class for all compression methods in the benchmark.

    Subclasses must set:
        name:    short identifier used in result tables
        family:  one of {"lossless", "near-lossless", "lossy"}
        device:  "cpu" or "gpu" (for honest throughput reporting)
    and implement encode()/decode().
    """

    name: str = "base"
    family: str = "lossy"
    device: str = "cpu"

    def __init__(self, domain: Domain = DEFAULT_DOMAIN):
        self.domain = domain

    @abc.abstractmethod
    def encode(self, x_dn: np.ndarray) -> EncodeResult:
        """Compress a (C,H,W) integer DN array. Returns payload + meta."""

    @abc.abstractmethod
    def decode(self, enc: EncodeResult) -> np.ndarray:
        """Reconstruct a (C,H,W) integer DN array from an EncodeResult."""

    # ---- convenience -----------------------------------------------------

    def operating_point(self) -> str:
        """Human-readable label for this configuration (e.g. 'CR=8' or 'PAE<=4').

        Override in subclasses that sweep a parameter.
        """
        return self.name

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} family={self.family!r} device={self.device!r}>"

"""Codec registry."""

from benchmark.codecs.base import Codec, EncodeResult

from benchmark.codecs.jpeg2000 import JPEG2000Codec
from benchmark.codecs.klt_dwt import KLTDWTCodec
from benchmark.codecs.ccsds123_py_codec import CCSDS123PyCodec
from benchmark.codecs.jpegls_codec import JPEGLSCodec
from benchmark.codecs.generic_codecs import LZ4Codec, ZlibCodec

# Old NTNU-based codec (broken, kept for reference).
try:
    from benchmark.codecs.ccsds123 import CCSDS123Codec
except ImportError:
    CCSDS123Codec = None

# HyCASS requires torch + the project models accessible on PYTHONPATH.
try:
    from benchmark.codecs.hycass_codec import HyCASSCodec
except ImportError:
    HyCASSCodec = None


__all__ = [
    "Codec", "EncodeResult",
    "JPEG2000Codec", "KLTDWTCodec", "CCSDS123PyCodec",
    "JPEGLSCodec", "LZ4Codec", "ZlibCodec",
    "CCSDS123Codec", "HyCASSCodec",
]

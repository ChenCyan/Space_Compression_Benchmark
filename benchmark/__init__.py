"""HSI compression benchmark package.

Provides a unified harness to compare multi-band (hyperspectral) compression
methods on equal footing:
  - lossless:      CCSDS-123.0-B-1, KLT+DWT (reversible)
  - near-lossless: CCSDS-123.0-B-2 (PAE-bounded), KLT+DWT (lossy / rate-controlled)
  - lossy (learned): HyCASS

All codecs share one interface (codecs.base.Codec) and are measured with one
set of metrics in one unified data domain, so compression-ratio / throughput /
error numbers are directly comparable.
"""

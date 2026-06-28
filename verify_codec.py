from benchmark.runner import _parse_codec_spec
from benchmark.domain import DEFAULT_DOMAIN
for spec in ["jpeg2000_lossless", "jpeg2000_lossless_th4", "jpeg2000_cr8", "jpeg2000_cr8_th4", "jpeg2000_tiled_cr4_t40_th4"]:
    c = _parse_codec_spec(spec, DEFAULT_DOMAIN)
    print("{} -> {} | {}".format(spec, c.name, c.operating_point()))

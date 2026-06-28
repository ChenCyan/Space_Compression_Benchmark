import numpy as np
from benchmark.domain import DEFAULT_DOMAIN
from datasets.berlinurbangradient import BerlinUrbanGradient
import lz4.frame, zlib

ds = BerlinUrbanGradient("datasets/berlin-urban-gradient/", split="test")
x_dn = DEFAULT_DOMAIN.to_dn(ds[0].numpy())
raw = x_dn.astype(np.uint16).tobytes()
print("raw:", len(raw), "bytes (111x80x80x2)")

lz4_c = lz4.frame.compress(raw, compression_level=1)
zlib_c = zlib.compress(raw, level=6)
print("LZ4: %d bytes CR=%.2f" % (len(lz4_c), len(raw)/len(lz4_c)))
print("zlib: %d bytes CR=%.2f" % (len(zlib_c), len(raw)/len(zlib_c)))

arr = np.frombuffer(raw, dtype=np.uint8)
unique = np.unique(arr)
print("byte unique: %d/256 (%.1f%%)" % (len(unique), len(unique)/256*100))
same = (arr[1:] == arr[:-1]).sum()
print("adjacent same: %d/%d (%.2f%%)" % (same, len(arr)-1, same/(len(arr)-1)*100))

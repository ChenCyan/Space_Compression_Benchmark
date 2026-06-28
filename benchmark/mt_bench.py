"""Multi-threading throughput comparison for classic codecs."""
import os, sys, time
import numpy as np

sys.path.insert(0, "/data/cyl/space_compression/hycass")

from benchmark.codecs.jpegls_codec import JPEGLSCodec
from benchmark.codecs.ccsds123_py_codec import CCSDS123PyCodec
from benchmark.codecs.generic_codecs import ZlibCodec
from benchmark.domain import Domain

DATA_DIR = "/data/cyl/space_compression/hycass/datasets/berlin-urban-gradient"
PATCHES_DIR = os.path.join(DATA_DIR, "patches")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

domain = Domain.from_data_dir(DATA_DIR)

with open(os.path.join(SPLITS_DIR, "test.csv")) as f:
    test_files = [line.strip() for line in f if line.strip()]

samples = [np.load(os.path.join(PATCHES_DIR, fn)) for fn in test_files[:5]]
# Convert to DN domain (critical: raw is float32, codecs expect uint16 DN)
samples_dn = [domain.to_dn(s) for s in samples]
raw_per = samples_dn[0].nbytes / 1024**2
total_raw = sum(s.nbytes for s in samples_dn) / 1024**2
print(f"Loaded {len(samples_dn)} samples, shape={samples_dn[0].shape}, dtype={samples_dn[0].dtype}, {total_raw:.1f} MB\n")

def bench(name, create_codec, threads_list, warmup=2, reps=5):
    results = {}
    for nt in threads_list:
        try:
            codec = create_codec(nt)
            for _ in range(warmup):
                codec.decode(codec.encode(samples_dn[0]))
            enc_t, dec_t = [], []
            for s in samples_dn[:reps]:
                t0 = time.perf_counter()
                enc = codec.encode(s)
                t1 = time.perf_counter()
                codec.decode(enc)
                t2 = time.perf_counter()
                enc_t.append(t1 - t0)
                dec_t.append(t2 - t1)
            # CR: re-encode to avoid timing contamination
            compressed = [len(codec.encode(s).payload) for s in samples_dn[:reps]]
            avg_raw = sum(s.nbytes for s in samples_dn[:reps]) / reps / 1024**2
            avg_comp = sum(compressed) / reps / 1024**2
            results[nt] = {
                "enc": round(avg_raw / np.mean(enc_t), 1),
                "dec": round(avg_raw / np.mean(dec_t), 1),
                "cr": round(avg_raw * 1024**2 / (sum(compressed) / reps), 2),
            }
        except Exception as e:
            print(f"  ERROR t={nt}: {e}")
            import traceback; traceback.print_exc()
            results[nt] = {"enc": "ERR", "dec": "ERR", "cr": "ERR"}
    return results

print("=== JPEG-LS (lossless) ===")
jls = bench("JPEG-LS", lambda nt: JPEGLSCodec(lossless=True, num_threads=nt), [1, 2, 4, 8])
for nt, r in jls.items(): print(f"  t={nt}: enc={r['enc']} MB/s  dec={r['dec']} MB/s  CR={r['cr']}")

print("\n=== CCSDS-123 (lossless) ===")
ccsds = bench("CCSDS-123", lambda nt: CCSDS123PyCodec(lossless=True, num_threads=nt), [1, 2, 4, 8])
for nt, r in ccsds.items(): print(f"  t={nt}: enc={r['enc']} MB/s  dec={r['dec']} MB/s  CR={r['cr']}")

print("\n=== zlib (level=6) ===")
zr = bench("zlib", lambda nt: ZlibCodec(level=6, num_threads=nt), [1, 4, 8])
for nt, r in zr.items(): print(f"  t={nt}: enc={r['enc']} MB/s  dec={r['dec']} MB/s  CR={r['cr']}")

print("\n=== SPEEDUP SUMMARY ===")
print("| Method | 1-thr | 8-thr | Speedup | CR(1) | CR(8) | Bottleneck |")
print("|:---|---:|---:|---:|---:|---:|:---|")
for name, res in [("JPEG-LS", jls), ("CCSDS-123", ccsds), ("zlib", zr)]:
    e1 = res[1]["enc"]
    k8 = 8 if 8 in res else max(res.keys())
    e8 = res[k8]["enc"]
    c1, c8 = res[1]["cr"], res[k8]["cr"]
    if isinstance(e1, str):
        print(f"| {name} | ERR | ERR | - | - | - | - |")
    else:
        sp = e8/e1
        if sp > 1.05: why = "波段独立, ThreadPool有效"
        elif sp < 0.95: why = "单波段<1ms, 线程开销主导"
        else: why = "瓶颈不在Python层"
        print(f"| {name} | {e1} | {e8} | {sp:.1f}x | {c1} | {c8} | {why} |")

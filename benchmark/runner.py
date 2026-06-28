"""Benchmark runner with optional fork-parallelism for CPU codecs."""

from __future__ import annotations
import csv, os, sys, time, numpy as np
from typing import Sequence
from benchmark.codecs.base import Codec, EncodeResult
from benchmark.codecs import (JPEG2000Codec, KLTDWTCodec, CCSDS123PyCodec,
                              JPEGLSCodec, LZ4Codec, ZlibCodec, HyCASSCodec)
from benchmark.codecs.jpeg2000_tiled import JPEG2000TileCodec
from benchmark.domain import Domain, DEFAULT_DOMAIN
from benchmark.metrics_unified import all_metrics


def _parse_codec_spec(spec: str, domain: Domain) -> Codec:
    parts = spec.split("_")
    if parts[0] == "jpeg2000":
        # Optional trailing _th{N} for OpenJPEG >= 2.4.0 code-block threading.
        # Applied only to the non-tiled (standard) codec.
        if parts[1] == "lossless":
            th = int(parts[2].replace("th", "")) if len(parts) > 2 and parts[2].startswith("th") else 1
            return JPEG2000Codec(cratio=1, num_threads=th, domain=domain)
        if parts[1] == "tiled":
            # jpeg2000_tiled_cr{CR}_t{TILE_SIZE}_th{NUM_THREADS}
            cr = float(parts[2].replace("cr", ""))
            tile_sz = int(parts[3].replace("t", ""))
            n_threads = int(parts[4].replace("th", ""))
            return JPEG2000TileCodec(cratio=cr, tile_size=tile_sz, num_threads=n_threads, domain=domain)
        cr = float(parts[1].replace("cr", ""))
        th = int(parts[2].replace("th", "")) if len(parts) > 2 and parts[2].startswith("th") else 1
        return JPEG2000Codec(cratio=cr, num_threads=th, domain=domain)
    if parts[0] == "klt" and parts[1] == "dwt":
        nc = int(parts[2].replace("nc", ""))
        cr = float(parts[3].replace("cr", ""))
        return KLTDWTCodec(n_components=nc, cratio=cr, domain=domain)
    if parts[0] == "ccsds123":
        if parts[1] == "lossless": return CCSDS123PyCodec(lossless=True, domain=domain)
        pae = int(parts[1].replace("pae", ""))
        return CCSDS123PyCodec(lossless=False, pae_bound=pae, domain=domain)
    if parts[0] == "jpegls":
        if parts[1] == "lossless": return JPEGLSCodec(lossless=True, domain=domain)
        ne = int(parts[1].replace("ne", ""))
        return JPEGLSCodec(lossless=False, lossy_error=ne, domain=domain)
    if parts[0] == "lz4":
        return LZ4Codec(domain=domain)
    if parts[0] == "zlib":
        return ZlibCodec(domain=domain)
    LEARNED = ["hycass", "cae1d", "cae3d", "sscnet", "hycot"]
    if any(parts[0] == p for p in LEARNED) and HyCASSCodec is not None:
        ckpt = os.environ.get("HYCASS_CHECKPOINT",
               f"pretrained/berlin-urban-gradient/{spec}.pth.tar")
        if not os.path.exists(ckpt): ckpt = None
        return HyCASSCodec(model_name=spec, checkpoint_path=ckpt, domain=domain)
    raise ValueError(f"Unknown codec spec: {spec!r}")


def measure_one(codec, x_dn, warmup=False):
    domain = codec.domain
    t0 = time.perf_counter(); enc = codec.encode(x_dn); t1 = time.perf_counter()
    if enc.reconstruction is not None: x_hat, t2 = enc.reconstruction, t1
    else: x_hat = codec.decode(enc); t2 = time.perf_counter()
    enc_time, dec_time = t1 - t0, t2 - t1
    raw_bits = domain.raw_bits(x_dn.shape)
    cr = raw_bits / max(1, enc.num_bits)
    bpppc = enc.num_bits / max(1, int(np.prod(x_dn.shape)))
    m = all_metrics(x_dn, x_hat, peak=float(domain.dn_peak))
    m["cr"], m["bpppc"] = cr, bpppc
    m["enc_time_s"], m["dec_time_s"] = enc_time, dec_time
    m["num_samples"] = 1
    raw_mb = int(np.prod(x_dn.shape)) * domain.bitdepth_native / 8 / 1e6
    m["throughput_mbs"] = raw_mb / max(enc_time + dec_time, 1e-9)
    return m


# ---- fork-parallel worker (must be top-level for pickling) ----

def _fork_worker(args):
    """Worker called by multiprocessing Pool. Reconstructs codec from spec dict."""
    spec, sample_float, warmup = args
    cls = spec["class"]
    domain = DEFAULT_DOMAIN
    if cls == "JPEG2000Codec": codec = JPEG2000Codec(cratio=spec.get("cratio",1.0), domain=domain)
    elif cls == "KLTDWTCodec": codec = KLTDWTCodec(n_components=spec.get("nc",111), cratio=spec.get("cratio",1.0), domain=domain)
    elif cls == "CCSDS123PyCodec": codec = CCSDS123PyCodec(lossless=spec.get("lossless",True), pae_bound=spec.get("pae",0), domain=domain)
    else: raise ValueError(cls)
    x_dn = domain.to_dn(np.asarray(sample_float, dtype=np.float32))
    return measure_one(codec, x_dn, warmup=warmup)


def _spec_from_codec(codec):
    s = {"class": type(codec).__name__}
    for attr in ["cratio","n_components","lossless","pae_bound"]:
        if hasattr(codec, attr): s[attr] = getattr(codec, attr)
    return s


def run_benchmark(codecs, dataset, *, max_samples=0, warmup_samples=2,
                  domain=DEFAULT_DOMAIN, verbose=True, workers=0):
    rows = []
    n_total = len(dataset)
    n_limit = min(max_samples, n_total) if max_samples > 0 else n_total

    for codec in codecs:
        if verbose: print("\n" + "="*60 + "\n  %s [%s]\n" % (codec.name, codec.family) + "="*60)
        # Fork unsafe with glymur/OpenJPEG (used by KLT+DWT, JPEG2000).
        # Only CCSDS-123 is fork-safe (pure Python + numba).
        fork_ok = "ccsds123" in codec.name
        use_fork = (workers > 0 and codec.device == "cpu" and fork_ok
                    and n_limit > warmup_samples + 1)
        spec = _spec_from_codec(codec) if use_fork else None

        if use_fork:
            import multiprocessing as mp
            samples = [dataset[i].numpy().astype(np.float32) for i in range(n_limit)]
            # Warmup in parent
            if warmup_samples > 0 and verbose:
                print("  warmup (JIT) ...", end=" ", flush=True)
                x_dn = domain.to_dn(np.asarray(samples[0]))
                measure_one(codec, x_dn, warmup=True)
            work = [(spec, samples[i], False) for i in range(warmup_samples, n_limit)]
            n_w = min(mp.cpu_count(), len(work), workers if workers else 8)
            if verbose: print("%d samples, %d workers ..." % (len(work), n_w), end="", flush=True)
            try: mp.set_start_method("fork", force=True)
            except RuntimeError: pass
            acc, count = {}, 0
            with mp.Pool(n_w) as pool:
                for row in pool.map(_fork_worker, work):
                    count += 1
                    for k, v in row.items(): acc[k] = acc.get(k,0.0) + v
        else:
            acc, count = {}, 0
            for idx in range(n_limit):
                x_dn = domain.to_dn(dataset[idx].numpy())
                if idx < warmup_samples:
                    if verbose and idx == 0: print("  warming up ...", end=" ", flush=True)
                    measure_one(codec, x_dn, warmup=True)
                    continue
                row = measure_one(codec, x_dn)
                count += 1
                for k, v in row.items(): acc[k] = acc.get(k,0.0) + v

        if count == 0: continue
        avg = {"method":codec.name,"family":codec.family,"operating_point":codec.operating_point(),"device":codec.device}
        for k, v in acc.items():
            avg[k] = v / count if k != "psnr" or v != float("inf") else float("inf")
        avg["num_samples_measured"] = count
        rows.append(avg)
        if verbose:
            p = avg.get("psnr",0); ps = "%.1f" % p if np.isfinite(p) else "inf"
            print("\r  done (%d samples). CR=%.1f PSNR=%s dB PAE=%.0f DN thru=%.1f MB/s" %
                  (count, avg.get("cr",0), ps, avg.get("pae",0), avg.get("throughput_mbs",0)))
    return rows


_COLS = ["method","family","operating_point","device","cr","bpppc","mse","mae","psnr","pae","sa","enc_time_s","dec_time_s","throughput_mbs","num_samples_measured"]

def save_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print("\nResults written to %s (%d rows)" % (path, len(rows)))


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
    from datasets.berlinurbangradient import BerlinUrbanGradient
    from datasets.hyspecnet11k import HySpecNet11k
    from datasets.npy_patch_dataset import NumpyPatchDataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset","-d",required=True)
    ap.add_argument("--dataset-type",default="berlin",
                    choices=["berlin","hyspecnet11k","generic"])
    ap.add_argument("--codecs",nargs="+",required=True)
    ap.add_argument("--output","-o",default="results/benchmark.csv")
    ap.add_argument("--max-samples",type=int,default=0)
    ap.add_argument("--warmup",type=int,default=2)
    ap.add_argument("--workers","-w",type=int,default=0,help="Parallel workers for CPU (0=auto)")
    args = ap.parse_args()
    if args.workers == 0: args.workers = int(os.environ.get("BENCH_WORKERS", "0"))
    print("Loading %s from %s ..." % (args.dataset_type, args.dataset))
    if args.dataset_type == "generic":
        ds = NumpyPatchDataset(args.dataset, split="test")
    elif args.dataset_type == "berlin":
        ds = BerlinUrbanGradient(args.dataset, split="test")
    else:
        ds = HySpecNet11k(args.dataset, split="test", mode="easy")
    print("Dataset size: %d samples" % len(ds))
    domain = Domain.from_data_dir(args.dataset) if args.dataset_type == "generic" else DEFAULT_DOMAIN
    cl = [_parse_codec_spec(s, domain) for s in args.codecs]
    for c in cl: print("  %s [%s] dev=%s" % (c.name, c.family, c.device))
    rows = run_benchmark(cl, ds, max_samples=args.max_samples, warmup_samples=args.warmup, workers=args.workers)
    save_csv(rows, args.output)

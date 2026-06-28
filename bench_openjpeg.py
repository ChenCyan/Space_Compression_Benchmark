import glymur, numpy as np, time
from benchmark.codecs import jp2_util

rng = np.random.default_rng(42)
cube = rng.integers(0, 10000, (111, 80, 80), dtype=np.uint16)

for n in [1, 2, 4, 6, 8]:
    glymur.set_option('lib.num_threads', n)
    jp2_util.encode_multicomponent(cube, lossless=True)  # warmup
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        jp2_util.encode_multicomponent(cube, lossless=True)
        times.append(time.perf_counter() - t0)
    avg = np.mean(times)
    print("enc thr={}: {:.1f}ms  thru={:.1f}MB/s".format(n, avg*1000, cube.nbytes/1e6/avg))

pl = jp2_util.encode_multicomponent(cube, lossless=True)
for n in [1, 2, 4, 6, 8]:
    glymur.set_option('lib.num_threads', n)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        jp2_util.decode_multicomponent(pl)
        times.append(time.perf_counter() - t0)
    avg = np.mean(times)
    print("dec thr={}: {:.1f}ms  thru={:.1f}MB/s".format(n, avg*1000, cube.nbytes/1e6/avg))

glymur.set_option('lib.num_threads', 1)

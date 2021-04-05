import time

import numpy as np

import cutde
import cutde.gpu

for i in range(3):
    n_obs = 5000
    n_src = 5000
    pts = cutde.gpu.to_gpu(np.random.rand(n_obs, 3).astype(np.float32), np.float32)
    tris = cutde.gpu.to_gpu(np.random.rand(n_src, 3, 3).astype(np.float32), np.float32)
    slips = cutde.gpu.to_gpu(np.random.rand(n_src, 3).astype(np.float32), np.float32)

    start = time.time()
    strain = cutde.strain_all_pairs(pts, tris, slips, 0.25)

    pairs = n_obs * n_src / 1e6
    rt = time.time() - start
    pairs_per_sec = pairs / rt
    print(pairs, "million took", rt, " -- ", pairs_per_sec, "pairs/sec")

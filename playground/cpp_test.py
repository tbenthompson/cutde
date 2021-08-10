import time

import cppimport.import_hook  # noqa: F401
import numpy as np
import simple_module

import cutde.fullspace as FS

n = 11


def get_pt_grid():
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    z = np.linspace(-3, 3, n)
    X, Z, Y = np.meshgrid(x, z, y)
    test_pts = np.hstack([arr.flatten()[:, np.newaxis] for arr in [X, Y, Z]])
    return test_pts


tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
slip = np.array([1.0, 0, 0])
obs_pts = get_pt_grid()
N_test = obs_pts.shape[0]
tris = np.array([tri] * N_test, dtype=np.float64)
slips = np.array([slip] * N_test)
nu = 0.25

fnc_name = "pairs_disp_fs"
vec_dim = 3
# if tris.shape[0] != obs_pts.shape[0]:
#     raise ValueError("There must be one input observation point per triangle.")

# check_inputs(obs_pts, tris, slips)
# float_type, (obs_pts, tris, slips) = solve_types(obs_pts, tris, slips)

n = obs_pts.shape[0]
block_size = 128
n_blocks = int(np.ceil(n / block_size))
# gpu_config = dict(block_size=block_size, float_type=cluda.np_to_c_type(float_type))
# module = cluda.load_gpu("pairs.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

gpu_results = np.empty(n * vec_dim)
gpu_obs_pts = obs_pts.flatten()
gpu_tris = tris.flatten()
gpu_slips = slips.flatten()

# getattr(module, "pairs_" + fnc_name)(
#     gpu_results,
#     np.int32(n),
#     gpu_obs_pts,
#     gpu_tris,
#     gpu_slips,
#     float_type(nu),
#     grid=(n_blocks, 1, 1),
#     block=(block_size, 1, 1),
# )

# out = gpu_results.get().reshape((n, vec_dim))
# return out
print(gpu_results)

for i in range(20):
    start = time.time()
    simple_module.pairs_disp_fs(
        gpu_results,
        n,
        gpu_obs_pts,
        gpu_tris,
        gpu_slips,
        nu,
        (n_blocks, 1, 1),
        (block_size, 1, 1),
    )
    print("C++", time.time() - start)
    new_out = gpu_results.reshape((n, vec_dim))
    start = time.time()
    cuda_out = FS.disp(obs_pts, tris, slips, nu)
    print("CUDA", time.time() - start)
print(np.where(new_out - cuda_out > 1e-15))

# import cppimport.import_hook
# import test2
# print(test2.square(2))

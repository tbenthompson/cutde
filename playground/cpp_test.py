import logging
import sys
import time

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root_logger.addHandler(handler)

import cppimport  # .import_hook  # noqa: F401
import numpy as np

simple_module = cppimport.imp_from_filepath("playground/simple_module.cpp")

import cutde.fullspace as FS
import cutde.halfspace as HS

n = 11


def get_pt_grid():
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    z = np.linspace(-6, 0, n)
    X, Z, Y = np.meshgrid(x, z, y)
    test_pts = np.hstack([arr.flatten()[:, np.newaxis] for arr in [X, Y, Z]])
    return test_pts


tri = np.array([[0, 0, -0.5], [1, 0, -0.5], [0, 1, -0.5]])
slip = np.array([1.0, 0, 0])
obs_pts = get_pt_grid()
N_test = obs_pts.shape[0]
tris = np.array([tri] * N_test, dtype=np.float64)
slips = np.array([slip] * N_test)
nu = 0.25

cmp_fnc = HS.strain
fnc_name = "pairs_strain_hs"
vec_dim = 6
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
gpu_obs_pts = obs_pts.ravel()
gpu_tris = tris.ravel()
gpu_slips = slips.ravel()

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

for i in range(2):
    start = time.time()
    getattr(simple_module, fnc_name)(
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
    cuda_out = cmp_fnc(obs_pts, tris, slips, nu)
    print("CUDA", time.time() - start)
print(new_out)
print(cuda_out)
print(np.where(new_out - cuda_out > 1e-13))

# import cppimport.import_hook
# import test2
# print(test2.square(2))

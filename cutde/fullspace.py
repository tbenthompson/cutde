import numpy as np

import cluda
import cutde
from cutde.TDdispFS import TDdispFS

def py_disp(obs_pt, tri, slip, nu):
    return TDdispFS(obs_pt, tri, slip, nu)

def call_clu(obs_pts, tris, slips, nu, fnc_name, out_dim):
    assert(obs_pts.shape[1] == 3)
    assert(tris.shape[0] == obs_pts.shape[0])
    assert(tris.shape[1] == 3)
    assert(tris.shape[2] == 3)
    assert(slips.shape[0] == obs_pts.shape[0])
    assert(slips.shape[1] == 3)

    n = obs_pts.shape[0]
    block_size = 128
    n_blocks = int(np.ceil(n / block_size))
    float_type = np.float32
    gpu_config = dict(
        block_size = block_size,
        float_type = cluda.np_to_c_type(float_type)
    )
    module = cluda.load_gpu(
        'fullspace.cu', tmpl_args = gpu_config,
        tmpl_dir = cutde.source_dir
    )

    gpu_results = cluda.empty_gpu(n * out_dim, float_type)
    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_tris = cluda.to_gpu(tris, float_type)
    gpu_slips = cluda.to_gpu(slips, float_type)

    getattr(module, fnc_name)(
        gpu_results, np.int32(n), gpu_obs_pts, gpu_tris, gpu_slips, float_type(nu),
        grid = (n_blocks, 1, 1), block = (block_size, 1, 1)
    )
    out = gpu_results.get().reshape((n, out_dim))
    return out

def call_clu_all_pairs(obs_pts, src_tris, slips, nu, fnc_name, out_dim):
    assert(obs_pts.shape[1] == 3)
    assert(src_tris.shape[1] == 3)
    assert(src_tris.shape[2] == 3)
    assert(slips.shape[0] == src_tris.shape[0])
    assert(slips.shape[1] == 3)

    n_obs = obs_pts.shape[0]
    n_src = src_tris.shape[0]
    block_size = 8
    n_obs_blocks = int(np.ceil(n_obs / block_size))
    n_src_blocks = int(np.ceil(n_src / block_size))
    float_type = np.float32
    gpu_config = dict(
        block_size = block_size,
        float_type = cluda.np_to_c_type(float_type)
    )
    module = cluda.load_gpu(
        'fullspace.cu', tmpl_args = gpu_config,
        tmpl_dir = cutde.source_dir
    )

    gpu_results = cluda.empty_gpu(n_obs * n_src * out_dim, float_type)
    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_src_tris = cluda.to_gpu(src_tris, float_type)
    gpu_slips = cluda.to_gpu(slips, float_type)

    getattr(module, fnc_name + '_all_pairs')(
        gpu_results, np.int32(n_obs), np.int32(n_src),
        gpu_obs_pts, gpu_src_tris, gpu_slips, float_type(nu),
        grid = (n_obs_blocks, n_src_blocks, 1), block = (block_size, block_size, 1)
    )
    out = gpu_results.get().reshape((n_obs, n_src, out_dim))
    return out

def clu_disp(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, 'disp_fullspace', 3)

def clu_strain(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, 'strain_fullspace', 6)

def clu_disp_all_pairs(obs_pts, tris, slips, nu):
    return call_clu_all_pairs(obs_pts, tris, slips, nu, 'disp_fullspace', 3)

def clu_strain_all_pairs(obs_pts, tris, slips, nu):
    return call_clu_all_pairs(obs_pts, tris, slips, nu, 'strain_fullspace', 6)

def strain_to_stress(strain, mu, nu):
    lam = 2 * mu * nu / (1 - 2 * nu)
    trace = np.sum(strain[:,:3], axis = 1)
    stress = np.empty_like(strain)
    stress[:,:3] = 2 * mu * strain[:,:3] + lam * trace[:,np.newaxis]
    stress[:,3:] = 2 * mu * strain[:,3:]
    return stress

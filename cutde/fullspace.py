import os
import warnings

import numpy as np

import cutde.gpu as cluda

from .TDdispFS import TDdispFS

source_dir = os.path.dirname(os.path.realpath(__file__))


def py_disp(obs_pt, tri, slip, nu):
    return TDdispFS(obs_pt, tri, slip, nu)


def solve_types(obs_pts, tris, slips):
    floats = [np.float32, np.float64]
    float_type = None
    out_arrs = []
    for name, arr in [("obs_pts", obs_pts), ("tris", tris), ("slips", slips)]:

        if float_type is None:
            float_type = arr.dtype

        if arr.dtype not in floats:
            __import__("ipdb").set_trace()
            raise ValueError(
                f"The {name} input array has type {arr.dtype} but must have"
                " a dtype of either np.float32 or np.float64"
            )

        if arr.dtype != float_type:
            warnings.warn(
                f"The {name} input array has type {arr.dtype} but obs_pts"
                " has dtype {float_type}. Converting {name} to {float_type}."
                " This may be expensive."
            )
            out_arrs.append(arr.astype(float_type))
        elif arr.flags.f_contiguous:
            warnings.warn(
                f"The {name} input array has Fortran ordering. "
                "Converting to C ordering. This may be expensive."
            )
            out_arrs.append(np.ascontiguousarray(arr))
        else:
            out_arrs.append(arr)

    return float_type.type, out_arrs


def check_inputs(obs_pts, tris, slips):
    if obs_pts.shape[1] != 3:
        raise ValueError(
            "The second dimension of the obs_pts array must be 3 because the "
            "observation points should be locations in three-dimensional space."
        )
    if tris.shape[1] != 3:
        raise ValueError(
            "The second dimension of the tris array must be 3 because there must be "
            "three vertices per triangle."
        )
    if tris.shape[2] != 3:
        raise ValueError(
            "The third dimension of the tris array must be 3 because the triangle "
            "vertices should be locations in three-dimensional space."
        )
    if slips.shape[0] != tris.shape[0]:
        raise ValueError(
            "The number of input slip vectors must be equal to the number of input"
            " triangles."
        )
    if slips.shape[1] != 3:
        raise ValueError(
            "The second dimension of the slips array must be 3 because each row "
            "should be a vector in the TDE coordinate system (strike-slip, dip-slip,"
            " tensile-slip)."
        )


def call_clu(obs_pts, tris, slips, nu, fnc_name, out_dim):
    if tris.shape[0] != obs_pts.shape[0]:
        raise ValueError("There must be one input observation point per triangle.")

    check_inputs(obs_pts, tris, slips)
    float_type, (obs_pts, tris, slips) = solve_types(obs_pts, tris, slips)

    n = obs_pts.shape[0]
    block_size = 128
    n_blocks = int(np.ceil(n / block_size))
    gpu_config = dict(block_size=block_size, float_type=cluda.np_to_c_type(float_type))
    module = cluda.load_gpu("fullspace.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = cluda.empty_gpu(n * out_dim, float_type)
    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_tris = cluda.to_gpu(tris, float_type)
    gpu_slips = cluda.to_gpu(slips, float_type)

    getattr(module, fnc_name)(
        gpu_results,
        np.int32(n),
        gpu_obs_pts,
        gpu_tris,
        gpu_slips,
        float_type(nu),
        grid=(n_blocks, 1, 1),
        block=(block_size, 1, 1),
    )
    out = gpu_results.get().reshape((n, out_dim))
    return out


def call_clu_all_pairs(obs_pts, tris, slips, nu, fnc_name, out_dim):
    check_inputs(obs_pts, tris, slips)
    float_type, (obs_pts, tris, slips) = solve_types(obs_pts, tris, slips)

    n_obs = obs_pts.shape[0]
    n_src = tris.shape[0]
    block_size = 256
    n_obs_blocks = int(np.ceil(n_obs / block_size))
    n_src_blocks = int(np.ceil(n_src / 1))
    gpu_config = dict(block_size=block_size, float_type=cluda.np_to_c_type(float_type))
    module = cluda.load_gpu("fullspace.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = cluda.empty_gpu(n_obs * n_src * out_dim, float_type)
    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_tris = cluda.to_gpu(tris, float_type)
    gpu_slips = cluda.to_gpu(slips, float_type)

    getattr(module, fnc_name + "_all_pairs")(
        gpu_results,
        np.int32(n_obs),
        np.int32(n_src),
        gpu_obs_pts,
        gpu_tris,
        gpu_slips,
        float_type(nu),
        grid=(n_obs_blocks, n_src_blocks, 1),
        block=(block_size, 1, 1),
    )
    out = gpu_results.get().reshape((n_obs, n_src, out_dim))
    return out


def disp(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, "disp_fullspace", 3)


def strain(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, "strain_fullspace", 6)


def disp_all_pairs(obs_pts, tris, slips, nu):
    return call_clu_all_pairs(obs_pts, tris, slips, nu, "disp_fullspace", 3)


def strain_all_pairs(obs_pts, tris, slips, nu):
    return call_clu_all_pairs(obs_pts, tris, slips, nu, "strain_fullspace", 6)


def strain_to_stress(strain, mu, nu):
    lam = 2 * mu * nu / (1 - 2 * nu)
    trace = np.sum(strain[:, :3], axis=1)
    stress = np.empty_like(strain)
    stress[:, :3] = 2 * mu * strain[:, :3] + lam * trace[:, np.newaxis]
    stress[:, 3:] = 2 * mu * strain[:, 3:]
    return stress

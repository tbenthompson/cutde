import os
import warnings

import numpy as np

import cutde.gpu as cluda

from .TDdispFS import TDdispFS

source_dir = os.path.dirname(os.path.realpath(__file__))


def py_disp(obs_pt, tri, slip, nu):
    return TDdispFS(obs_pt, tri, slip, nu)


class Placeholder:
    pass


placeholder = Placeholder()


def solve_types(obs_pts, tris, slips):
    type_map = {
        np.int32: np.float32,
        np.int64: np.float64,
        np.float32: np.float32,
        np.float64: np.float64,
    }

    float_type = None
    out_arrs = []
    for name, arr in [("obs_pts", obs_pts), ("tris", tris), ("slips", slips)]:
        if arr is placeholder:
            out_arrs.append(arr)
            continue

        dtype = arr.dtype.type

        if dtype not in type_map:
            raise ValueError(
                f"The {name} input array has type {arr.dtype} but must have a float or"
                " integer dtype."
            )

        if float_type is None:
            float_type = type_map[dtype]

            # If we're using OpenCL, we need to check if float64 is allowed.
            # If not, convert to float32.
            if cluda.ocl_backend:
                import cutde.opencl

                cutde.opencl.ensure_initialized()
                extensions = (
                    cutde.opencl.gpu_ctx.devices[0].extensions.strip().split(" ")
                )
                if "cl_khr_fp64" not in extensions and float_type is np.float64:
                    warnings.warn(
                        "The OpenCL implementation being used does not support "
                        "float64. This will require converting arrays to float32."
                    )
                    float_type = np.float32

        out_arr = arr
        if dtype != float_type:
            warnings.warn(
                f"The {name} input array has type {out_arr.dtype} but needs "
                "to be converted"
                f" to dtype {np.dtype(float_type)}. Converting {name} to "
                f"{np.dtype(float_type)} may be expensive."
            )
            out_arr = out_arr.astype(float_type)

        if out_arr.flags.f_contiguous:
            warnings.warn(
                f"The {name} input array has Fortran ordering. "
                "Converting to C ordering. This may be expensive."
            )
            out_arr = np.ascontiguousarray(out_arr)

        out_arrs.append(out_arr)

    return float_type, out_arrs


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
    if slips is not placeholder and (slips.shape[0] != tris.shape[0]):
        raise ValueError(
            "The number of input slip vectors must be equal to the number of input"
            " triangles."
        )
    if slips is not placeholder and (slips.shape[1] != 3):
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


def call_clu_matrix(obs_pts, tris, nu, fnc_name, out_dim):
    check_inputs(obs_pts, tris, placeholder)
    float_type, (obs_pts, tris, _) = solve_types(obs_pts, tris, placeholder)

    n_obs = obs_pts.shape[0]
    n_src = tris.shape[0]
    block_size = 16
    n_obs_blocks = int(np.ceil(n_obs / block_size))
    n_src_blocks = int(np.ceil(n_src / block_size))
    gpu_config = dict(block_size=block_size, float_type=cluda.np_to_c_type(float_type))
    module = cluda.load_gpu("fullspace.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = cluda.empty_gpu(n_obs * out_dim * n_src * 3, float_type)
    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_tris = cluda.to_gpu(tris, float_type)

    getattr(module, fnc_name + "_matrix")(
        gpu_results,
        np.int32(n_obs),
        np.int32(n_src),
        gpu_obs_pts,
        gpu_tris,
        float_type(nu),
        grid=(n_obs_blocks, n_src_blocks, 1),
        block=(block_size, block_size, 1),
    )
    out = gpu_results.get().reshape((n_obs, out_dim, n_src, 3))
    return out


def call_clu_free(obs_pts, tris, slips, nu, fnc_name, out_dim):
    check_inputs(obs_pts, tris, slips)
    float_type, (obs_pts, tris, slips) = solve_types(obs_pts, tris, slips)

    n_obs = obs_pts.shape[0]
    n_src = tris.shape[0]
    block_size = 256
    n_obs_blocks = int(np.ceil(n_obs / block_size))
    gpu_config = dict(block_size=block_size, float_type=cluda.np_to_c_type(float_type))
    module = cluda.load_gpu("fullspace.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = cluda.zeros_gpu(n_obs * out_dim, float_type)
    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_tris = cluda.to_gpu(tris, float_type)
    gpu_slips = cluda.to_gpu(slips, float_type)

    getattr(module, fnc_name + "_free")(
        gpu_results,
        np.int32(n_obs),
        np.int32(n_src),
        gpu_obs_pts,
        gpu_tris,
        gpu_slips,
        float_type(nu),
        grid=(n_obs_blocks, 1, 1),
        block=(block_size, 1, 1),
    )
    out = gpu_results.get().reshape((n_obs, out_dim))
    return out


def disp(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, "disp_fullspace", 3)


def strain(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, "strain_fullspace", 6)


def disp_matrix(obs_pts, tris, nu):
    return call_clu_matrix(obs_pts, tris, nu, "disp_fullspace", 3)


def strain_matrix(obs_pts, tris, nu):
    return call_clu_matrix(obs_pts, tris, nu, "strain_fullspace", 6)


def disp_free(obs_pts, tris, slips, nu):
    return call_clu_free(obs_pts, tris, slips, nu, "disp_fullspace", 3)


def strain_free(obs_pts, tris, slips, nu):
    return call_clu_free(obs_pts, tris, slips, nu, "strain_fullspace", 6)


def strain_to_stress(strain, mu, nu):
    lam = 2 * mu * nu / (1 - 2 * nu)
    trace = np.sum(strain[:, :3], axis=1)
    stress = np.empty_like(strain)
    stress[:, :3] = 2 * mu * strain[:, :3] + lam * trace[:, np.newaxis]
    stress[:, 3:] = 2 * mu * strain[:, 3:]
    return stress

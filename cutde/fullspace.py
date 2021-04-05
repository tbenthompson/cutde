import os
import warnings

import numpy as np

import cutde.gpu as cluda

from .TDdispFS import TDdispFS

source_dir = os.path.dirname(os.path.realpath(__file__))


def py_disp(obs_pt, tri, slip, nu):
    return TDdispFS(obs_pt, tri, slip, nu)


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

        if dtype != float_type:
            warnings.warn(
                f"The {name} input array has type {arr.dtype} but needs to be converted"
                f" to dtype {np.dtype(float_type)}. Converting {name} to "
                f"{np.dtype(float_type)} may be expensive."
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
    block_size = 16
    n_obs_blocks = int(np.ceil(n_obs / block_size))
    n_src_blocks = int(np.ceil(n_src / block_size))
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
        block=(block_size, block_size, 1),
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

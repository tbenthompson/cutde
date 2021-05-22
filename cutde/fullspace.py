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


def call_clu(obs_pts, tris, slips, nu, fnc):
    fnc_name, vec_dim = fnc
    if tris.shape[0] != obs_pts.shape[0]:
        raise ValueError("There must be one input observation point per triangle.")

    check_inputs(obs_pts, tris, slips)
    float_type, (obs_pts, tris, slips) = solve_types(obs_pts, tris, slips)

    n = obs_pts.shape[0]
    block_size = 128
    n_blocks = int(np.ceil(n / block_size))
    gpu_config = dict(block_size=block_size, float_type=cluda.np_to_c_type(float_type))
    module = cluda.load_gpu("pairs.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = cluda.empty_gpu(n * vec_dim, float_type)
    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_tris = cluda.to_gpu(tris, float_type)
    gpu_slips = cluda.to_gpu(slips, float_type)

    getattr(module, "pairs_" + fnc_name)(
        gpu_results,
        np.int32(n),
        gpu_obs_pts,
        gpu_tris,
        gpu_slips,
        float_type(nu),
        grid=(n_blocks, 1, 1),
        block=(block_size, 1, 1),
    )
    out = gpu_results.get().reshape((n, vec_dim))
    return out


def call_clu_matrix(obs_pts, tris, nu, fnc):
    fnc_name, vec_dim = fnc
    check_inputs(obs_pts, tris, placeholder)
    float_type, (obs_pts, tris, _) = solve_types(obs_pts, tris, placeholder)

    n_obs = obs_pts.shape[0]
    n_src = tris.shape[0]
    block_size = 16
    n_obs_blocks = int(np.ceil(n_obs / block_size))
    n_src_blocks = int(np.ceil(n_src / block_size))
    gpu_config = dict(block_size=block_size, float_type=cluda.np_to_c_type(float_type))
    module = cluda.load_gpu("matrix.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = cluda.empty_gpu(n_obs * vec_dim * n_src * 3, float_type)
    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_tris = cluda.to_gpu(tris, float_type)

    getattr(module, "matrix_" + fnc_name)(
        gpu_results,
        np.int32(n_obs),
        np.int32(n_src),
        gpu_obs_pts,
        gpu_tris,
        float_type(nu),
        grid=(n_obs_blocks, n_src_blocks, 1),
        block=(block_size, block_size, 1),
    )
    out = gpu_results.get().reshape((n_obs, vec_dim, n_src, 3))
    return out


def call_clu_free(obs_pts, tris, slips, nu, fnc):
    fnc_name, vec_dim = fnc
    check_inputs(obs_pts, tris, slips)
    float_type, (obs_pts, tris, slips) = solve_types(obs_pts, tris, slips)

    n_obs = obs_pts.shape[0]
    n_src = tris.shape[0]
    block_size = 256
    n_obs_blocks = int(np.ceil(n_obs / block_size))
    gpu_config = dict(block_size=block_size, float_type=cluda.np_to_c_type(float_type))
    module = cluda.load_gpu("free.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = cluda.zeros_gpu(n_obs * vec_dim, float_type)
    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_tris = cluda.to_gpu(tris, float_type)
    gpu_slips = cluda.to_gpu(slips, float_type)

    getattr(module, "free_" + fnc_name)(
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
    out = gpu_results.get().reshape((n_obs, vec_dim))
    return out


def process_block_inputs(obs_start, obs_end, src_start, src_end):
    out_arrs = []
    for name, a in [
        ("obs_start", obs_start),
        ("obs_end", obs_end),
        ("src_start", src_start),
        ("src_end", src_end),
    ]:
        a = np.array(a)
        if a.shape[0] != np.array(obs_start).shape[0]:
            raise ValueError(f"The length of {name} must match obs_start.")
        if not (a.dtype.type is np.int32 or a.dtype.type is np.int64):
            raise ValueError(f"The {name} array must have integer type.")

        # Don't bother warning for these conversions since the cost of
        # converting a single value per block is tiny.
        if a.flags.f_contiguous or a.dtype.type != np.int32:
            a = np.ascontiguousarray(a, dtype=np.int32)

        out_arrs.append(a)

    return out_arrs


def call_clu_block(obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, fnc):
    fnc_name, vec_dim = fnc
    check_inputs(obs_pts, tris, placeholder)
    float_type, (obs_pts, tris, _) = solve_types(obs_pts, tris, placeholder)
    obs_start, obs_end, src_start, src_end = process_block_inputs(
        obs_start, obs_end, src_start, src_end
    )

    block_sizes = vec_dim * 3 * (obs_end - obs_start) * (src_end - src_start)
    block_end = np.cumsum(block_sizes)
    block_start = np.empty(block_end.shape[0] + 1, dtype=block_end.dtype)
    block_start[:-1] = block_end - block_sizes
    block_start[-1] = block_end[-1]

    n_blocks = obs_end.shape[0]
    team_size = 16
    gpu_config = dict(float_type=cluda.np_to_c_type(float_type))
    module = cluda.load_gpu("blocks.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    gpu_results = cluda.zeros_gpu(block_end[-1], float_type)
    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_tris = cluda.to_gpu(tris, float_type)
    gpu_obs_start = cluda.to_gpu(obs_start, np.int32)
    gpu_obs_end = cluda.to_gpu(obs_end, np.int32)
    gpu_src_start = cluda.to_gpu(src_start, np.int32)
    gpu_src_end = cluda.to_gpu(src_end, np.int32)
    gpu_block_start = cluda.to_gpu(block_start, np.int32)

    getattr(module, "blocks_" + fnc_name)(
        gpu_results,
        gpu_obs_pts,
        gpu_tris,
        gpu_obs_start,
        gpu_obs_end,
        gpu_src_start,
        gpu_src_end,
        gpu_block_start,
        float_type(nu),
        grid=(n_blocks, 1, 1),
        block=(team_size, 1, 1),
    )
    return gpu_results.get(), block_start


def call_clu_aca(
    obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, tol, max_iter, fnc
):
    fnc_name, vec_dim = fnc
    check_inputs(obs_pts, tris, placeholder)
    float_type, (obs_pts, tris, _) = solve_types(obs_pts, tris, placeholder)
    obs_start, obs_end, src_start, src_end = process_block_inputs(
        obs_start, obs_end, src_start, src_end
    )

    # TODO: Get a basic one worker for each block version working.
    # TODO: Implement a team for each block.
    default_chunk_size = 16
    n_blocks = obs_end.shape[0]

    gpu_config = dict(float_type=cluda.np_to_c_type(float_type))
    module = cluda.load_gpu("aca.cu", tmpl_args=gpu_config, tmpl_dir=source_dir)

    chunk_start = 0
    chunk_size = min(n_blocks - chunk_start, default_chunk_size)
    chunk_end = chunk_start + chunk_size

    n_obs_per_block = obs_end[chunk_start:chunk_end] - obs_start[chunk_start:chunk_end]
    n_src_per_block = src_end[chunk_start:chunk_end] - src_start[chunk_start:chunk_end]
    n_rows = n_obs_per_block * vec_dim
    n_cols = n_src_per_block * 3
    block_sizes = n_rows * n_cols

    # Storage for the U, V output matrices. These will be in a packed format.
    gpu_buffer = cluda.empty_gpu(block_sizes.sum(), float_type)

    # Storage for temporary rows and columns.
    workspace_IJstar = np.max(n_cols) + np.max(n_rows)
    workspace_IJref = 3 * np.max(n_cols) + vec_dim * np.max(n_rows)
    workspace_per_block = workspace_IJstar + workspace_IJref
    fworkspace_size = workspace_per_block * n_blocks
    gpu_fworkspace = cluda.empty_gpu(fworkspace_size, float_type)

    # results_ptrs is a simple linked lists where each
    # pair of integers in the arrays is a pair where the elements are:
    # 1. The index of the start of the term in the output buffer. The U term is
    #    stored first and then the V term follows.
    # 2. The index in results_ptrs of the next term for this block.
    uv_ptrs_size = np.minimum(n_rows, n_cols)
    uv_ptrs_ends = np.cumsum(uv_ptrs_size)
    uv_ptrs_starts = uv_ptrs_ends - uv_ptrs_size
    gpu_uv_ptrs_starts = cluda.to_gpu(uv_ptrs_starts, np.int32)
    gpu_uv_ptrs = cluda.empty_gpu(uv_ptrs_ends[-1], np.int32)
    gpu_iworkspace = cluda.empty_gpu(uv_ptrs_ends[-1], np.int32)

    # Output space for specifying the number of terms used for each
    # approximation.
    gpu_n_terms = cluda.empty_gpu(chunk_size, np.int32)

    # Storage space for a pointer to the next empty portion of the output
    # buffer.
    gpu_next_ptr = cluda.zeros_gpu(1, np.int32)

    # The index of the starting reference rows/cols.
    # TODO:
    # TODO:
    # TODO:
    # TODO:
    Iref0 = 0 * (np.random.rand(chunk_size) * n_rows).astype(np.int32)
    Jref0 = 0 * (np.random.rand(chunk_size) * n_cols).astype(np.int32)
    gpu_Iref0 = cluda.to_gpu(Iref0, np.int32)
    gpu_Jref0 = cluda.to_gpu(Jref0, np.int32)

    gpu_obs_pts = cluda.to_gpu(obs_pts, float_type)
    gpu_tris = cluda.to_gpu(tris, float_type)
    gpu_obs_start = cluda.to_gpu(obs_start, np.int32)
    gpu_obs_end = cluda.to_gpu(obs_end, np.int32)
    gpu_src_start = cluda.to_gpu(src_start, np.int32)
    gpu_src_end = cluda.to_gpu(src_end, np.int32)

    print(f"gpu_buffer.shape = {gpu_buffer.shape}")
    print(f"gpu_uv_ptrs.shape = {gpu_uv_ptrs.shape}")
    print(f"gpu_n_terms.shape = {gpu_n_terms.shape}")
    print(f"gpu_next_ptr.shape = {gpu_next_ptr.shape}")
    print(f"gpu_fworkspace.shape = {gpu_fworkspace.shape}")
    print(f"gpu_iworkspace.shape = {gpu_iworkspace.shape}")
    print(f"gpu_uv_ptrs_starts.shape = {gpu_uv_ptrs_starts.shape}")
    print(f"gpu_Iref0.shape = {gpu_Iref0.shape}")
    print(f"gpu_Jref0.shape = {gpu_Jref0.shape}")
    print(f"obs_pts.shape = {obs_pts.shape}")
    print(f"tris.shape = {tris.shape}")
    print(f"gpu_obs_start.shape = {gpu_obs_start.shape}")
    print(f"gpu_obs_end.shape = {gpu_obs_end.shape}")
    print(f"gpu_src_start.shape = {gpu_src_start.shape}")
    print(f"gpu_src_end.shape = {gpu_src_end.shape}")

    getattr(module, "aca_" + fnc_name)(
        gpu_buffer,
        gpu_uv_ptrs,
        gpu_n_terms,
        gpu_next_ptr,
        gpu_fworkspace,
        gpu_iworkspace,
        gpu_uv_ptrs_starts,
        gpu_Iref0,
        gpu_Jref0,
        gpu_obs_pts,
        gpu_tris,
        gpu_obs_start,
        gpu_obs_end,
        gpu_src_start,
        gpu_src_end,
        float_type(nu),
        float_type(tol),
        np.int32(max_iter),
        grid=(chunk_size, 1, 1),
        block=(1, 1, 1),
    )

    # TODO: post-process the buffer to collect the U, V vectors
    buffer = gpu_buffer.get()
    uv_ptrs = gpu_uv_ptrs.get()
    n_terms = gpu_n_terms.get()
    appxs = []
    for i in range(chunk_size):
        us = []
        vs = []
        uv_ptr0 = uv_ptrs_starts[i]
        for k in range(n_terms[i]):
            ptr = uv_ptrs[uv_ptr0 + k]
            us.append(buffer[ptr : (ptr + n_rows[i])])
            vs.append(buffer[(ptr + n_rows[i]) : (ptr + n_rows[i] + n_cols[i])])
        appxs.append((np.array(us), np.array(vs)))
    return appxs


DISP = ("disp", 3)
STRAIN = ("strain", 6)


def disp(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, DISP)


def strain(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, STRAIN)


def disp_matrix(obs_pts, tris, nu):
    return call_clu_matrix(obs_pts, tris, nu, DISP)


def strain_matrix(obs_pts, tris, nu):
    return call_clu_matrix(obs_pts, tris, nu, STRAIN)


def disp_free(obs_pts, tris, slips, nu):
    return call_clu_free(obs_pts, tris, slips, nu, DISP)


def strain_free(obs_pts, tris, slips, nu):
    return call_clu_free(obs_pts, tris, slips, nu, STRAIN)


def disp_block(obs_pts, tris, obs_start, obs_end, src_start, src_end, nu):
    return call_clu_block(
        obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, DISP
    )


def strain_block(obs_pts, tris, obs_start, obs_end, src_start, src_end, nu):
    return call_clu_block(
        obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, STRAIN
    )


def disp_aca(obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, tol, max_iter):
    return call_clu_aca(
        obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, tol, max_iter, DISP
    )


def strain_aca(
    obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, tol, max_iter
):
    return call_clu_aca(
        obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, tol, max_iter, STRAIN
    )

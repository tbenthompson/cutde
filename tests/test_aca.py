import numpy as np
import py_aca
from test_tde import setup_matrix_test

import cutde


# @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
# @pytest.mark.parametrize("F_ordered", [True, False])
# @pytest.mark.parametrize("field", ["disp", "strain"])
def test_aca():
    """
    This checks that the OpenCL/CUDA ACA implementation is producing *exactly*
    the same results as the Python implementation.
    """
    dtype, F_ordered, field = np.float64, False, "disp"

    if field == "disp":
        matrix_fnc = cutde.disp_matrix
        aca_fnc = cutde.disp_aca
        vec_dim = 3
    else:
        matrix_fnc = cutde.strain_matrix
        aca_fnc = cutde.strain_aca
        vec_dim = 6

    pts = []
    tris = []
    n_sets = 1
    set_sizes = np.arange(50, 50 + n_sets)
    set_sizes_edges = np.zeros(set_sizes.shape[0] + 1, dtype=np.int32)
    set_sizes_edges[1:] = np.cumsum(set_sizes)
    for i in range(n_sets):
        S = set_sizes[i]
        this_pts, this_tris, _ = setup_matrix_test(dtype, F_ordered, n_obs=S, n_src=S)
        this_pts[:, 0] -= 150
        pts.append(this_pts)
        tris.append(this_tris)
    pts = np.concatenate(pts)
    tris = np.concatenate(tris)

    obs_starts = []
    obs_ends = []
    src_starts = []
    src_ends = []
    for i in range(n_sets):
        for j in range(n_sets):
            obs_starts.append(set_sizes_edges[i])
            obs_ends.append(set_sizes_edges[i + 1])
            src_starts.append(set_sizes_edges[j])
            src_ends.append(set_sizes_edges[j + 1])

    M1 = matrix_fnc(pts, tris, 0.25)
    M1 = M1.reshape((M1.shape[0] * M1.shape[1], M1.shape[2] * M1.shape[3]))

    M2 = aca_fnc(pts, tris, obs_starts, obs_ends, src_starts, src_ends, 0.25, 1e-4, 200)

    for block_idx in range(len(obs_starts)):
        os = obs_starts[block_idx]
        oe = obs_ends[block_idx]
        ss = src_starts[block_idx]
        se = src_ends[block_idx]

        block = M1[(os * vec_dim) : (oe * vec_dim), (3 * ss) : (3 * se)]

        U, V = py_aca.ACA_plus(
            block.shape[0],
            block.shape[1],
            lambda Istart, Iend: block[Istart:Iend, :],
            lambda Jstart, Jend: block[:, Jstart:Jend],
            1e-4,
            verbose=True,
            Iref=0,
            Jref=0,
        )

        U2, V2 = M2[block_idx]
        np.testing.assert_almost_equal(U, U2)
        np.testing.assert_almost_equal(V, V2)

        diff = block - U2.dot(V2)
        diff_frob = np.sqrt(np.sum(diff ** 2))
        block_frob = np.sqrt(np.sum(block ** 2))
        print(diff_frob, block_frob, diff_frob / block_frob)
        assert diff_frob / block_frob < 1e-3


if __name__ == "__main__":
    # test_aca(1)
    test_aca()

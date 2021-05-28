import time

import numpy as np
import py_aca
import pytest
from test_tde import setup_matrix_test

import cutde
from cutde.aca import call_clu_aca
from cutde.fullspace import DISP, STRAIN


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("F_ordered", [True, False])
@pytest.mark.parametrize("field", ["disp", "strain"])
def test_aca_slow(dtype, F_ordered, field):
    # The fast version of this test doesn't test multiple blocks and would miss
    # some potential errors.
    runner(dtype, F_ordered, field, 10, compare_against_py=True)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("F_ordered", [True, False])
@pytest.mark.parametrize("field", ["disp", "strain"])
def test_aca_fast(dtype, F_ordered, field):
    runner(dtype, F_ordered, field, 1, compare_against_py=False)


def runner(dtype, F_ordered, field, n_sets, compare_against_py=False, benchmark=False):
    """
    This checks that the OpenCL/CUDA ACA implementation is producing *exactly*
    the same results as the prototype Python implementation and that both ACA
    implementations are within the expected Frobenius norm tolerance of the
    exact calculation.
    """
    if field == "disp":
        matrix_fnc = cutde.disp_matrix
        aca_fnc = cutde.disp_aca
        field_spec = DISP
        vec_dim = 3
    else:
        matrix_fnc = cutde.strain_matrix
        aca_fnc = cutde.strain_aca
        field_spec = STRAIN
        vec_dim = 6

    pts = []
    tris = []
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

    iters = 5 if benchmark else 1
    times = []
    for i in range(iters):
        start = time.time()
        if compare_against_py:
            M2 = call_clu_aca(
                pts,
                tris,
                obs_starts,
                obs_ends,
                src_starts,
                src_ends,
                0.25,
                [1e-4] * len(obs_starts),
                [200] * len(obs_starts),
                field_spec,
                Iref0=np.zeros_like(obs_starts),
                Jref0=np.zeros_like(obs_starts),
            )
        else:
            M2 = aca_fnc(
                pts,
                tris,
                obs_starts,
                obs_ends,
                src_starts,
                src_ends,
                0.25,
                [1e-4] * len(obs_starts),
                [200] * len(obs_starts),
            )
        times.append(time.time() - start)

    if benchmark:
        # ignore the first runtime since it includes compilation and CUDA initialization
        print("aca runtime, min=", np.min(times[1:]), "  median=", np.median(times[1:]))

    for block_idx in range(len(obs_starts)):
        os = obs_starts[block_idx]
        oe = obs_ends[block_idx]
        ss = src_starts[block_idx]
        se = src_ends[block_idx]
        block = M1[(os * vec_dim) : (oe * vec_dim), (3 * ss) : (3 * se)]
        U2, V2 = M2[block_idx]
        if compare_against_py:
            U, V = py_aca.ACA_plus(
                block.shape[0],
                block.shape[1],
                lambda Istart, Iend: block[Istart:Iend, :],
                lambda Jstart, Jend: block[:, Jstart:Jend],
                1e-4,
                verbose=False,
                Iref=0,
                Jref=0,
                vec_dim=3 if field == "disp" else 6,
            )

            if pts.dtype.type is np.float64:
                U2, V2 = M2[block_idx]
                np.testing.assert_almost_equal(U, U2, 10)
                np.testing.assert_almost_equal(V, V2, 10)

        diff = block - U2.dot(V2)
        diff_frob = np.sqrt(np.sum(diff ** 2))
        assert diff_frob < 5e-4


if __name__ == "__main__":
    runner(np.float32, False, "disp", 40, compare_against_py=False, benchmark=True)

import logging
import sys
import time

import numpy as np
import pytest
import scipy.io

import cutde


def enable_logging():
    root = logging.getLogger()
    level = logging.INFO
    # level = logging.DEBUG
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


enable_logging()


def get_pt_grid():
    n = 11
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    z = np.linspace(-3, 3, n)
    X, Z, Y = np.meshgrid(x, z, y)
    test_pts = np.hstack([arr.flatten()[:, np.newaxis] for arr in [X, Y, Z]])
    return test_pts


def get_simple_test():
    correct = scipy.io.loadmat("tests/result_simple.mat")
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    slip = np.array([1.0, 0, 0])
    return correct, get_pt_grid(), tri, slip


def get_complex_test():
    correct = scipy.io.loadmat("tests/result_complex.mat")
    tri = np.array([[0, 0.1, 0.1], [1, -0.2, -0.2], [1, 1, 0.3]])
    slip = [1.3, 1.4, 1.5]
    return correct, get_pt_grid(), tri, slip


def py_tde_tester(setup_fnc, N_test=-1):
    correct, test_pts, tri, slip = setup_fnc()

    if N_test == -1:
        N_test = correct["UEf"].shape[0]

    results = np.empty((N_test, 3))
    start = time.time()
    for i in range(N_test):
        pt = test_pts[i, :]
        results[i, :] = cutde.py_disp(pt, tri, slip, 0.25)
        np.testing.assert_almost_equal(results[i, 0], correct["UEf"][i, 0])
        np.testing.assert_almost_equal(results[i, 1], correct["UNf"][i, 0])
        np.testing.assert_almost_equal(results[i, 2], correct["UVf"][i, 0])
        if i % 10000 == 0 and i != 0:
            print(i, (time.time() - start) / i)

    np.testing.assert_almost_equal(results[:, 0], correct["UEf"][:N_test, 0])
    np.testing.assert_almost_equal(results[:, 1], correct["UNf"][:N_test, 0])
    np.testing.assert_almost_equal(results[:, 2], correct["UVf"][:N_test, 0])


def test_py_simple():
    py_tde_tester(get_simple_test, N_test=1000)


def test_py_complex():
    py_tde_tester(get_complex_test, N_test=1000)


def cluda_tde_tester(setup_fnc):
    correct, test_pts, tri, slip = setup_fnc()
    N_test = correct["UEf"].shape[0]

    nu = 0.25
    sm = 1.0

    tris = np.array([tri] * N_test, dtype=np.float64)
    slips = np.array([slip] * N_test)

    disp = cutde.disp(test_pts[:N_test], tris, slips, 0.25)
    strain = cutde.strain(test_pts[:N_test], tris, slips, nu)
    stress = cutde.strain_to_stress(strain, sm, nu)

    np.testing.assert_almost_equal(disp[:, 0], correct["UEf"][:N_test, 0])
    np.testing.assert_almost_equal(disp[:, 1], correct["UNf"][:N_test, 0])
    np.testing.assert_almost_equal(disp[:, 2], correct["UVf"][:N_test, 0])
    np.testing.assert_almost_equal(strain, correct["Strain"][:N_test])
    np.testing.assert_almost_equal(stress, correct["Stress"][:N_test])

    test_ptsF = np.asfortranarray(test_pts[:N_test])
    trisF = np.asfortranarray(tris)
    slipsF = np.asfortranarray(slips)
    dispF = cutde.disp(test_ptsF, trisF, slipsF, 0.25)
    np.testing.assert_almost_equal(disp, dispF)


def test_cluda_simple():
    cluda_tde_tester(get_simple_test)


def setup_matrix_test(dtype, F_ordered):
    np.random.seed(10)

    n_obs = 100
    n_src = 100

    def random_vals(shape, max_val):
        out = (np.random.rand(*shape) * max_val).astype(dtype)
        return np.asfortranarray(out) if F_ordered else out

    pts = random_vals((n_obs, 3), max_val=100)
    tris = random_vals((n_src, 3, 3), max_val=100)
    slips = random_vals((n_src, 3), max_val=100)

    return pts, tris, slips


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("F_ordered", [True, False])
@pytest.mark.parametrize("field", ["disp", "strain"])
def test_matrix(dtype, F_ordered, field):
    pts, tris, slips = setup_matrix_test(dtype, F_ordered)

    if field == "disp":
        simple_fnc = cutde.disp
        matrix_fnc = cutde.disp_matrix
    else:
        simple_fnc = cutde.strain
        matrix_fnc = cutde.strain_matrix

    strain_mat1 = []
    slips[:] = 1
    for i in range(pts.shape[0]):
        tiled_pt = np.tile(pts[i, np.newaxis, :], (tris.shape[0], 1))
        strain_mat1.append(simple_fnc(tiled_pt, tris, slips, 0.25))
    strain_mat1 = np.array(strain_mat1)
    S1 = np.sum(strain_mat1, axis=1).flatten()

    strain_mat2 = matrix_fnc(pts, tris, 0.25)
    M = strain_mat2.reshape((-1, 3, slips.size))
    S2 = M.dot(slips.flatten()).flatten()

    rtol = 2e-4 if pts.dtype.type in [np.float32, np.int32] else 4e-10
    np.testing.assert_allclose(S1, S2, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("F_ordered", [True, False])
@pytest.mark.parametrize("field", ["disp", "strain"])
def test_matrix_free(dtype, F_ordered, field):
    pts, tris, slips = setup_matrix_test(dtype, F_ordered)

    if field == "disp":
        matrix_fnc = cutde.disp_matrix
        free_fnc = cutde.disp_free
    else:
        matrix_fnc = cutde.strain_matrix
        free_fnc = cutde.strain_free

    strain_mat = matrix_fnc(pts, tris, 0.25)
    M = strain_mat.reshape((-1, slips.size))
    S1 = M.dot(slips.flatten())

    S2 = np.zeros_like(S1)
    for d in range(3):
        slips_chunk = np.zeros_like(slips)
        slips_chunk[:, d] = slips[:, d]
        S2 += free_fnc(pts, tris, slips_chunk, 0.25).flatten()
    # S2 = free_fnc(pts, tris, slips, 0.25).flatten()

    if pts.dtype.type in [np.float32, np.int32]:
        rtol, atol = 2e-3, 3e-3
    else:
        rtol, atol = 4e-10, 1e-15
    np.testing.assert_allclose(S1, S2, rtol=rtol, atol=atol)

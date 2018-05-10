import time
import numpy as np
import scipy.io


import cutde.fullspace

def get_pt_grid():
    n = 101;
    x = np.linspace(-3, 3, n);
    y = np.linspace(-3, 3, n);
    z = np.linspace(-3, 3, n);
    X, Z, Y = np.meshgrid(x, z, y)
    test_pts = np.hstack((arr.flatten()[:,np.newaxis] for arr in [X, Y, Z]))
    return test_pts

def get_simple_test():
    correct = scipy.io.loadmat('tests/result_simple.mat')
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    slip = np.array([1.0, 0, 0])
    return correct, get_pt_grid(), tri, slip

def get_complex_test():
    correct = scipy.io.loadmat('tests/result_complex.mat')
    tri = np.array([
        [0,0.1,0.1],
        [1,-0.2,-0.2],
        [1,1,0.3]
    ])
    slip = [1.3,1.4,1.5];
    return correct, get_pt_grid(), tri, slip

def py_tde_tester(setup_fnc, N_test = -1):
    correct, test_pts, tri, slip = setup_fnc()

    if N_test == -1:
        N_test = correct['UEf'].shape[0]

    results = np.empty((N_test, 3))
    start = time.time()
    for i in range(N_test):
        pt = test_pts[i,:]
        results[i,:] = cutde.fullspace.py_disp(pt, tri, slip, 0.25)
        np.testing.assert_almost_equal(results[i,0], correct['UEf'][i,0])
        np.testing.assert_almost_equal(results[i,1], correct['UNf'][i,0])
        np.testing.assert_almost_equal(results[i,2], correct['UVf'][i,0])
        if i % 10000 == 0 and i != 0:
            print(i, (time.time() - start) / i)

    np.testing.assert_almost_equal(results[:,0], correct['UEf'][:N_test,0])
    np.testing.assert_almost_equal(results[:,1], correct['UNf'][:N_test,0])
    np.testing.assert_almost_equal(results[:,2], correct['UVf'][:N_test,0])

def test_py_simple():
    py_tde_tester(get_simple_test, N_test = 1000)

def test_py_complex():
    py_tde_tester(get_complex_test, N_test = 1000)

def cluda_tde_tester(setup_fnc):
    correct, test_pts, tri, slip = setup_fnc()
    N_test = correct['UEf'].shape[0]

    nu = 0.25
    sm = 1.0

    tris = np.array([tri] * N_test)
    slips = np.array([slip] * N_test)

    # disp = cutde.fullspace.clu_disp(test_pts[:N_test], tris, slips, 0.25)
    strain = cutde.fullspace.clu_strain(test_pts[:N_test], tris, slips, nu)
    stress = cutde.fullspace.strain_to_stress(strain, sm, nu)

    # np.testing.assert_almost_equal(disp[:,0], correct['UEf'][:N_test,0])
    # np.testing.assert_almost_equal(disp[:,1], correct['UNf'][:N_test,0])
    # np.testing.assert_almost_equal(disp[:,2], correct['UVf'][:N_test,0])
    np.testing.assert_almost_equal(strain, correct['Strain'][:N_test])
    np.testing.assert_almost_equal(stress, correct['Stress'][:N_test])

def test_cluda_simple():
    cluda_tde_tester(get_simple_test)

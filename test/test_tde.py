import time
import numpy as np
import scipy.io


from TDdispFS import TDdispFS
import tectosaur.util.gpu as gpu
def get_gpu_module(float_type):
    return gpu.load_gpu('farfield_direct.cl', tmpl_args = get_gpu_config(float_type))

def get_pt_grid():
    n = 101;
    x = np.linspace(-3, 3, n);
    y = np.linspace(-3, 3, n);
    z = np.linspace(-3, 3, n);
    X, Z, Y = np.meshgrid(x, z, y)
    # X = np.swapaxes(X, 1, 2)
    # Y = np.swapaxes(Y, 1, 2)
    # Z = np.swapaxes(Z, 1, 2)
    test_pts = np.hstack((arr.flatten()[:,np.newaxis] for arr in [X, Y, Z]))
    return test_pts

def get_simple_test():
    correct = scipy.io.loadmat('result_simple.mat')
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    slip = np.array([1.0, 0, 0])
    return correct, get_pt_grid(), tri, slip

def get_complex_test():
    correct = scipy.io.loadmat('result_complex.mat')
    tri = np.array([
        [0,0.1,0.1],
        [1,-0.2,-0.2],
        [1,1,0.3]
    ])
    slip = [1.3,1.4,1.5];
    return correct, get_pt_grid(), tri, slip

def py_tde_tester(setup_fnc):
    correct, test_pts, tri, slip = setup_fnc()

    N_test = correct['UEf'].shape[0]

    results = np.empty((N_test, 3))
    import time
    start = time.time()
    for i in range(N_test):
        pt = test_pts[i,:]
        results[i,:] = TDdispFS(pt, tri, slip, 0.25)
        np.testing.assert_almost_equal(results[i,0], correct['UEf'][i,0])
        np.testing.assert_almost_equal(results[i,1], correct['UNf'][i,0])
        np.testing.assert_almost_equal(results[i,2], correct['UVf'][i,0])
        if i % 10000 == 0 and i != 0:
            print(i, (time.time() - start) / i)

    np.testing.assert_almost_equal(results[:,0], correct['UEf'][:N_test,0])
    np.testing.assert_almost_equal(results[:,1], correct['UNf'][:N_test,0])
    np.testing.assert_almost_equal(results[:,2], correct['UVf'][:N_test,0])

def cuda_tde(obs_pts, tris, slips, nu):
    n = obs_pts.shape[0]
    block_size = 128
    n_blocks = int(np.ceil(n / block_size))
    float_type = np.float64
    gpu_config = dict(
        block_size = block_size,
        float_type = gpu.np_to_c_type(float_type)
    )
    module = gpu.load_gpu('disp_fullspace.cu', tmpl_args = gpu_config, tmpl_dir = '.', save_code = True)

    gpu_results = gpu.empty_gpu(n * 3, float_type)
    gpu_obs_pts = gpu.to_gpu(obs_pts, float_type)
    gpu_tris = gpu.to_gpu(tris, float_type)
    gpu_slips = gpu.to_gpu(slips, float_type)

    start = time.time()
    module.disp_fullspace(
        gpu_results, np.int32(n), gpu_obs_pts, gpu_tris, gpu_slips, float_type(nu),
        grid = (n_blocks, 1, 1), block = (block_size, 1, 1)
    )
    out = gpu_results.get().reshape((n, 3))
    print(time.time() - start)
    return out

def cuda_tde_tester(setup_fnc):
    correct, test_pts, tri, slip = setup_fnc()
    N_test = correct['UEf'].shape[0]

    # m = 100
    # new_test_pts = np.tile(test_pts, (m, 1))
    # results = cuda_tde(
    #     new_test_pts,
    #     np.array([tri] * new_test_pts.shape[0]),
    #     np.array([slip] * new_test_pts.shape[0]),
    #     0.25
    # )

    results = cuda_tde(
        test_pts[:N_test],
        np.array([tri] * N_test),
        np.array([slip] * N_test),
        0.25
    )

    np.testing.assert_almost_equal(results[:,0], correct['UEf'][:N_test,0])
    np.testing.assert_almost_equal(results[:,1], correct['UNf'][:N_test,0])
    np.testing.assert_almost_equal(results[:,2], correct['UVf'][:N_test,0])

def test_py_simple():
    py_tde_tester(get_simple_test)

def test_py_complex():
    py_tde_tester(get_complex_test)

def test_cuda_simple():
    cuda_tde_tester(get_simple_test)

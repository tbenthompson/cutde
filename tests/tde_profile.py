import time

import numpy as np
from scipy.sparse.linalg import gmres

import cutde
import cutde.gpu


def surface(n_els_per_dim):
    surf_L = 4000
    mesh_xs = np.linspace(-surf_L, surf_L, n_els_per_dim + 1)
    mesh_ys = np.linspace(-surf_L, surf_L, n_els_per_dim + 1)
    mesh_xg, mesh_yg = np.meshgrid(mesh_xs, mesh_ys)
    surf_pts = np.array([mesh_xg, mesh_yg, 0 * mesh_yg]).reshape((3, -1)).T.copy()
    surf_tris = []
    ny = n_els_per_dim + 1

    def idx(i, j):
        return i * ny + j

    for i in range(n_els_per_dim):
        for j in range(n_els_per_dim):
            x1, x2 = mesh_xs[i : i + 2]
            y1, y2 = mesh_ys[j : j + 2]
            surf_tris.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            surf_tris.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    surf_tris = np.array(surf_tris, dtype=np.int64)
    surf_tri_pts = surf_pts[surf_tris]
    surf_centroids = np.mean(surf_tri_pts, axis=1)
    return (
        surf_centroids + np.array([0, 0, 0.01]),
        surf_tri_pts,
        np.random.rand(surf_tri_pts.shape[0], 3),
    )


def main():
    pts, tris, slips = surface(50)
    pts = cutde.gpu.to_gpu(pts, np.float32)
    tris = cutde.gpu.to_gpu(tris, np.float32)
    slips = cutde.gpu.to_gpu(slips, np.float32)

    pairs = pts.shape[0] * tris.shape[0] / 1e6

    def profile(fnc, n_iter=5):
        for i in range(n_iter):
            start = time.time()
            out = fnc()
            rt = time.time() - start
            pairs_per_sec = pairs / rt
            print(pairs, "million took", rt, " -- ", pairs_per_sec, "million pairs/sec")
        return out

    print("profiling matrix")
    profile(lambda: cutde.strain_matrix(pts, tris, 0.25))

    print("profiling matrix free")
    profile(lambda: cutde.strain_free(pts, tris, slips, 0.25))

    print("profiling matrix vector product")
    disp_mat = cutde.disp_matrix(pts, tris, 0.25)
    disp_mat2d = disp_mat.reshape((pts.shape[0] * 3, tris.shape[0] * 3))
    slips_np = slips.get().flatten()
    profile(lambda: disp_mat2d.dot(slips_np))

    print("profiling iterative inverse")
    lhs = np.empty_like(disp_mat)
    lhs[:, :, :, 0] = disp_mat[:, :, :, 1]
    lhs[:, :, :, 1] = disp_mat[:, :, :, 0]
    lhs[:, :, :, 2] = disp_mat[:, :, :, 2]
    lhs = lhs.reshape((pts.shape[0] * 3, tris.shape[0] * 3))
    lhs += np.eye(lhs.shape[0])
    b = np.random.rand(lhs.shape[0])

    def track(x):
        track.n_iter += 1

    track.n_iter = 0
    out = profile(lambda: gmres(lhs, b, callback=track), n_iter=1)
    print(out)
    print(f"iterative solve took {track.n_iter} iterations")


if __name__ == "__main__":
    main()

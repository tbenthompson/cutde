import numpy as np
import matplotlib.pyplot as plt
import common

n = 50
gauss_params = (1.0, 0.0, 0.3)
gauss_center = (1.0, 0.0, 0.0)
sm = 1.0
nu = 0.25
folder = 'results/bend_rects'
common.check_folder(folder)

def normals(tri_pts):
    unscaled_normals = np.cross(
        tri_pts[:,2,:] - tri_pts[:,0,:],
        tri_pts[:,2,:] - tri_pts[:,1,:]
    )
    return unscaled_normals / np.linalg.norm(unscaled_normals, axis = 1)[:, np.newaxis]

def main():
    pts, rects = common.bend_mesh(n)
    pts, tris = common.build_weird_tri_mesh(pts, rects)

    n_tris = tris.shape[0]
    n_rects = rects.shape[0]
    assert(3 * n_rects == n_tris)
    tri_pts = pts[tris]
    rect_pts = pts[rects]
    tri_centers = np.mean(tri_pts, axis = 1)
    tri_dist = np.linalg.norm(tri_centers - gauss_center, axis = 1)
    rect_centers = np.mean(rect_pts, axis = 1)
    rect_dist = np.linalg.norm(rect_centers - gauss_center, axis = 1)

    slip = np.zeros((rect_centers.shape[0], 3))
    slip[:,0] = common.gaussian(*gauss_params, rect_dist)
    def xyz_to_tde(vec, rect_pts):
        rect_normals = normals(rect_pts)
        dip_vec = np.zeros_like(rect_normals)
        dip_vec[:,2] = 1.0
        strike_vec = np.cross(rect_normals, dip_vec)

        def proj(V, b):
            numer = np.sum(V * b, axis = 1)[:, np.newaxis] * b
            denom = np.linalg.norm(b, axis = 1)[:, np.newaxis] ** 2
            return numer / denom
        out = np.empty_like(vec)
        out[:,0] = np.linalg.norm(proj(vec, strike_vec), axis = 1)
        out[:,1] = np.linalg.norm(proj(vec, dip_vec), axis = 1)
        out[:,2] = np.linalg.norm(proj(vec, rect_normals), axis = 1)
        return out

    slip_tde = xyz_to_tde(slip, rect_pts)
    slip_tris = common.dofs_to_tris(slip_tde)

    strain, stress = common.eval_tris(rect_centers, tri_pts, slip_tris, sm, nu)

    stress_tris = common.dofs_to_tris(stress)
    common.plot_tris((pts, tris), stress_tris[:,3], 'rectsxy', folder)
    common.plot_tris((pts, tris), stress_tris[:,5], 'rectsyz', folder)

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import common

n = 50
gauss_params = (1.0, 0.0, 0.3)
gauss_center = (1.0, 0.0, 0.0)
sm = 1.0
nu = 0.25
folder = 'results/bend_tris'
common.check_folder(folder)

def main():
    pts, rects = common.bend_mesh(n)
    m = common.rect_to_tri_mesh(pts, rects)

    n_tris = m[1].shape[0]
    tri_pts = m[0][m[1]]
    tri_centers = np.mean(tri_pts, axis = 1)

    dist = np.linalg.norm(tri_centers - gauss_center, axis = 1)
    strike_slip = common.gaussian(*gauss_params, dist)
    slip = np.zeros((strike_slip.shape[0], 3))
    slip[:,0] = strike_slip

    strain, stress = common.eval_tris(tri_centers, tri_pts, slip, sm, nu)

    common.plot_tris(m, stress[:,3], 'trisxy', folder)
    common.plot_tris(m, stress[:,5], 'trisyz', folder)

if __name__ == "__main__":
    main()

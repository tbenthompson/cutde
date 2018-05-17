import os
import numpy as np
import matplotlib.pyplot as plt

import common

import cutde.fullspace

n = 50
gauss_params = (1.0, 0.0, 0.3)
sm = 1.0
nu = 0.25
corners = [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]]
tri_srces = True
rotate = True
if tri_srces:
    folder = 'results/rectobs_trisrc'
else:
    folder = 'results/rects'
if rotate:
    folder += '_rot'
common.check_folder(folder)

def main():
    pts, rects = common.make_rect_of_rects(n, n, corners)
    pts, tris = common.build_weird_tri_mesh(pts, rects)
    if rotate:
        pts = pts.dot(np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))
    # plt.triplot(pts[:, 0], pts[:, 2], tris)
    # plt.show()

    n_tris = tris.shape[0]
    n_rects = rects.shape[0]
    assert(3 * n_rects == n_tris)
    tri_pts = pts[tris]
    rect_pts = pts[rects]
    tri_centers = np.mean(tri_pts, axis = 1)
    tri_dist = np.linalg.norm(tri_centers, axis = 1)
    rect_centers = np.mean(rect_pts, axis = 1)
    rect_dist = np.linalg.norm(rect_centers, axis = 1)

    plt.figure(figsize = (15,15))
    plt.triplot(pts[:,0], pts[:,2], tris)
    plt.plot(rect_centers[:,0], rect_centers[:,2], '*')
    plt.savefig(os.path.join(folder,'rect_setup.pdf'))


    if tri_srces:
        strike_slip = common.gaussian(*gauss_params, tri_dist)
        slip_tris = np.zeros((strike_slip.shape[0], 3))
        slip_tris[:,0] = strike_slip
    else:
        strike_slip = common.gaussian(*gauss_params, rect_dist)
        slip = np.zeros((strike_slip.shape[0], 3))
        slip[:,0] = strike_slip
        slip_tris = common.dofs_to_tris(slip)

    common.plot_tris((pts, tris), slip_tris[:,0], 'inputslip', folder)

    strain = np.empty((n_rects, 6))
    for i in range(n_rects):
        obs_pts = np.tile(rect_centers[i,np.newaxis,:], (n_tris, 1))
        all_strains = cutde.fullspace.clu_strain(obs_pts, tri_pts, slip_tris, nu)
        strain[i, :] = np.sum(all_strains, axis = 0)
    stress = cutde.fullspace.strain_to_stress(strain, sm, nu)

    stress_tris = common.dofs_to_tris(stress)
    # for d in range(6):
    #     plot_tris((pts, tris), stress_tris[:,d])
    common.plot_tris((pts, tris), stress_tris[:,3], 'rectsxy', folder)
    common.plot_tris((pts, tris), stress_tris[:,5], 'rectsyz', folder)



if __name__ == "__main__":
    main()

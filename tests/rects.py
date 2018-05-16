import numpy as np
import matplotlib.pyplot as plt

from tris import gaussian, gauss_params, sm, nu, plot_tris
from tris import main as tri_main
from mesh_gen import make_rect, rect_points

import cutde.fullspace

n = 50
corners = [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]]

def dofs_to_tris(field, factor = 3):
    n_rects = field.shape[0]
    out = np.tile(field.reshape((n_rects, -1))[:,np.newaxis,:], (1,factor,1))
    out_shape = list(field.shape)
    out_shape[0] *= factor
    return out.reshape(out_shape)

def build_weird_tri_mesh(nx, ny, corners):
    true_nx = 2 * nx - 1
    x = np.linspace(0, 1, 2 * nx - 1)
    y = np.linspace(0, 1, ny)
    pts = rect_points(corners, x, y)

    def v_idx(i, j):
        return j * true_nx + i * 2

    def midpt_idx(i, j):
        return j * true_nx + i * 2 + 1

    tris = []
    rects = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            top_left = v_idx(i, j)
            top_right = v_idx(i + 1, j)
            bottom_left = v_idx(i, j + 1)
            bottom_right = v_idx(i + 1, j + 1)
            top_middle = midpt_idx(i, j)
            tris.append([top_left, bottom_left, top_middle])
            tris.append([top_middle, bottom_right, top_right])
            tris.append([bottom_left, bottom_right, top_middle])
            rects.append([top_left, bottom_left, bottom_right, top_right])
    tris = np.array(tris, dtype = np.int)
    rects = np.array(rects, dtype = np.int)
    return pts, tris, rects

def main():
    pts, tris, rects = build_weird_tri_mesh(n, n, corners)

    n_tris = tris.shape[0]
    n_rects = rects.shape[0]
    assert(3 * n_rects == n_tris)
    tri_pts = pts[tris]
    rect_pts = pts[rects]
    rect_centers = np.mean(rect_pts, axis = 1)
    dist = np.linalg.norm(rect_centers, axis = 1)

    plt.figure(figsize = (15,15))
    plt.triplot(pts[:,0], pts[:,2], tris)
    plt.plot(rect_centers[:,0], rect_centers[:,2], '*')
    plt.savefig('rect_setup.pdf')
    # plt.show()


    strike_slip = gaussian(*gauss_params, dist)
    slip = np.zeros((strike_slip.shape[0], 3))
    slip[:,0] = strike_slip
    slip_tris = dofs_to_tris(slip)

    plot_tris((pts, tris), slip_tris[:,0], 'inputslip')

    strain = np.empty((n_rects, 6))
    for i in range(n_rects):
        obs_pts = np.tile(rect_centers[i,np.newaxis,:], (n_tris, 1))
        all_strains = cutde.fullspace.clu_strain(obs_pts, tri_pts, slip_tris, nu)
        strain[i, :] = np.sum(all_strains, axis = 0)
    stress = cutde.fullspace.strain_to_stress(strain, sm, nu)

    stress_tris = dofs_to_tris(stress)
    # for d in range(6):
    #     plot_tris((pts, tris), stress_tris[:,d])
    plot_tris((pts, tris), stress_tris[:,3], 'rectsxy')
    plot_tris((pts, tris), stress_tris[:,5], 'rectsyz')



if __name__ == "__main__":
    main()
    tri_main()

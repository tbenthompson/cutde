import os
import numpy as np
import matplotlib.pyplot as plt

import cutde.fullspace

def check_folder(f):
    if not os.path.exists(f):
        os.makedirs(f)

def two_rects_mesh(n, corners1, corners2):
    face1_pts, face1_rects = make_rect_of_rects(n, n, corners1)
    face2_pts, face2_rects = make_rect_of_rects(n, n, corners2)
    pts = np.vstack((face1_pts, face2_pts))
    offset_face2_rects = face2_rects + face1_pts.shape[0]
    rects = np.vstack((face1_rects, offset_face2_rects))
    return pts, rects

def twist_mesh(n):
    corners1 = [[-2, 0, 1], [-2, 0, -1], [-1, 0, -1], [0, 0, 1]]
    corners2 = [[0, 0, 1], [-1, 0, -1], [2, 2, -1], [2, 2, 1]]
    return two_rects_mesh(n, corners1, corners2)

def bend_mesh(n):
    corners1 = [[-2, 0, 1], [-2, 0, -1], [0, 0, -1], [0, 0, 1]]
    corners2 = [[0, 0, 1], [0, 0, -1], [2, 2, -1], [2, 2, 1]]
    out = two_rects_mesh(n, corners1, corners2)
    return out

def planar_mesh(n):
    # corners = [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]
    corners = [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]]
    return make_rect_of_rects(n, n, corners)

def gaussian(a, b, c, x):
    # return np.ones_like(x)
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

def plot_tris(m, field, name, folder):
    plt.figure(figsize = (15,15))
    plt.title(name)
    plt.tripcolor(m[0][:,0], m[0][:,2], m[1], field)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig(os.path.join(folder, name + '.pdf'))
    # plt.show()

# Corners are ordered: lower left, lower right, upper right, upper left
def rect_points(corners, xhat_vals, yhat_vals):
    nx = xhat_vals.shape[0]
    ny = yhat_vals.shape[0]
    corners = np.array(corners)

    rect_basis = [
        lambda x, y: x * y,
        lambda x, y: (1 - x) * y,
        lambda x, y: (1 - x) * (1 - y),
        lambda x, y: x * (1 - y)
    ]

    X, Y = np.meshgrid(xhat_vals, yhat_vals)
    vertices = np.vstack((X.reshape(nx * ny), Y.reshape(nx * ny))).T

    pts = np.sum([
        np.outer(rect_basis[i](vertices[:,0], vertices[:,1]), corners[i, :])
        for i in range(4)
    ], axis = 0)
    return pts

def rect_topology(nx, ny):
    def v_idx(i, j):
        return j * nx + i

    rects = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            top_left = v_idx(i, j)
            top_right = v_idx(i + 1, j)
            bottom_left = v_idx(i, j + 1)
            bottom_right = v_idx(i + 1, j + 1)
            rects.append([top_left, bottom_left, bottom_right, top_right])
    return np.array(rects, dtype = np.int)

def make_rect_of_rects(nx, ny, corners):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    return rect_points(corners, x, y), rect_topology(nx, ny)

def rect_to_tri_mesh(pts, rects):
    tris = []
    for i in range(rects.shape[0]):
        top_left, bottom_left, bottom_right, top_right = rects[i]
        tris.append([top_left, bottom_left, top_right])
        tris.append([bottom_left, bottom_right, top_right])
    return pts, np.array(tris, dtype = np.int)

def build_weird_tri_mesh(pts, rects):
    new_pts = np.empty((pts.shape[0] + rects.shape[0], 3))
    new_pts[:pts.shape[0],:] = pts
    tris = np.empty((rects.shape[0] * 3, 3))
    for i in range(rects.shape[0]):
        top_left, bottom_left, bottom_right, top_right = rects[i]
        top_middle = pts.shape[0] + i
        new_pts[top_middle,:] = (pts[top_left, :] + pts[top_right, :]) / 2.0
        tris[i * 3 + 0,:] = [top_left, bottom_left, top_middle]
        tris[i * 3 + 1,:] = [top_middle, bottom_right, top_right]
        tris[i * 3 + 2,:] = [bottom_left, bottom_right, top_middle]
    return new_pts, np.array(tris, dtype = np.int)

def dofs_to_tris(field, factor = 3):
    n_rects = field.shape[0]
    out = np.tile(field.reshape((n_rects, -1))[:,np.newaxis,:], (1,factor,1))
    out_shape = list(field.shape)
    out_shape[0] *= factor
    return out.reshape(out_shape)

def eval_tris(obs_pts, tri_pts, slip, sm, nu):
    all_strains = cutde.fullspace.clu_strain_all_pairs(obs_pts, tri_pts, slip, nu)
    strain = np.sum(all_strains, axis = 1)
    stress = cutde.fullspace.strain_to_stress(strain, sm, nu)
    return strain, stress

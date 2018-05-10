import numpy as np

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

    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            top_left = v_idx(i, j)
            top_right = v_idx(i + 1, j)
            bottom_left = v_idx(i, j + 1)
            bottom_right = v_idx(i + 1, j + 1)
            tris.append([top_left, bottom_left, top_right])
            tris.append([bottom_left, bottom_right, top_right])
    return np.array(tris, dtype = np.int)

#TODO: Technically, this is make quadrilateral!
def make_rect(nx, ny, corners):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    return rect_points(corners, x, y), rect_topology(nx, ny)

import matplotlib.pyplot as plt
import numpy as np

import cutde.fullspace as FS

xs = np.linspace(-2, 2, 200)
ys = np.linspace(-2, 2, 200)
obsx, obsy = np.meshgrid(xs, ys)
pts = np.array([obsx, obsy, 0 * obsy]).reshape((3, -1)).T.copy()

fault_pts = np.array([[-1, 0, 0], [1, 0, 0], [1, 0, -1], [-1, 0, -1]])
fault_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

disp_mat = FS.disp_matrix(obs_pts=pts, tris=fault_pts[fault_tris], nu=0.25)

slip = np.array([[1, 0, 0], [1, 0, 0]])
disp = disp_mat.reshape((-1, 6)).dot(slip.flatten())

disp_grid = disp.reshape((*obsx.shape, 3))

plt.figure(figsize=(5, 5), dpi=300)
cntf = plt.contourf(obsx, obsy, disp_grid[:, :, 0], levels=21)
plt.contour(
    obsx,
    obsy,
    disp_grid[:, :, 0],
    colors="k",
    linestyles="-",
    linewidths=0.5,
    levels=21,
)
plt.colorbar(cntf)
plt.title("$u_x$")
plt.tight_layout()
plt.savefig("docs/example.png", bbox_inches="tight")

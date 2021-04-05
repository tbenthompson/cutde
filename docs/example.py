import matplotlib.pyplot as plt
import numpy as np

import cutde

xs = np.linspace(-2, 2, 200)
ys = np.linspace(-2, 2, 200)
obsx, obsy = np.meshgrid(xs, ys)
pts = np.array([obsx, obsy, 0 * obsy]).reshape((3, -1)).T.copy()

fault_pts = np.array([[-1, 0, 0], [1, 0, 0], [1, 0, -1], [-1, 0, -1]])
fault_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

#
slip = np.array([[1, 0, 0], [1, 0, 0]])

disp_mat = cutde.disp_all_pairs(
    obs_pts=pts, tris=fault_pts[fault_tris], slips=slip, nu=0.25
)
disp = np.sum(disp_mat, axis=1).reshape((*obsx.shape, 3))

plt.figure(figsize=(5, 5), dpi=300)
cntf = plt.contourf(obsx, obsy, disp[:, :, 0], levels=21)
plt.contour(
    obsx, obsy, disp[:, :, 0], colors="k", linestyles="-", linewidths=0.5, levels=21
)
plt.colorbar(cntf)
plt.title("$u_x$")
plt.tight_layout()
plt.savefig("docs/example.png", bbox_inches="tight")

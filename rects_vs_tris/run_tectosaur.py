# TODO: stress must be continuous even if traction isn't
# TODO: Is there some kind of asymmetry about the constraining. Is the "test function" a displacement or a traction? Oh wow. This is the key here.
# TODO: do what is necessary for getting hanging nodes to work!

# Int phi * t = Int phi * Int H u
# u should be continuous --> this is true because it's an input
# phi should be continuous because the trial functions represent displacements
# I THINK THIS RESULTS IN A NON-SQUARE MATRIX!
# t should not be continuous

import os
import numpy as np
import matplotlib.pyplot as plt
import common
import common_tectosaur
import logging
from tectosaur.mesh.modify import remove_duplicate_pts
from tectosaur.mesh.refine import refine_to_size, refine

n = 40
gauss_params = (1.0, 0.0, 0.3)
gauss_center = (0.0, 0.0, 0.0)

which = 'planar'
folder = f'results/{which}_tectosaur_dbem'
common.check_folder(folder)

def main():
    # abc = remove_duplicate_pts(common.rect_to_tri_mesh(*common.twist_mesh(10)))

    pts, rects = getattr(common, which + '_mesh')(n)
    pts, tris = common.rect_to_tri_mesh(pts, rects)
    pts, tris = remove_duplicate_pts((pts, tris))
    # pts, tris = refine((pts, tris))
    # pts, tris = refine_to_size((pts, tris), 0.02)[0]
    # np.save('mesh.npy', (pts, tris))
    # plt.triplot(pts[:,0], pts[:,2], tris)
    # plt.show()
    # plt.triplot(pts[:,0], pts[:,1], tris)
    # plt.show()

    n_tris = tris.shape[0]
    tri_pts = pts[tris].reshape((-1,3))
    dist = np.linalg.norm(tri_pts - gauss_center, axis = 1)
    strike_slip = common.gaussian(*gauss_params, dist)
    slip = np.zeros((strike_slip.shape[0], 3))
    slip[:,0] = strike_slip

    common_tectosaur.plot_tectosaur(
        (pts, tris), slip[:,0].reshape(-1,3), 'inputslip', folder
    )

    slip_to_traction = common_tectosaur.get_slip_to_traction(
        (pts, tris), common_tectosaur.tectosaur_cfg
    )
    traction = slip_to_traction(slip)

    sxy = traction.reshape(-1,3,3)[:,:,0]
    syz = traction.reshape(-1,3,3)[:,:,2]
    common_tectosaur.plot_tectosaur((pts, tris), sxy, 'sxy', folder)
    common_tectosaur.plot_tectosaur((pts, tris), syz, 'syz', folder)
    os.system(f'google-chrome {folder}/*.pdf')


if __name__ == "__main__":
    main()

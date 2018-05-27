import sys
import numpy as np
from mayavi import mlab

points, tris = np.load(sys.argv[1])

mlab.triangular_mesh(
    points[:,0], points[:,1], points[:,2], tris,
    # scalars = dist_center,
    # representation = 'wireframe'
)

mlab.show()

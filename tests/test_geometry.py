import sys

import numpy as np

from cutde.geometry import compute_efcs_to_tdcs_rotations, compute_projection_transforms


def test_efcs_to_tdcs_orthogonal():
    tp = np.random.random_sample((1, 3, 3))
    R = compute_efcs_to_tdcs_rotations(tp)
    for d1 in range(3):
        for d2 in range(d1, 3):
            c = np.sum(R[:, d1, :] * R[:, d2, :], axis=1)
            true = 1 if d1 == d2 else 0
            np.testing.assert_almost_equal(c, true)


def test_projection_transforms():
    if sys.version_info.minor <= 7:
        print(
            "Skipping test_projection_transforms for Python <= 3.7"
            " due to buggy conda packages."
        )
        return
    from pyproj import Transformer

    transformer = Transformer.from_crs(
        "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        "+proj=geocent +datum=WGS84 +units=m +no_defs",
    )
    rand_pt = np.random.random_sample(3)
    pts = np.array([rand_pt, rand_pt + np.array([0, 0, -1000.0])])
    transforms = compute_projection_transforms(pts, transformer)

    # The vertical component should transform without change in scale.
    np.testing.assert_almost_equal(np.linalg.norm(transforms[:, :, 2], axis=1), 1.0)

    # The transformation should be identical for vertical motions
    np.testing.assert_allclose(
        transforms[0, :, [0, 1]], transforms[1, :, [0, 1]], rtol=1e-2
    )

    # Check length of degree of latitude and longitude. Approximate.
    np.testing.assert_allclose(transforms[0, 1, 0], 1.11e5, rtol=5e-3)
    np.testing.assert_allclose(transforms[0, 2, 1], 1.11e5, rtol=5e-3)

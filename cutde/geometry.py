import numpy as np


def strain_to_stress(strain, mu, nu):
    """
    Compute stress given strain like:
        stress = 2 * mu * strain + lambda * Id(3) * trace(strain)

    Parameters
    ----------
    strain : {array-like}, shape (n_tensors, 6)
        The strain tensors ordered like (e_xx, e_yy, e_zz, e_xy, e_xz, e_yz)
    mu : float
        Shear modulus
    nu : float
        Poisson ratio

    Returns
    -------
    stress : np.ndarray, shape (n_tensors, 6)
        The stress tensors ordered like (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)
    """
    lam = 2 * mu * nu / (1 - 2 * nu)
    trace = np.sum(strain[:, :3], axis=1)
    stress = np.empty_like(strain)
    stress[:, :3] = 2 * mu * strain[:, :3] + lam * trace[:, np.newaxis]
    stress[:, 3:] = 2 * mu * strain[:, 3:]
    return stress


def compute_normal_vectors(tri_pts) -> np.ndarray:
    """
    Compute normal vectors for each triangle.

    Parameters
    ----------
    tri_pts : {array-like}, shape (n_triangles, 3, 3)
        The vertices of the triangles.

    Returns
    -------
    normals : np.ndarray, shape (n_triangles, 3)
        The normal vectors for each triangle.
    """
    leg1 = tri_pts[:, 1] - tri_pts[:, 0]
    leg2 = tri_pts[:, 2] - tri_pts[:, 0]
    # The normal vector is one axis of the TDCS and can be
    # computed as the cross product of the two corner-corner tangent vectors.
    Vnormal = np.cross(leg1, leg2, axis=1)
    # And it should be normalized to have unit length of course!
    Vnormal /= np.linalg.norm(Vnormal, axis=1)[:, None]
    return Vnormal


def compute_projection_transforms(origins, transformer) -> np.ndarray:
    """
    Convert vectors from one coordinate system to another. Unlike positions,
    this cannot be done with a simple pyproj call. We first need to set up a
    vector start and end point, convert those into the new coordinate system
    and then recompute the direction/distance between the start and end point.

    The output matrices are not pure rotation matrices because there is also
    a scaling of vector lengths. For example, converting from latitude to
    meters will result in a large scale factor.

    You can obtain the inverse transformation either by computing the inverse
    of the matrix or by passing an inverse pyproj.Transformer.

    Parameters
    ----------
    origins : {array-like}, shape (N, 3)
        The points at which we will compute rotation matrices
    transformer : pyproj.Transformer
        A pyproj.Transformer that will perform the necessary projection step.

    Returns
    -------
    transform_mats : np.ndarray, shape (n_triangles, 3, 3)
        The 3x3 rotation and scaling matrices that transform vectors from the
        EFCS to TDCS.
    """

    out = np.empty((origins.shape[0], 3, 3), dtype=origins.dtype)
    for d in range(3):
        eps = 1.0
        targets = origins.copy()
        targets[:, d] += eps
        proj_origins = np.array(
            transformer.transform(origins[:, 0], origins[:, 1], origins[:, 2])
        ).T.copy()
        proj_targets = np.array(
            transformer.transform(targets[:, 0], targets[:, 1], targets[:, 2])
        ).T.copy()
        out[:, :, d] = proj_targets - proj_origins
    return out


def compute_efcs_to_tdcs_rotations(tri_pts) -> np.ndarray:
    """
    Build rotation matrices that convert from an Earth-fixed coordinate system
    (EFCS) to a triangular dislocation coordinate system (TDCS).

    In the EFCS, the vectors will be directions/length in a map projection or
    an elliptical coordinate system.

    In the TDCS, the coordinates/vectors will be separated into:
    `(along-strike-distance, along-dip-distance, tensile-distance)`

    Note that in the Nikhoo and Walter 2015 and the Okada convention, the dip
    vector points upwards. This is different from the standard geologic
    convention where the dip vector points downwards.

    It may be useful to extract normal, dip or strike vectors from the rotation
    matrices that are returned by this function. The strike vectors are:
    `rot_mats[:, 0, :]`, the dip vectors are `rot_mats[:, 1, :]` and the normal
    vectors are `rot_mats[:, 2, :]`.

    To transform from TDCS back to EFCS, we simply need the transpose of the
    rotation matrices because the inverse of an orthogonal matrix is its
    transpose. To get this you can run `np.transpose(rot_mats, (0, 2, 1))`.

    Parameters
    ----------
    tri_pts : {array-like}, shape (n_triangles, 3, 3)
        The vertices of the triangles.

    Returns
    -------
    rot_mats : np.ndarray, shape (n_triangles, 3, 3)
        The 3x3 rotation matrices that transform vectors from the EFCS to TDCS.
    """
    Vnormal = compute_normal_vectors(tri_pts)
    eY = np.array([0, 1, 0])
    eZ = np.array([0, 0, 1])
    # The strike vector is defined as orthogonal to both the (0,0,1) vector and
    # the normal.
    Vstrike_raw = np.cross(eZ[None, :], Vnormal, axis=1)
    Vstrike_length = np.linalg.norm(Vstrike_raw, axis=1)

    # If eZ == Vnormal, we will get Vstrike = (0,0,0). In this case, just set
    # Vstrike equal to (0,±1,0).
    Vstrike = np.where(
        Vstrike_length[:, None] == 0, eY[None, :] * Vnormal[:, 2, None], Vstrike_raw
    )
    Vstrike /= np.linalg.norm(Vstrike, axis=1)[:, None]
    Vdip = np.cross(Vnormal, Vstrike, axis=1)
    return np.transpose(np.array([Vstrike, Vdip, Vnormal]), (1, 0, 2))

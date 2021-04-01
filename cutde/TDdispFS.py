# Modified from code presented in:
# Nikkhoo, M., Walter, T. R. (2015): Triangular dislocation: an analytical,
# artefact-free solution. - Geophysical Journal International, 201,
# 1117-1139. doi: 10.1093/gji/ggv035

# Original documentation:
# TDdispFS
# calculates displacements associated with a triangular dislocation in an
# elastic full-space.
#
# TD: Triangular Dislocation
# EFCS: Earth-Fixed Coordinate System
# TDCS: Triangular Dislocation Coordinate System
# ADCS: Angular Dislocation Coordinate System
#
# INPUTS
# X, Y and Z:
# Coordinates of calculation points in EFCS (East, North, Up). X, Y and Z
# must have the same size.
#
# P1,P2 and P3:
# Coordinates of TD vertices in EFCS.
#
# Ss, Ds and Ts:
# TD slip vector components (Strike-slip, Dip-slip, Tensile-slip).
#
# nu:
# Poisson's ratio.
#
# OUTPUTS
# ue, un and uv:
# Calculated displacement vector components in EFCS. ue, un and uv have
# the same unit as Ss, Ds and Ts in the inputs.
#
# Original documentation license:
# Copyright (c) 2014 Mehdi Nikkhoo
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np


def build_tri_coordinate_system(tri):
    Vnorm = normalize(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    eY = np.array([0, 1, 0])
    eZ = np.array([0, 0, 1])
    Vstrike = np.cross(eZ, Vnorm)
    if np.linalg.norm(Vstrike) == 0:
        Vstrike = eY * Vnorm[2]
    Vstrike = normalize(Vstrike)
    Vdip = np.cross(Vnorm, Vstrike)
    return np.array([Vnorm, Vstrike, Vdip])


def normalize(v):
    return v / np.linalg.norm(v)


def trimodefinder(obs, tri):
    # trimodefinder calculates the normalized barycentric coordinates of
    # the points with respect to the TD vertices and specifies the appropriate
    # artefact-free configuration of the angular dislocations for the
    # calculations. The input matrices x, y and z share the same size and
    # correspond to the y, z and x coordinates in the TDCS, respectively. p1,
    # p2 and p3 are two-component matrices representing the y and z coordinates
    # of the TD vertices in the TDCS, respectively.
    # The components of the output (trimode) corresponding to each calculation
    # points, are 1 for the first configuration, -1 for the second
    # configuration and 0 for the calculation point that lie on the TD sides.

    a = (
        (tri[1, 1] - tri[2, 1]) * (obs[0] - tri[2, 0])
        + (tri[2, 0] - tri[1, 0]) * (obs[1] - tri[2, 1])
    ) / (
        (tri[1, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0])
        + (tri[2, 0] - tri[1, 0]) * (tri[0, 1] - tri[2, 1])
    )
    b = (
        (tri[2, 1] - tri[0, 1]) * (obs[0] - tri[2, 0])
        + (tri[0, 0] - tri[2, 0]) * (obs[1] - tri[2, 1])
    ) / (
        (tri[1, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0])
        + (tri[2, 0] - tri[1, 0]) * (tri[0, 1] - tri[2, 1])
    )
    c = 1 - a - b

    result = 1
    if (
        (a <= 0 and b > c and c > a)
        or (b <= 0 and c > a and a > b)
        or (c <= 0 and a > b and b > c)
    ):
        result = -1
    if (
        (a == 0 and b >= 0 and c >= 0)
        or (a >= 0 and b == 0 and c >= 0)
        or (a >= 0 and b >= 0 and c == 0)
    ):
        result = 0
    if result == 0 and obs[2] != 0:
        result = 1
    return result


def TDSetupD(obs, alpha, slip_b, nu, TriVertex, SideVec):
    # TDSetupD transforms coordinates of the calculation points as well as
    # slip vector components from ADCS into TDCS. It then calculates the
    # displacements in ADCS and transforms them into TDCS.

    # Transformation matrix
    A = np.array([[SideVec[2], -SideVec[1]], [SideVec[1], SideVec[2]]])

    # Transform coordinates of the calculation points from TDCS into ADCS
    r1 = A.dot([obs[1] - TriVertex[1], obs[2] - TriVertex[2]])
    y1 = r1[0]
    z1 = r1[1]

    # Transform the in-plane slip vector components from TDCS into ADCS
    r2 = A.dot([slip_b[1], slip_b[2]])
    by1 = r2[0]
    bz1 = r2[1]

    # Calculate displacements associated with an angular dislocation in ADCS
    [u, v0, w0] = AngDisDisp(obs[0], y1, z1, -np.pi + alpha, slip_b[0], by1, bz1, nu)

    # Transform displacements from ADCS into TDCS
    r3 = A.T.dot([v0, w0])
    v = r3[0]
    w = r3[1]
    return u, v, w


def AngDisDisp(x, y, z, alpha, bx, by, bz, nu):
    # AngDisDisp calculates the "incomplete" displacements (without the
    # Burgers' function contribution) associated with an angular dislocation in
    # an elastic full-space.

    cosA = np.cos(alpha)
    sinA = np.sin(alpha)
    eta = y * cosA - z * sinA
    zeta = y * sinA + z * cosA
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Avoid complex results for the logarithmic terms
    if zeta > r:
        zeta = r
    if z > r:
        z = r

    ux = bx / 8 / np.pi / (1 - nu) * (x * y / r / (r - z) - x * eta / r / (r - zeta))
    vx = (
        bx
        / 8
        / np.pi
        / (1 - nu)
        * (
            eta * sinA / (r - zeta)
            - y * eta / r / (r - zeta)
            + y ** 2 / r / (r - z)
            + (1 - 2 * nu) * (cosA * np.log(r - zeta) - np.log(r - z))
        )
    )
    wx = (
        bx
        / 8
        / np.pi
        / (1 - nu)
        * (
            eta * cosA / (r - zeta)
            - y / r
            - eta * z / r / (r - zeta)
            - (1 - 2 * nu) * sinA * np.log(r - zeta)
        )
    )

    uy = (
        by
        / 8
        / np.pi
        / (1 - nu)
        * (
            x ** 2 * cosA / r / (r - zeta)
            - x ** 2 / r / (r - z)
            - (1 - 2 * nu) * (cosA * np.log(r - zeta) - np.log(r - z))
        )
    )
    vy = (
        by
        * x
        / 8
        / np.pi
        / (1 - nu)
        * (y * cosA / r / (r - zeta) - sinA * cosA / (r - zeta) - y / r / (r - z))
    )
    wy = (
        by
        * x
        / 8
        / np.pi
        / (1 - nu)
        * (z * cosA / r / (r - zeta) - cosA ** 2 / (r - zeta) + 1 / r)
    )

    uz = (
        bz
        * sinA
        / 8
        / np.pi
        / (1 - nu)
        * ((1 - 2 * nu) * np.log(r - zeta) - x ** 2 / r / (r - zeta))
    )
    vz = bz * x * sinA / 8 / np.pi / (1 - nu) * (sinA / (r - zeta) - y / r / (r - zeta))
    wz = bz * x * sinA / 8 / np.pi / (1 - nu) * (cosA / (r - zeta) - z / r / (r - zeta))

    return ux + uy + uz, vx + vy + vz, wx + wy + wz


def TDdispFS(obs, tri, slip, nu):
    transform = build_tri_coordinate_system(tri)

    transformed_obs = transform.dot(obs - tri[1])
    transformed_tri = np.zeros((3, 3))
    transformed_tri[0, :] = transform.dot(tri[0] - tri[1])
    transformed_tri[2, :] = transform.dot(tri[2] - tri[1])
    np.testing.assert_almost_equal(transformed_tri[1], [0, 0, 0])
    np.testing.assert_almost_equal(transformed_tri[0][0], 0)
    np.testing.assert_almost_equal(transformed_tri[2][0], 0)

    e12 = normalize(transformed_tri[1] - transformed_tri[0])
    e13 = normalize(transformed_tri[2] - transformed_tri[0])
    e23 = normalize(transformed_tri[2] - transformed_tri[1])

    A = np.arccos(e12.T.dot(e13))
    B = np.arccos(-e12.T.dot(e23))
    C = np.arccos(e23.T.dot(e13))

    mode = trimodefinder(
        np.array([transformed_obs[1], transformed_obs[2], transformed_obs[0]]),
        transformed_tri[:, 1:],
    )

    slip_b = np.array([slip[2], slip[0], slip[1]])
    if mode == 1:
        # Calculate first angular dislocation contribution
        u1Tp, v1Tp, w1Tp = TDSetupD(
            transformed_obs, A, slip_b, nu, transformed_tri[0], -e13
        )
        # Calculate second angular dislocation contribution
        u2Tp, v2Tp, w2Tp = TDSetupD(
            transformed_obs, B, slip_b, nu, transformed_tri[1], e12
        )
        # Calculate third angular dislocation contribution
        u3Tp, v3Tp, w3Tp = TDSetupD(
            transformed_obs, C, slip_b, nu, transformed_tri[2], e23
        )
        out = np.array([u1Tp + u2Tp + u3Tp, v1Tp + v2Tp + v3Tp, w1Tp + w2Tp + w3Tp])
    elif mode == -1:
        # Calculate first angular dislocation contribution
        u1Tn, v1Tn, w1Tn = TDSetupD(
            transformed_obs, A, slip_b, nu, transformed_tri[0], e13
        )
        # Calculate second angular dislocation contribution
        u2Tn, v2Tn, w2Tn = TDSetupD(
            transformed_obs, B, slip_b, nu, transformed_tri[1], -e12
        )
        # Calculate third angular dislocation contribution
        u3Tn, v3Tn, w3Tn = TDSetupD(
            transformed_obs, C, slip_b, nu, transformed_tri[2], -e23
        )
        out = np.array([u1Tn + u2Tn + u3Tn, v1Tn + v2Tn + v3Tn, w1Tn + w2Tn + w3Tn])
    else:
        out = np.array([np.nan, np.nan, np.nan])

    a = np.array(
        [
            -transformed_obs[0],
            transformed_tri[0][1] - transformed_obs[1],
            transformed_tri[0][2] - transformed_obs[2],
        ]
    )
    b = -transformed_obs
    c = np.array(
        [
            -transformed_obs[0],
            transformed_tri[2][1] - transformed_obs[1],
            transformed_tri[2][2] - transformed_obs[2],
        ]
    )
    na = np.sqrt(np.sum(a ** 2))
    nb = np.sqrt(np.sum(b ** 2))
    nc = np.sqrt(np.sum(c ** 2))

    FiN = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    FiD = na * nb * nc + np.sum(a * b) * nc + np.sum(a * c) * nb + np.sum(b * c) * na
    Fi = -2 * np.arctan2(FiN, FiD) / 4 / np.pi

    # Calculate the complete displacement vector components in TDCS
    out += slip_b * Fi

    # Transform the complete displacement vector components from TDCS into EFCS
    return transform.T.dot(out)

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

import tectosaur
from tectosaur.mesh.modify import remove_duplicate_pts
from tectosaur.mesh.refine import refine_to_size, refine
from tectosaur.ops.sparse_integral_op import make_integral_op
from tectosaur.constraint_builders import free_edge_constraints, \
    continuity_constraints, build_composite_constraints
from tectosaur.constraints import build_constraint_matrix, Term, ConstraintEQ
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp
from tectosaur.ops.neg_op import NegOp
from tectosaur.ops.composite_op import CompositeOp
from tectosaur_topo.solve import iterative_solve
from tectosaur_topo.assemble import prec_spilu

n = 20
gauss_params = (1.0, 0.0, 0.3)
gauss_center = (0.0, 0.0, 0.0)

which = 'planar'
folder = f'results/{which}_tectosaur_dbem2'
common.check_folder(folder)


def build_mesh():
    # abc = remove_duplicate_pts(common.rect_to_tri_mesh(*common.twist_mesh(10)))
    pts, rects = getattr(common, which + '_mesh')(n)
    pts, tris = common.rect_to_tri_mesh(pts, rects)
    pts, tris = remove_duplicate_pts((pts, tris))
    # pts, tris = refine((pts, tris))
    # pts, tris = refine_to_size((pts, tris), 0.02)[0]
    np.save('mesh.npy', (pts, tris))
    # plt.triplot(pts[:,0], pts[:,2], tris)
    # plt.show()
    # plt.triplot(pts[:,0], pts[:,1], tris)
    # plt.show()
    return pts, tris

def get_slip_field(pts, tris):
    n_tris = tris.shape[0]
    tri_pts = pts[tris].reshape((-1,3))
    dist = np.linalg.norm(tri_pts - gauss_center, axis = 1)
    strike_slip = common.gaussian(*gauss_params, dist)
    slip = np.zeros((strike_slip.shape[0], 3))
    slip[:,0] = strike_slip
    return slip.flatten()

def build_ops(pts, tris):
    tectosaur_cfg = common_tectosaur.tectosaur_cfg
    # tectosaur.logger.setLevel(tectosaur_cfg['log_level'])
    # tectosaur_topo.logger.setLevel(tectosaur_cfg['log_level'])
    all_tri_idxs = np.arange(tris.shape[0])
    ops = dict()
    for K in ['U', 'T', 'A', 'H']:
        ops[K] = make_integral_op(
            pts, tris, 'elastic' + K + '3',
            [tectosaur_cfg['sm'], tectosaur_cfg['pr']],
            tectosaur_cfg, all_tri_idxs, all_tri_idxs
        )
    ops['M'] = MassOp(tectosaur_cfg['quad_mass_order'], pts, tris)
    return ops

def jump_constraints(jump, negative):
    n_dofs_per_side = jump.shape[0]
    cs = []
    coeff_2 = 1.0 if negative else -1.0
    for i in range(n_dofs_per_side):
        dof_1 = i
        dof_2 = i + n_dofs_per_side
        ts = []
        ts.append(Term(1.0, dof_1))
        ts.append(Term(coeff_2, dof_2))
        cs.append(ConstraintEQ(ts, jump[i]))
    return cs

# Galerkin dual boundary element method
def gdbem(ops):
    S = SumOp
    N = NegOp
    U = ops['U']
    T = ops['T']
    A = ops['A']
    H = ops['H']
    M = ops['M']

    # VERSION 2: skip two rows since they are redundant given the constraints
    # on side A: apply displacement integral equation
    # on side B: apply traction integral equation

    # T and A flip sign with normal vector flip
    # on side A -- eq1 = K_U*t_A + K_U*t_B - (K_T+M)*u_A + K_T*u_B
    # on side B -- eq2 = K_A*t_A + (K_A+M)*t_B - K_H*u_A + K_H*u_B
    #
    # v = [
    #    t_A, t_B, u_A, u_B
    # ]
    mat = [
        [N(U), N(U), S([T,M]), N(T)],
        [0,0,0,0],
        [0,0,0,0],
        [N(A), S([N(A),M]), N(H), H],
    ]
    return mat

def build_system(pts, tris, ops, slip):
    trac_cs = []
    trac_cs.extend(jump_constraints(np.zeros_like(slip), True))

    disp_cs = []
    disp_cs.extend(continuity_constraints(tris, np.array([])))
    disp_cs.extend(jump_constraints(slip, False))

    ND = tris.shape[0] * 9
    n_total_dofs = ND * 4
    cs = build_composite_constraints(
        (trac_cs, 0), (disp_cs, 2 * ND)
    )
    cm, c_rhs = build_constraint_matrix(cs, n_total_dofs)

    chunk_mat = gdbem(ops)
    ops_and_starts = []
    for i in range(4):
        for j in range(4):
            chunk = chunk_mat[i][j]
            if chunk == 0:
                continue
            ops_and_starts.append((chunk, i * ND, j * ND))
    lhs = CompositeOp(*ops_and_starts)
    rhs = -lhs.dot(c_rhs)
    return cm, lhs, rhs

def solve_system(system):
    cm, lhs, rhs = system
    cfg = dict(solver_tol = 1e-3)
    prec = lambda x: x
    out = iterative_solve(lhs, cm, rhs, prec, cfg)
    return out.reshape(4, -1, 3)[0, :, :]

def main():
    pts, tris = build_mesh()
    slip = get_slip_field(pts, tris)
    common_tectosaur.plot_tectosaur(
        (pts, tris), slip.reshape(-1,3)[:,0].reshape(-1,3), 'inputslip', folder
    )
    ops = build_ops(pts, tris)
    system = build_system(pts, tris, ops, slip)
    traction = solve_system(system)

    sxy = traction.reshape(-1,3,3)[:,:,0]
    syz = traction.reshape(-1,3,3)[:,:,2]
    common_tectosaur.plot_tectosaur((pts, tris), sxy, 'sxy', folder)
    common_tectosaur.plot_tectosaur((pts, tris), syz, 'syz', folder)
    os.system(f'google-chrome {folder}/*.pdf')


if __name__ == "__main__":
    main()

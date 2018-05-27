import os
import logging
import numpy as np
import matplotlib.pyplot as plt


tectosaur_cfg = dict(
    quad_mass_order = 4,
    quad_vertadj_order = 8,
    quad_far_order = 3,
    quad_near_order = 5,
    quad_near_threshold = 2.5,
    float_type = np.float32,
    use_fmm = False,
    fmm_order = 150,
    fmm_mac = 3.0,
    pts_per_cell = 450,
    log_level = logging.INFO,
    sm = 1.0,
    pr = 0.25
)

def traction_continuity_constraints(pts, surface_tris, fault_tris, tensor_dim = 3):
    from tectosaur.constraints import ConstraintEQ, Term
    from tectosaur.constraint_builders import find_touching_pts
    from tectosaur.util.geometry import unscaled_normals
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]

    touching_pt = find_touching_pts(surface_tris)
    if fault_tris.shape[0] > 0:
        fault_touching_pt = find_touching_pts(fault_tris)
    else:
        fault_touching_pt = []
    fault_touching_pt.extend(
        [[] for i in range(len(touching_pt) - len(fault_touching_pt))]
    )
    constraints = []

    normals = unscaled_normals(pts[surface_tris])
    normals /= np.linalg.norm(normals, axis = 1)[:, np.newaxis]

    jj = 0
    for i, tpt in enumerate(touching_pt):
        if len(tpt) == 0:
            continue

        for independent_idx in range(len(tpt)):
            independent = tpt[independent_idx]
            independent_tri_idx = independent[0]
            independent_tri = surface_tris[independent_tri_idx]

            for dependent_idx in range(independent_idx + 1, len(tpt)):
                dependent = tpt[dependent_idx]
                dependent_tri_idx = dependent[0]
                dependent_tri = surface_tris[dependent_tri_idx]

                n1 = normals[independent_tri_idx]
                n2 = normals[dependent_tri_idx]
                same_plane = np.all(np.abs(n1 - n2) < 1e-6)
                if not same_plane:
                    jj += 1
                    print('not same plane', jj, n1, n2)
                    continue

                # Check for anything that touches across the fault.
                crosses = (
                    fault_tris.shape[0] > 0
                    and check_if_crosses_fault(
                        independent_tri, dependent_tri, fault_touching_pt, fault_tris
                    )
                )

                if crosses:
                    continue

                for d in range(tensor_dim):
                    independent_dof = (independent_tri_idx * 3 + independent[1]) * tensor_dim + d
                    dependent_dof = (dependent_tri_idx * 3 + dependent[1]) * tensor_dim + d
                    if dependent_dof <= independent_dof:
                        continue
                    diff = 0.0
                    constraints.append(ConstraintEQ(
                        [Term(1.0, dependent_dof), Term(-1.0, independent_dof)], diff
                    ))
    return constraints

def get_slip_to_traction(m, tectosaur_cfg):
    import tectosaur
    from tectosaur.ops.sparse_integral_op import make_integral_op
    from tectosaur.constraint_builders import free_edge_constraints, continuity_constraints
    from tectosaur.constraints import build_constraint_matrix
    from tectosaur.ops.mass_op import MassOp
    from scipy.sparse.linalg import spsolve
    pts, tris = m
    tectosaur.logger.setLevel(tectosaur_cfg['log_level'])
    n_dofs = tris.shape[0] * 9
    cs = []
    # cs.extend(traction_continuity_constraints(pts, tris, np.array([])))
    cs = continuity_constraints(tris, np.array([]))
    # cs.extend(free_edge_constraints(tris))
    cm, c_rhs = build_constraint_matrix(cs, n_dofs)
    all_tri_idxs = np.arange(tris.shape[0])
    hypersingular_op = make_integral_op(
        pts, tris, 'elasticH3',
        [tectosaur_cfg['sm'], tectosaur_cfg['pr']],
        tectosaur_cfg, all_tri_idxs, all_tri_idxs
    )
    traction_mass_op = MassOp(tectosaur_cfg['quad_mass_order'], pts, tris)
    constrained_traction_mass_op = cm.T.dot(traction_mass_op.mat.dot(cm))
    def slip_to_traction(slip):
        rhs = hypersingular_op.dot(slip.reshape(-1))
        # rhs = cm.dot(cm.T.dot(hypersingular_op.dot(slip.reshape(-1))))
        out = spsolve(traction_mass_op.mat, rhs)
        # out = cm.dot(spsolve(constrained_traction_mass_op, cm.T.dot(rhs)))
        return out
    return slip_to_traction

def plot_tectosaur(m, field, name, folder, show = False):
    pts, tris = m
    pt_field = np.empty(pts.shape[0])
    pt_field[tris] = field

    plt.figure()
    cmap = 'PuOr'
    plt.tricontourf(
        pts[:,0], pts[:,2], tris,
        pt_field, cmap = cmap,
        # levels = f_levels,
        extend = 'both'
    )
    plt.colorbar()
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, name + '.pdf'))
        # TODO: CLEAR FIGURE?

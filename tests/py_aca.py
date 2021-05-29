"""
This Python ACA implementation is included here in order to help test the
OpenCL/CUDA implementation in cutde.
"""
import numpy as np


def ACA_plus(
    n_rows,
    n_cols,
    calc_rows,
    calc_cols,
    eps,
    max_iter=None,
    verbose=False,
    Iref=None,
    Jref=None,
    vec_dim=3,
):
    """
    Run the ACA+ plus algorithm on a matrix implicitly defined by the
    row and column computation functions passed as arguments.

    :param n_rows:
    :param n_cols:
    :param calc_rows: A function that accepts two parameters (Istart, Iend)
        specifying the first and last row desired and returns a numpy array
        with shape (Iend-Istart, N_col) with the corresponding rows of the
        input matrix
    :param calc_cols: A function that accepts two parameters (Jstart, Jend)
        specifying the first and last column desired and returns a numpy array
        with shape (N_rows, Jend-Jstart) with the corresponding columns of the
        input matrix
    :param eps: The tolerance of the approximation. The convergence condition is
        in terms of the difference in Frobenius norm between the target matrix
        and the approximation
    :param max_iter:
    :param verbose: Should we print information at each iteration. Just included
        for demonstration here.

    :return U_ACA: The left-hand approximation matrix.
    :return V_ACA: The right-hand approximation matrix.
    """

    us = []  # The left vectors of the approximation
    vs = []  # The right vectors of the approximation
    prevIstar = []  # Previously used i^* pivots
    prevJstar = []  # Previously used j^* pivots

    # a quick helper function that will help find the largest entry in
    # an array while excluding some list of `disallowed`  entries.
    def argmax_not_in_list(arr, disallowed):
        arg_sorted = arr.argsort()
        max_idx = arg_sorted.shape[0] - 1
        while True:
            if arg_sorted[max_idx] in disallowed:
                max_idx -= 1
            else:
                break
        return arg_sorted[max_idx]

    # A function that will return a contiguous block of rows of the
    # residual matrix
    def calc_residual_rows(Istart, Iend):
        # First calculate the rows of the original matrix.
        out = calc_rows(Istart, Iend).copy()
        # Then subtract the current terms of the approximation
        for i in range(len(us)):
            out -= us[i][Istart:Iend][:, None] * vs[i][None, :]
        return out

    # See above, except this function calculates a block of columns.
    def calc_residual_cols(Jstart, Jend):
        out = calc_cols(Jstart, Jend).copy()
        for i in range(len(us)):
            out -= vs[i][Jstart:Jend][None, :] * us[i][:, None]
        return out

    # A function for finding a reference row and updating
    # it with respect to the already constructed approximation.
    def reset_reference_row(Iref):
        # When a row gets used in the approximation, we will need to
        # reset to use a different reference row. Just, increment!
        while True:
            Iref = (Iref + vec_dim) % n_rows
            Iref -= Iref % vec_dim
            if Iref not in prevIstar:
                break

        # Grab the "row" (actually three rows corresponding to the
        # x, y, and z components for a single observation point)
        return calc_residual_rows(Iref, Iref + vec_dim), Iref

    # Same function as above but for the reference column
    def reset_reference_col(Jref):
        while True:
            Jref = (Jref + 3) % n_cols
            Jref -= Jref % 3
            if Jref not in prevJstar:
                break

        return calc_residual_cols(Jref, Jref + 3), Jref

    # If we haven't converged before running for max_iter, we'll stop anyway.
    if max_iter is None:
        max_iter = np.min([n_rows, n_cols])
    else:
        max_iter = np.min([n_rows, n_cols, max_iter])

    # Choose our starting random reference row and column.
    # These will get incremented by 3 inside reset_reference_row
    # so pre-subtract that.
    if Iref is None:
        Iref = np.random.randint(n_rows)
    Iref -= vec_dim
    if Jref is None:
        Jref = np.random.randint(n_cols)
    Jref -= 3
    # And collect the corresponding blocks of rows/columns
    RIref, Iref = reset_reference_row(Iref)
    RJref, Jref = reset_reference_col(Jref)
    if verbose:
        print(Iref, Jref)

    # Create a buffer for storing the R_{i^*,j} and R_{i, j^*}
    RIstar = np.zeros(n_cols, dtype=RIref.dtype)
    RJstar = np.zeros(n_rows, dtype=RIref.dtype)

    for k in range(max_iter):
        if verbose:
            print("start iteration", k)
            print("RIref", RIref.flatten()[:5])
            print("RJref", RJref.flatten()[:5])
        # These two lines find the column in RIref with the largest entry
        # (step 1 above).
        maxabsRIref = np.max(np.abs(RIref), axis=0)
        Jstar = argmax_not_in_list(maxabsRIref, prevJstar)

        # And these two find the row in RJref with the largest entry (step 1 above).
        maxabsRJref = np.max(np.abs(RJref), axis=1)
        Istar = argmax_not_in_list(maxabsRJref, prevIstar)

        # Check if we should pivot first based on row or based on column (step 2 above)
        Jstar_val = maxabsRIref[Jstar]
        Istar_val = maxabsRJref[Istar]
        if verbose:
            print(f"pivot guess {Istar}, {Jstar}, {Istar_val}, {Jstar_val}")
        if Istar_val > Jstar_val:
            # If we pivot first on the row, then calculate the corresponding row
            # of the residual matrix.
            RIstar[:] = calc_residual_rows(Istar, Istar + 1)[0]

            # Then find the largest entry in that row vector to identify which
            # column to pivot on. (See step 3 above)
            Jstar = argmax_not_in_list(np.abs(RIstar), prevJstar)

            # Calculate the corresponding residual column!
            RJstar[:] = calc_residual_cols(Jstar, Jstar + 1)[:, 0]
        else:
            # If we pivot first on the column, then calculate the corresponding column
            # of the residual matrix.
            RJstar[:] = calc_residual_cols(Jstar, Jstar + 1)[:, 0]

            # Then find the largest entry in that row vector to identify which
            # column to pivot on.  (See step 3 above)
            Istar = argmax_not_in_list(np.abs(RJstar), prevIstar)

            # Calculate the corresponding residual row!
            RIstar[:] = calc_residual_rows(Istar, Istar + 1)[0]

        # Record the pivot row and column so that we don't re-use them.
        prevIstar.append(Istar)
        prevJstar.append(Jstar)

        # Add the new rank-1 outer product to the approximation (see step 4 above)
        vs.append(RIstar / RIstar[Jstar])
        us.append(RJstar.copy())
        if verbose:
            print("us", us[-1][:5])
            print("vs", vs[-1][:5])

        # How "large" was this update to the approximation?
        step_size = np.sqrt(np.sum(us[-1] ** 2) * np.sum(vs[-1] ** 2))
        if verbose:
            print(
                f"term={k}, "
                f"pivot row={Istar:4d}, pivot col={Jstar:4d}, "
                f"step size={step_size:1.3e}, "
                f"tolerance={eps:1.3e}"
            )

        # The convergence criteria will simply be whether the Frobenius norm of the
        # step is smaller than the user provided tolerance.
        if step_size < eps:
            break

        # We also break here if this is the last iteration to avoid wasting effort
        # updating the reference row/column
        if k == max_iter - 1:
            break

        # If we didn't converge, let's prep the reference residual row and
        # column for the next iteration:

        # If we pivoted on the reference row, then choose a new reference row.
        # Remember that we are using a x,y,z vector "row" or
        # set of 3 rows in an algebraic sense.
        if Iref <= Istar < Iref + vec_dim:
            RIref, Iref = reset_reference_row(Iref)
        else:
            # If we didn't change the reference row of the residual matrix "R",
            # update the row to account for the new components of the approximation.
            RIref -= us[-1][Iref : Iref + vec_dim][:, None] * vs[-1][None, :]

        # If we pivoted on the reference column, then choose a new reference column.
        # Remember that we are using a x,y,z vector "column" or
        # set of 3 columns in an algebraic sense.
        if Jref <= Jstar < Jref + 3:
            RJref, Jref = reset_reference_col(Jref)
        else:
            # If we didn't change the reference column of the residual matrix "R",
            # update the column to account for the new components of the approximation.
            RJref -= vs[-1][Jref : Jref + 3][None, :] * us[-1][:, None]

    # Return the left and right approximation matrices.
    # The approximate is such that:
    # M ~ U_ACA.dot(V_ACA)
    U_ACA = np.array(us, dtype=RIref.dtype).T
    V_ACA = np.array(vs, dtype=RIref.dtype)

    return U_ACA, V_ACA


def SVD_recompress(U_ACA, V_ACA, eps):
    """
    Recompress an ACA matrix approximation via SVD.

    :param U_ACA: The left-hand approximation matrix.
    :param V_ACA: The right-hand approximation matrix.
    :param eps: The tolerance of the approximation. The convergence condition is
        in terms of the difference in Frobenius norm between the target matrix
        and the approximation.

    :return U_SVD: The SVD recompressed left-hand approximation matrix.
    :return V_SVD: The SVD recompressed right-hand approximation matrix.
    """
    UQ, UR = np.linalg.qr(U_ACA)
    VQ, VR = np.linalg.qr(V_ACA.T)
    W, SIG, Z = np.linalg.svd(UR.dot(VR.T))

    frob_K = np.sqrt(np.cumsum(SIG[::-1] ** 2))[::-1]
    r = np.argmax(frob_K < eps)

    U = UQ.dot(W[:, :r] * SIG[:r])
    V = Z[:r, :].dot(VQ.T)
    return U, V

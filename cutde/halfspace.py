from .aca import call_clu_aca
from .coordinators import (
    DISP_HS,
    STRAIN_HS,
    call_clu,
    call_clu_block,
    call_clu_free,
    call_clu_matrix,
)
from .geometry import strain_to_stress  # noqa: F401

DISP_SPEC = DISP_HS
STRAIN_SPEC = STRAIN_HS


def disp(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, DISP_HS)


def strain(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, STRAIN_HS)


def disp_matrix(obs_pts, tris, nu):
    return call_clu_matrix(obs_pts, tris, nu, DISP_HS)


def strain_matrix(obs_pts, tris, nu):
    return call_clu_matrix(obs_pts, tris, nu, STRAIN_HS)


def disp_free(obs_pts, tris, slips, nu):
    return call_clu_free(obs_pts, tris, slips, nu, DISP_HS)


def strain_free(obs_pts, tris, slips, nu):
    return call_clu_free(obs_pts, tris, slips, nu, STRAIN_HS)


def disp_block(obs_pts, tris, obs_start, obs_end, src_start, src_end, nu):
    return call_clu_block(
        obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, DISP_HS
    )


def strain_block(obs_pts, tris, obs_start, obs_end, src_start, src_end, nu):
    return call_clu_block(
        obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, STRAIN_HS
    )


def disp_aca(obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, tol, max_iter):
    return call_clu_aca(
        obs_pts,
        tris,
        obs_start,
        obs_end,
        src_start,
        src_end,
        nu,
        tol,
        max_iter,
        DISP_HS,
    )


def strain_aca(
    obs_pts, tris, obs_start, obs_end, src_start, src_end, nu, tol, max_iter
):
    return call_clu_aca(
        obs_pts,
        tris,
        obs_start,
        obs_end,
        src_start,
        src_end,
        nu,
        tol,
        max_iter,
        STRAIN_HS,
    )

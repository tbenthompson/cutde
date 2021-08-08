from .coordinators import DISP_HS, call_clu
from .geometry import strain_to_stress  # noqa: F401


def disp(obs_pts, tris, slips, nu):
    return call_clu(obs_pts, tris, slips, nu, DISP_HS)

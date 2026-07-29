"""
File that creates the pressure vessel: twwo concetric filament wound shells with a hydrogen annulus between them.

It is conformal to the graphite moderator cone. Refer to F-910093037.

July 2026, J.S. Piaskowy
"""

import numpy as np
import openmc

from ..config import GCRConfig
from .moderator import ModeratorAssembly


def _offset_cone(apex_z, alpha, r2, t, name, boundary_type='transmission'):
    """
    results in a coaxial, co-angular cone with the modrator cone, offset by a perpendicular distance t.

    Needed for shell creation.
    """

    cone = openmc.ZCone(x0=0.0, y0=0.0, z0=apex_z - t / np.sin(alpha),
                        r2=r2, name=name, boundary_type=boundary_type)
    return cone

def build_pressure_vessel(cfg: GCRConfig, materials: dict,
                          mod: ModeratorAssembly,
                          end_curvature_sphere: openmc.Sphere) -> list:

    glass = materials['fibreglass']
    h2 = materials['hydrogen_pv']

    r2, apex, alpha = mod.outer_cone.r2, mod.cone_apex_z, mod.cone_half_angle

    t1 = cfg.pv_standoff
    t2 = t1 + cfg.pv_shell_thickness
    t3 = t2 + cfg.pv_gap_thickness
    t4 = t3 + cfg.pv_shell_thickness

    surface1 = (mod.outer_cone if t1 == 0.0 else _offset_cone(apex, alpha, r2, t1, 'pv_standoff'))
    surface2 = _offset_cone(apex, alpha, r2, t2, 'pv_inner_shell_od')
    surface3 = _offset_cone(apex, alpha, r2, t3, 'pv_outer_shell_id')
    surface4 = _offset_cone(apex, alpha, r2, t4, 'pv_outer_shell-od', boundary_type='vacuum')
    
    def bounded(region):
        """
        Axial bounds, same as in graphite moderator
        """

        excluded = -mod.start_planes[0]

        for plane in mod.start_planes[1:]:
            excluded &= -plane
        region &= ~excluded
        for plane in mod.end_planes:
            region &= -plane
        return region & -end_curvature_sphere

    return [
            openmc.Cell(fill=glass, region=bounded(+surface1 & -surface2), name='pv_inner_shell'),
            openmc.Cell(fill=h2,    region=bounded(+surface2 & -surface3), name='pv_annulus_h2'),
            openmc.Cell(fill=glass, region=bounded(+surface3 & -surface4), name='pv_outer_shell'),
            ]






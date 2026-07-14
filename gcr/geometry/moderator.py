"""The structures OUTSIDE the cavity slots: the graphite cone moderator,
the spherical end-cap moderators, the nozzle assembly, and the vacuum
bounding sphere.

Every builder here is a function of (cfg, materials, cavities, ...) that
RETURNS cells.  Two interface points deserve attention:

* build_moderator takes an optional `tie_rods: TieRods`.  Because the cap
  planes travel inside that object together with the cylinder surfaces,
  the carve is bounded by construction -- the unbounded-carve bug that
  produced undefined graphite voids cannot be re-created through this
  interface.

* build_end_moderator / build_nozzle_end WRITE the cells they create back
  onto each Cavity (end_beo_cell, nozzle_beo_cell, ...).  That is a
  deliberate, documented mutation: resolve_cavity_overlaps later needs to
  trim exactly those cells, and the Cavity dataclass declares the fields
  explicitly so nothing appears by surprise.
"""

import numpy as np
import openmc

from ..config import GCRConfig
from .hexmaths import bounding_sphere_offset
from .tie_rods import TieRods


def create_bounding_sphere(cfg: GCRConfig, offset: float = None) -> openmc.Sphere:
    """The vacuum sphere that closes the geometry at the nozzle end.

    `offset` defaults to the value the placement geometry implies
    (hexmaths.bounding_sphere_offset), which is what main() always passed.
    """
    if offset is None:
        offset = bounding_sphere_offset(cfg)
    R_curv = cfg.L + cfg.L_conv - offset
    return openmc.Sphere(
        x0=0.0, y0=0.0,
        z0=cfg.L + cfg.L_conv + 2.0 + 15.0 - R_curv,
        r=R_curv + 15,
        boundary_type='vacuum',
    )


def build_moderator(cfg: GCRConfig, materials: dict, cavities: list,
                    end_curvature_sphere: openmc.Sphere,
                    tie_rods: TieRods = None,
                    inner_radius: float = None) -> list:
    """The big graphite cone around all seven cavity slots.

    Constructed as (cone interior) minus (each transformed hexagonal slot)
    minus (region above all the cavity top planes) minus (bounded tie-rod
    regions, if rods were built).
    """
    graphite = materials['graphite']
    R_in = inner_radius if inner_radius is not None else cfg.moderator_cone_inner_radius

    alpha = 3 * cfg.tilt
    L = cfg.L
    L_conv = cfg.L_conv
    t_mod_top = cfg.moderator_top_thickness

    H = L + L_conv + t_mod_top
    R_out = np.abs(np.tan(alpha)) * H + R_in

    h = H * (1 / (R_out / R_in - 1))
    GraphiteOuter = openmc.ZCone(x0=0, y0=0, z0=-h, r2=(R_in / h) ** 2,
                                 name='Graphite Moderator Cone', boundary_type='vacuum')
    # These two planes are created for their vacuum boundary side effect and
    # as transform templates below (as in the original).
    openmc.ZPlane(z0=-t_mod_top, boundary_type='vacuum', name='PlaneStartMod')
    PlaneNozzleEnd = openmc.ZPlane(z0=L + L_conv + 10.0, boundary_type='vacuum',
                                   name='PlaneNozzleEnd')

    GraphiteRegion = -GraphiteOuter

    EndPlanes = []
    StartPlanes = []

    # Exclude each cavity's transformed hexagonal slot
    for cavity in cavities:
        R = cavity.rotation
        translation = cavity.translation
        plane_start = cavity.plane_moderator_top_start

        cavity_hexagon = -cavity.hex_planes[0]
        for plane in cavity.hex_planes[1:]:
            cavity_hexagon &= -plane
        cavity_hexagon &= +plane_start

        EndPlanes.append(PlaneNozzleEnd.rotate(R, pivot=(0.0, 0.0, 0.0)).translate(translation))
        StartPlanes.append(plane_start.rotate(R, pivot=(0.0, 0.0, 0.0)).translate(translation))

        cavity_hexagon_transformed = (cavity_hexagon
                                      .rotate(R, pivot=(0.0, 0.0, 0.0))
                                      .translate(translation))
        cavity_hexagon_transformed &= -end_curvature_sphere
        GraphiteRegion &= ~cavity_hexagon_transformed

    GraphiteRegion &= -end_curvature_sphere

    for plane in EndPlanes:
        GraphiteRegion &= -plane

    ExcludedRegion = -StartPlanes[0]
    for plane in StartPlanes[1:]:
        ExcludedRegion &= -plane
    GraphiteRegion &= ~ExcludedRegion

    # Exclude tie rods -- BOUNDED, using the caps that arrive with the
    # surfaces inside the TieRods object.  (The old code carved +outer only,
    # i.e. an infinite cylinder, leaving undefined voids past the rod ends.)
    if tie_rods is not None:
        for outer_surf, (plane_bot, plane_top) in zip(tie_rods.outer_surfaces,
                                                      tie_rods.axial_bounds):
            GraphiteRegion &= (+outer_surf | -plane_bot | +plane_top)

    moderator = openmc.Cell(fill=graphite, region=GraphiteRegion,
                            name='graphite moderator')
    return [moderator]


def build_end_moderator(cfg: GCRConfig, materials: dict, cavities: list) -> list:
    """Spherical BeO/graphite end caps behind each cavity's fuel zone.

    Also stores the created cells on each Cavity (end_beo_cell,
    end_graphite_cell) for later trimming by resolve_cavity_overlaps.
    """
    graphite = materials['graphite']
    BeO = materials['BeO']

    L = cfg.L
    EndCapSphere = openmc.Sphere(x0=0.0, y0=0.0, z0=L, r=cfg.R3 + cfg.d_tw_outer / 2)
    EndCapMaterialDivide = openmc.ZPlane(z0=L + 8)

    cells = []
    memo = {}

    for cavity in cavities:
        R = cavity.rotation
        translation = cavity.translation
        tx, ty, tz = translation
        PlaneFuelZoneEnd = cavity.plane_fuel_zone_end

        BeOEndCap = -EndCapSphere & +PlaneFuelZoneEnd & -EndCapMaterialDivide
        GraphiteEndCap = -EndCapSphere & +PlaneFuelZoneEnd & +EndCapMaterialDivide

        BeOTransformed = (BeOEndCap.rotate(R, pivot=(0.0, 0.0, 0.0), memo=memo)
                          .translate(translation, memo=memo))
        GraphiteTransformed = (GraphiteEndCap.rotate(R, pivot=(0.0, 0.0, 0.0), memo=memo)
                               .translate(translation, memo=memo))

        beo_cell = openmc.Cell(fill=BeO, region=BeOTransformed,
                               name=f'end_BeO cav({tx:.1f},{ty:.1f},{tz:.1f})')
        graphite_cell = openmc.Cell(fill=graphite, region=GraphiteTransformed,
                                    name=f'end_graphite cav({tx:.1f},{ty:.1f},{tz:.1f})')

        cavity.end_beo_cell = beo_cell
        cavity.end_graphite_cell = graphite_cell
        cells += [beo_cell, graphite_cell]

    return cells


def build_nozzle_end(cfg: GCRConfig, materials: dict, cavities: list,
                     layered=None, r_throat: float = None,
                     tie_rods: TieRods = None) -> list:
    """Convergent-divergent nozzle assembly behind each cavity.

    The nozzle hydrogen takes the LAST propellant layer's material when the
    model is layered (outlet conditions), else the canonical hydrogen.

    If tie rods were built, their bounded regions are carved from the
    nozzle BeO cells: the rod cap planes are perpendicular to the TILTED
    rod axis (not to z), so a thin lens of each rod cell (~ r_outer *
    tan(~4 deg) ~ 2 mm) pokes past z = L into the nozzle moderator.
    Without this carve, rod cell and nozzle BeO cell would both claim
    that lens.  (The nozzle H2 cells need no carve: the ridges sit at
    ~44 cm from each cavity axis, outside the r_outlet ~ 40 cm cone.)
    """
    if layered is not None and layered.n_layers > 1:
        hydrogen_nozzle = layered.h2_layers[-1]
    else:
        hydrogen_nozzle = materials['hydrogen']
    BeO = materials['BeO']

    if r_throat is None:
        r_throat = cfg.nozzle_throat_radius

    L = cfg.L
    L_conv = cfg.L_conv
    r_outlet = cfg.r_outlet

    PlaneNozzleStart = openmc.ZPlane(z0=L, name='PlaneNozzleStart')
    PlaneThroat = openmc.ZPlane(z0=L + L_conv, name='PlaneThroat')
    PlaneNozzleEnd = openmc.ZPlane(z0=L + L_conv + 1.0, name='PlaneNozzleEnd')
    PlaneEndModerator = openmc.ZPlane(z0=L + L_conv + 10.0, boundary_type='vacuum',
                                      name='PlaneEndModerator')

    ThroatCylinder = openmc.ZCylinder(r=r_throat, name='ThroatCylinder')

    h = L_conv / (1 - r_throat / r_outlet)
    NozzleConvergentCone = openmc.ZCone(x0=0.0, y0=0.0, z0=L + h,
                                        r2=(r_outlet / h) ** 2,
                                        name='NozzleConvergentCone')
    NozzleDivergentCone = openmc.ZCone(x0=0.0, y0=0.0,
                                       z0=L + L_conv + 1.0 - (h - L_conv),
                                       r2=(r_outlet / h) ** 2,
                                       name='NozzleDivergentCone')
    EndCapSphere = openmc.Sphere(x0=0.0, y0=0.0, z0=L, r=cfg.R3 + cfg.d_tw_outer / 2,
                                 name='EndCapSphere')

    cells = []

    for cavity in cavities:
        R = cavity.rotation
        translation = cavity.translation
        tx, ty, tz = translation

        UpperRegion = +PlaneNozzleStart & -PlaneEndModerator
        for plane in cavity.hex_planes:
            UpperRegion &= -plane

        NozzleModRegion = (UpperRegion & +NozzleConvergentCone
                           & +ThroatCylinder & +NozzleDivergentCone)
        H2_Cone = UpperRegion & -PlaneThroat & +EndCapSphere & -NozzleConvergentCone
        H2_Throat = UpperRegion & -ThroatCylinder & +PlaneThroat & -PlaneNozzleEnd
        H2_Cone_Div = UpperRegion & +PlaneNozzleEnd & -NozzleDivergentCone

        memo = {}
        NozzleModRegion = (NozzleModRegion.rotate(R, pivot=(0., 0., 0.), memo=memo)
                           .translate(translation, memo=memo))
        H2_Cone = (H2_Cone.rotate(R, pivot=(0., 0., 0.), memo=memo)
                   .translate(translation, memo=memo))
        H2_Cone_Div = (H2_Cone_Div.rotate(R, pivot=(0., 0., 0.), memo=memo)
                       .translate(translation, memo=memo))
        H2_Throat = (H2_Throat.rotate(R, pivot=(0., 0., 0.), memo=memo)
                     .translate(translation, memo=memo))

        tag = f'cav({tx:.1f},{ty:.1f},{tz:.1f})'

        # Bounded rod carve -- same union idiom as the slots and the
        # graphite moderator; see the docstring above for why.
        if tie_rods is not None:
            for outer_surf, (plane_bot, plane_top) in zip(tie_rods.outer_surfaces,
                                                          tie_rods.axial_bounds):
                NozzleModRegion &= (+outer_surf | -plane_bot | +plane_top)

        nozzle_cell = openmc.Cell(fill=BeO, region=NozzleModRegion, name=f'nozzle_BeO {tag}')
        cone_cell = openmc.Cell(fill=hydrogen_nozzle, region=H2_Cone,
                                name=f'nozzle_cone_H2 {tag}')
        throat_cell = openmc.Cell(fill=hydrogen_nozzle, region=H2_Throat,
                                  name=f'nozzle_throat_H2 {tag}')
        div_cell = openmc.Cell(fill=hydrogen_nozzle, region=H2_Cone_Div,
                               name=f'nozzle_cone_div_H2 {tag}')

        cavity.nozzle_beo_cell = nozzle_cell
        cavity.nozzle_cone_h2_cell = cone_cell
        cavity.nozzle_throat_h2_cell = throat_cell
        cavity.nozzle_cone_div_h2_cell = div_cell

        cells += [nozzle_cell, cone_cell, throat_cell, div_cell]

    return cells

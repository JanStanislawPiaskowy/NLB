"""One hexagonal cavity: fuel, transparent wall, liner tubes, propellant,
BeO moderator wedge, and top cap -- all inside its own OpenMC universe.

The public interface is deliberately simple:

    cavity = build_cavity(cfg, materials, layered, placement)

`build_cavity` is a *function that returns a value*.  It never touches any
shared state -- the orchestrator (gcr.model.GCR) decides what to collect
and where to put it.  Compare with the old method, which silently mutated
five different attributes of `self`; the tie-rod bug happened precisely
because a later method had to *guess* what an earlier one had left behind.

The physics and every numeric expression are ported 1:1 from the original
build_cavity.  Intentional non-physics deviations (documented in
README - Migration notes): the unused `hl_out` variable was dropped, a
duplicated `& -PlaneFuelZoneStart` term in the injector pipe was removed
(the region is identical), and the wall-plane coefficients now come from
gcr.geometry.hexmaths instead of a second inline copy of the same formula.
"""

from dataclasses import dataclass, field

import numpy as np
import openmc

from ..config import GCRConfig
from ..materials import LayeredMaterials
from ..transforms import rotation_matrix
from .hexmaths import HEX_ANGLES, Placement, hex_plane_coefficients, ridge_vertices


@dataclass
class Cavity:
    """Everything the rest of the package needs to know about one cavity.

    Attributes filled by build_cavity:
        slot_cell        -- the root-universe cell holding the cavity universe
        hex_planes       -- the six LOCAL wall planes (untransformed)
        rotation, translation -- the placement actually applied
        plane_moderator_top_start, plane_fuel_zone_end -- local axial planes
        tie_rod_vertices -- the six local ridge endpoints (x0, y0, xL, yL)

    Attributes filled LATER by other builders (end moderator, nozzle), and
    consumed by resolve_cavity_overlaps.  They are declared here with None
    defaults so every possible field is visible in one place -- no more
    string-keyed dict entries appearing at runtime.
    """
    slot_cell: openmc.Cell
    hex_planes: list
    rotation: np.ndarray
    translation: tuple
    plane_moderator_top_start: openmc.ZPlane
    plane_fuel_zone_end: openmc.ZPlane
    tie_rod_vertices: list = field(default_factory=list)

    # Filled by build_end_moderator / build_nozzle_end:
    end_beo_cell: openmc.Cell = None
    end_graphite_cell: openmc.Cell = None
    nozzle_beo_cell: openmc.Cell = None
    nozzle_cone_h2_cell: openmc.Cell = None
    nozzle_cone_div_h2_cell: openmc.Cell = None
    nozzle_throat_h2_cell: openmc.Cell = None


def build_cavity(cfg: GCRConfig, materials: dict, layered: LayeredMaterials,
                 placement: Placement) -> Cavity:
    """Build one cavity at the given placement and return it.

    Walk through the numbered sections in order -- they follow the physical
    radial build-up: fuel -> transparent wall -> neon -> liner tubes ->
    propellant -> BeO wedge -> top cap -> assembly into a universe.
    """
    fuel_inner = materials['fuel_inner']
    graphite = materials['graphite']
    hydrogen = materials['hydrogen']
    hydrogen_liner = materials['hydrogen_liner']
    hydrogen_tori = materials['hydrogen_tori']
    neon = materials['neon']
    SiO2 = materials['SiO2']
    BeO = materials['BeO']
    Be = materials['Be']

    L = cfg.L
    R0, R1, R2, R3 = cfg.R0, cfg.R1, cfg.R2, cfg.R3
    r_tw = cfg.r_tw
    d_tw_outer = cfg.d_tw_outer
    r_inlet = cfg.r_inlet
    r_outlet = cfg.r_outlet
    hl = cfg.hex_side_length
    tilt = cfg.tilt

    x0, y0, z0 = placement.translation   # only used in cell-name labels

    def _tag(name: str) -> str:
        """Original naming convention: 'name cav(x, y, z)'."""
        return f'{name} cav({x0:.1f},{y0:.1f},{z0:.1f})'

    # ── 1) Fuel region: inner cylinder + outer buffer annulus ────────────────
    FuelZoneInner = openmc.ZCylinder(x0=0, y0=0, r=R0, name='fuel zone inner')
    MiddleOfTransparentWall = openmc.ZCylinder(
        x0=0, y0=0, r=R3 + d_tw_outer / 2, name='middle of-transparent wall')
    FuelZoneBuffer = openmc.ZCylinder(x0=0, y0=0, r=R2, name='fuelZoneBuffer')

    PlaneFuelZoneStart = openmc.ZPlane(z0=0)
    PlaneFuelZoneEnd = openmc.ZPlane(z0=L)

    # Axial layer planes, SHARED between fuel and propellant so both use
    # identical axial slicing.
    n_layers = cfg.n_axial_layers
    if n_layers > 1:
        dz = L / n_layers
        layer_planes = [PlaneFuelZoneStart]
        for k in range(1, n_layers):
            layer_planes.append(openmc.ZPlane(z0=k * dz, name=_tag(f'axial_layer_z{k}')))
        layer_planes.append(PlaneFuelZoneEnd)
    else:
        layer_planes = None

    FuelRegionInner = +PlaneFuelZoneStart & -PlaneFuelZoneEnd & -FuelZoneInner
    FuelRegionBuffer = (+PlaneFuelZoneStart & -PlaneFuelZoneEnd
                        & +FuelZoneInner & -FuelZoneBuffer)
    NeonRegion = (+PlaneFuelZoneStart & -PlaneFuelZoneEnd
                  & +FuelZoneBuffer & -MiddleOfTransparentWall)  # refined below

    if n_layers <= 1:
        fuel_cells = [
            openmc.Cell(fill=materials['fuel_inner'], region=FuelRegionInner),
            openmc.Cell(fill=materials['fuel_outer'], region=FuelRegionBuffer),
        ]
    else:
        fuel_cells = []
        for k in range(n_layers):
            slab = +layer_planes[k] & -layer_planes[k + 1]
            fuel_cells.append(openmc.Cell(
                fill=layered.fuel_inner_layers[k],
                region=slab & -FuelZoneInner,
                name=_tag(f'fuel_inner_layer_{k}')))
            fuel_cells.append(openmc.Cell(
                fill=layered.fuel_outer_layers[k],
                region=slab & +FuelZoneInner & -FuelZoneBuffer,
                name=_tag(f'fuel_outer_layer_{k}')))

    # ── 2) Transparent wall: stack of silica tori along the length ────────────
    n_tori = int((L - r_tw) / (d_tw_outer - r_tw)) + 1
    # Recalculate the torus diameter for a whole number of tori
    d_tw_outer = (L - r_tw + n_tori * r_tw) / n_tori
    r_t_id = (d_tw_outer - 2 * r_tw) / 2

    tori_inner, tori_outer = [], []
    z_torus_0 = d_tw_outer / 2
    for i in range(n_tori):
        # intentional overlap by r_tw between neighbouring rings -- kept as-is
        z_torus = z_torus_0 + i * (d_tw_outer - r_tw)
        tori_inner.append(openmc.ZTorus(
            x0=0, y0=0, z0=z_torus,
            a=R3 + d_tw_outer / 2, b=r_t_id, c=r_t_id,
            name=_tag(f'torus_inner {i}')))
        tori_outer.append(openmc.ZTorus(
            x0=0, y0=0, z0=z_torus,
            a=R3 + d_tw_outer / 2, b=d_tw_outer / 2, c=d_tw_outer / 2,
            name=_tag(f'torus_outer {i}')))

    torus_cells_inner, torus_cells_outer = [], []
    for i, (torus_inner, torus_outer) in enumerate(zip(tori_inner, tori_outer)):
        hydrogen_region = -torus_inner
        silicon_region = +torus_inner & -torus_outer
        # Exclude the space already claimed by the previous ring's SiO2 wall
        if i > 0:
            hydrogen_region &= +tori_outer[i - 1]
            silicon_region &= +tori_outer[i - 1]
        torus_cells_inner.append(openmc.Cell(
            fill=hydrogen_tori, region=hydrogen_region, name=f'tw_hydrogen {i}'))
        torus_cells_outer.append(openmc.Cell(
            fill=SiO2, region=silicon_region, name=f'tw_SiO2 {i}'))

    # ── 3) Neon region refinement: exclude every torus ──────────────────────────
    for torus in tori_outer:
        NeonRegion &= +torus
    fuel_cells.append(openmc.Cell(fill=neon, region=NeonRegion))

    # ── 3.5) Liner tubes ──────────────────────────────────────────────────────────
    n = cfg.n_liner_tubes
    n_tubes = 2 * n  # 144 total

    t_max = 2 * np.pi * r_outlet / (2 * n - np.pi)
    t_min = 2 * np.pi * r_inlet / (2 * n - np.pi)

    tan_beta = (r_outlet - r_inlet + 0.5 * (t_max - t_min)) / L
    cos_beta = np.sqrt(1 / (1 + tan_beta ** 2))
    h_liner = (L * t_min) / (cos_beta * (t_max - t_min))
    beta = float(np.arctan(tan_beta))
    liner_diam_ratio = 0.54 / 0.6

    LinerCells = []
    LinerTubeOutsides = []

    for i in range(n_tubes):
        LinerTubeOutside = openmc.ZCone(r2=((t_min / 2) / h_liner) ** 2,
                                        name=f'LinerOut_{i}')
        LinerTubeInside = openmc.ZCone(r2=(liner_diam_ratio * (t_min / 2) / h_liner) ** 2,
                                       name=f'LinerIn_{i}')
        LinerTubeOutsidePrevious = openmc.ZCone(r2=((t_min / 2) / h_liner) ** 2,
                                                name=f'LinerOut_Previous{i}')

        Rxx_liner = rotation_matrix(xx=-beta)
        pivot = (0, r_inlet + t_min / 2, 0)
        LinerTubeOutside = LinerTubeOutside.translate(
            (0, r_inlet + t_min / 2, -h_liner)).rotate(Rxx_liner, pivot)
        LinerTubeInside = LinerTubeInside.translate(
            (0, r_inlet + t_min / 2, -h_liner)).rotate(Rxx_liner, pivot)
        LinerTubeOutsidePrevious = LinerTubeOutsidePrevious.translate(
            (0, r_inlet + t_min / 2, -h_liner)).rotate(Rxx_liner, pivot)

        Rzz_liner = rotation_matrix(zz=i * np.pi / (n_tubes / 2))
        Rzz_previous = rotation_matrix(zz=(i - 1) * np.pi / (n_tubes / 2))
        LinerTubeOutside = LinerTubeOutside.rotate(Rzz_liner, pivot=(0.0, 0.0, 0.0))
        LinerTubeInside = LinerTubeInside.rotate(Rzz_liner, pivot=(0.0, 0.0, 0.0))
        LinerTubeOutsidePrevious = LinerTubeOutsidePrevious.rotate(
            Rzz_previous, pivot=(0.0, 0.0, 0.0))

        LinerTubeOutsides.append(LinerTubeOutside)

        beryllium_region = (+LinerTubeInside & -LinerTubeOutside & +LinerTubeOutsidePrevious
                            & +PlaneFuelZoneStart & -PlaneFuelZoneEnd)
        hydrogen_region = (-LinerTubeInside
                           & +PlaneFuelZoneStart & -PlaneFuelZoneEnd)

        LinerCells.append(openmc.Cell(fill=Be, region=beryllium_region,
                                      name=f'Liner beryllium cell_{i}'))
        LinerCells.append(openmc.Cell(fill=hydrogen_liner, region=hydrogen_region,
                                      name=f'HydrogenCoolant Liner_{i}'))

    # ── 3.9) Tie-rod ridge vertices + top-cap start plane ────────────────────────────
    t_mod_top = cfg.moderator_top_thickness
    PlaneModeratorTopStart = openmc.ZPlane(z0=-t_mod_top, name='Moderator Start Plane',
                                           boundary_type='vacuum')

    # The six ridge lines where adjacent wall planes intersect.  The maths
    # lives in hexmaths.ridge_vertices -- the SAME plane coefficients used
    # for the walls below, so the two can never drift apart.
    TieRodVertexData = ridge_vertices(tilt, hl, L)

    # ── 4) Propellant region ────────────────────────────────────────────────────────────
    h = L * (1 / ((r_outlet + t_max) / (r_inlet + t_min) - 1))  # apex above PlaneStart
    PropWall = openmc.ZCone(x0=0, y0=0, z0=-h, r2=((r_inlet + t_min) / h) ** 2,
                            name='PropWall')

    if n_layers <= 1:
        PropellantRegion = (+PlaneModeratorTopStart & -PlaneFuelZoneEnd
                            & -PropWall & +MiddleOfTransparentWall)
        for linertube in LinerTubeOutsides:
            LinerRegion = +PlaneFuelZoneStart & -PlaneFuelZoneEnd & -linertube
            PropellantRegion &= ~LinerRegion
        for torus in tori_outer:
            PropellantRegion &= +torus
        PropCell = [openmc.Cell(fill=hydrogen, region=PropellantRegion)]
    else:
        PropCell = []
        HeaderRegion = (+PlaneModeratorTopStart & -PlaneFuelZoneStart
                        & -PropWall & +MiddleOfTransparentWall)
        PropCell.append(openmc.Cell(fill=layered.h2_header, region=HeaderRegion,
                                    name=_tag('propellant_header')))

        for k in range(n_layers):
            layer_region = (+layer_planes[k] & -layer_planes[k + 1]
                            & -PropWall & +MiddleOfTransparentWall)
            for linertube in LinerTubeOutsides:
                LinerRegion = +PlaneFuelZoneStart & -PlaneFuelZoneEnd & -linertube
                layer_region &= ~LinerRegion
            for torus in tori_outer:
                layer_region &= +torus
            PropCell.append(openmc.Cell(fill=layered.h2_layers[k], region=layer_region,
                                        name=_tag(f'propellant_layer_{k}')))

    # ── 5) BeO moderator wedge (outside the propellant cone, inside the hexagon) ──────────
    BeryliumRegion = +PropWall & +PlaneModeratorTopStart & -PlaneFuelZoneEnd
    for linertube in LinerTubeOutsides:
        LinerRegion = +PlaneFuelZoneStart & -PlaneFuelZoneEnd & -linertube
        BeryliumRegion &= ~LinerRegion

    # ── 6) Top moderator cap with fuel injector / extraction pipes ─────────────────────────
    UraniumExtractionPipe = openmc.ZCylinder(x0=0, y0=0, r=1, name='UraniumExtractionPipe')
    FuelInjectorOut = openmc.ZCylinder(x0=0, y0=0, r=R1 - 2, name='FuelInjectorOut')
    FuelInjectorIn = openmc.ZCylinder(x0=0, y0=0, r=R1 - 5, name='FuelInjectorIn')
    BerylliumGraphiteDivision = openmc.ZPlane(z0=-t_mod_top / 2,
                                              name='BerylliumGraphiteDivision')

    TopCapRegion = ((-MiddleOfTransparentWall & +PlaneModeratorTopStart & -PlaneFuelZoneStart
                     & +UraniumExtractionPipe)
                    & ~(+FuelInjectorIn & -FuelInjectorOut))

    FuelExtractionPipe = -UraniumExtractionPipe & +PlaneModeratorTopStart & -PlaneFuelZoneStart
    FuelInjectorPipe = (+FuelInjectorIn & -FuelInjectorOut
                        & +PlaneModeratorTopStart & -PlaneFuelZoneStart)

    TopModCells = [
        openmc.Cell(fill=graphite, region=TopCapRegion & -BerylliumGraphiteDivision,
                    name='Graphite top cap cell'),
        openmc.Cell(fill=BeO, region=TopCapRegion & +BerylliumGraphiteDivision,
                    name='Beryllium top cap cell'),
        openmc.Cell(fill=fuel_inner, region=FuelExtractionPipe),
        openmc.Cell(fill=fuel_inner, region=FuelInjectorPipe),
    ]

    # The six tilted wall planes.  Coefficients come from hexmaths -- the same
    # single formula that produced the ridge vertices in section 3.9.
    hex_planes = []
    for i, angle in enumerate(HEX_ANGLES):
        A, B, C, D = hex_plane_coefficients(angle, tilt, hl)
        plane = openmc.Plane(a=A, b=B, c=C, d=D, name=f'hex plane {i}')
        BeryliumRegion &= -plane
        hex_planes.append(plane)

    BeryliumCell = [openmc.Cell(fill=BeO, region=BeryliumRegion, name=_tag('BeO_mod'))]

    # ── 7) Assemble the universe and place it in the root frame ──────────────────────────────
    cells = (fuel_cells + torus_cells_inner + torus_cells_outer
             + PropCell + BeryliumCell + TopModCells + LinerCells)
    cavity_universe = openmc.Universe(cells=cells)

    local_slot = +PlaneModeratorTopStart & -PlaneFuelZoneEnd
    for plane in hex_planes:
        local_slot &= -plane

    R = placement.rotation
    slot_transformed = (local_slot
                        .rotate(R, pivot=(0.0, 0.0, 0.0))
                        .translate(placement.translation))

    slot_cell = openmc.Cell(fill=cavity_universe, region=slot_transformed,
                            name=_tag('cavity_slot'))
    # OpenMC's fill transform maps root coordinates INTO the daughter
    # universe, hence the transpose (inverse) of the placement rotation.
    slot_cell.rotation = R.T
    slot_cell.translation = placement.translation

    return Cavity(
        slot_cell=slot_cell,
        hex_planes=hex_planes,
        rotation=R,
        translation=placement.translation,
        plane_moderator_top_start=PlaneModeratorTopStart,
        plane_fuel_zone_end=PlaneFuelZoneEnd,
        tie_rod_vertices=TieRodVertexData,
    )

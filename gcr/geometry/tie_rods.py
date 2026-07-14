"""Tie rods along the ridges where the hexagonal cavity slots meet.

STATUS: implemented and importable, but NOT included in the model by
default (GCR(include_tie_rods=False)).  Enable deliberately, run a
geometry-debug dry run, and inspect the plots before trusting results.

What was wrong before, and what this version does differently
--------------------------------------------------------------
The original build_tie_rods had correct ridge mathematics (verified
numerically: the vertex lines of the three cavities sharing an inner
vertex coincide to machine precision).  It failed for two other reasons:

1.  build_moderator carved each rod's INFINITE cylinder out of the
    graphite, while the rod cells only exist between two axial cap
    planes.  Everywhere the extended cylinder crossed graphite beyond
    the caps was left UNDEFINED -> lost particles.  Here, the builder
    hands the moderator the caps together with the cylinders (one
    TieRods object), and build_moderator carves the bounded region
        (+outer | -cap_bottom | +cap_top)
    -- the same union-carve idiom already used on the cavity slots.

2.  Duplicate ridges were merged by comparing START POINTS with a 0.5 cm
    tolerance.  That worked only by luck: inner vertices coincide
    exactly, but the six outer peripheral-peripheral vertices mismatch
    by ~3 mm (tapered hexagonal prisms cannot tile exactly; the residual
    is ~ tilt^2 * hl).  This version merges by LINE -- same direction
    within tierod_merge_angle_deg AND perpendicular offset within
    tierod_merge_distance -- and centres the surviving rod on the
    AVERAGE of the merged ridge lines, so it sits symmetrically between
    the slightly mismatched hexagon edges.

Design decision left open (on purpose)
--------------------------------------
Rods span the fuel-zone length only (local z in [0, L] of the ridge).
Physically they are full-length structural members; extending them
requires carving the END moderator too, which this builder does not do.
Change `z_bottom` / `z_top` only together with that carve.
"""

from dataclasses import dataclass, field

import numpy as np
import openmc

from ..config import GCRConfig
from ..transforms import rotation_matrix


@dataclass
class RodLine:
    """One unique ridge line after merging.

    g0, gL   -- averaged global endpoints of the merged segments
    cavities -- indices (into the cavities list) of every slot sharing it
    """
    g0: np.ndarray
    gL: np.ndarray
    cavities: set = field(default_factory=set)

    @property
    def direction(self) -> np.ndarray:
        d = self.gL - self.g0
        return d / np.linalg.norm(d)


@dataclass
class TieRods:
    """Everything downstream code needs about the built rods.

    `outer_surfaces` and `axial_bounds` travel TOGETHER because a carve
    is only valid with both: the moderator builder receives this object
    whole, so it physically cannot repeat the unbounded-carve mistake.
    """
    cells: list = field(default_factory=list)
    outer_surfaces: list = field(default_factory=list)
    axial_bounds: list = field(default_factory=list)   # [(plane_bot, plane_top), ...]
    lines: list = field(default_factory=list)          # [RodLine, ...]


def _merge_ridge_lines(segments, tol_dist: float, tol_angle_rad: float) -> list:
    """Group near-coincident ridge segments into unique RodLines.

    Two segments belong to the same ridge if their directions agree within
    `tol_angle_rad` AND the perpendicular distance from one start point to
    the other's line is below `tol_dist`.  Merged endpoints are averaged.
    """
    lines: list[RodLine] = []
    sums: list[dict] = []   # running sums for averaging, parallel to `lines`

    for g0, gL, cavity_idx in segments:
        d = gL - g0
        d = d / np.linalg.norm(d)

        matched = None
        for line, s in zip(lines, sums):
            ld = line.direction
            cos_ang = float(np.clip(abs(np.dot(d, ld)), -1.0, 1.0))
            if np.arccos(cos_ang) > tol_angle_rad:
                continue
            # perpendicular offset of this segment's start from the line
            v = g0 - line.g0
            perp = v - np.dot(v, ld) * ld
            if np.linalg.norm(perp) < tol_dist:
                matched = (line, s)
                break

        if matched is None:
            lines.append(RodLine(g0=g0.copy(), gL=gL.copy(), cavities={cavity_idx}))
            sums.append({'g0': g0.copy(), 'gL': gL.copy(), 'n': 1})
        else:
            line, s = matched
            s['g0'] += g0
            s['gL'] += gL
            s['n'] += 1
            line.g0 = s['g0'] / s['n']    # rod sits on the AVERAGED ridge
            line.gL = s['gL'] / s['n']
            line.cavities.add(cavity_idx)

    return lines


def build_tie_rods(cfg: GCRConfig, materials: dict, cavities: list) -> TieRods:
    """Build one rod per unique ridge and carve it from every sharing slot.

    Must be called AFTER all cavities exist and BEFORE build_moderator
    (the moderator needs the returned TieRods to carve itself).

    Steps:
      1) transform each cavity's six local ridge segments to global frame,
      2) merge near-coincident segments into unique ridge lines,
      3) build three concentric cells (H2 coolant / Be / graphite coat)
         per rod, bounded by per-rod cap planes,
      4) carve the bounded rod region out of every sharing slot cell.
    """
    Be = materials['Be']
    graphite = materials['graphite']
    # NOTE: average-density canonical hydrogen, as in the original.  A
    # per-layer refinement is possible but the coolant volume is tiny.
    hydrogen = materials['hydrogen']

    r_inner = cfg.diameter_tierod_inner / 2
    r_middle = cfg.diameter_tierod_outer / 2
    r_outer = cfg.diameter_tierod_outer / 2 + cfg.thickness_tierod_graphite

    L = cfg.L

    # ── 1) All ridge segments in global coordinates ─────────────────────────
    segments = []
    for cavity_idx, cavity in enumerate(cavities):
        R = cavity.rotation
        t = np.asarray(cavity.translation, dtype=float)
        for (lx0, ly0, lxL, lyL) in cavity.tie_rod_vertices:
            g0 = R @ np.array([lx0, ly0, 0.0]) + t
            gL = R @ np.array([lxL, lyL, L]) + t
            segments.append((g0, gL, cavity_idx))

    # ── 2) Merge by line, not by endpoint ─────────────────────────────────────
    lines = _merge_ridge_lines(
        segments,
        tol_dist=cfg.tierod_merge_distance,
        tol_angle_rad=np.deg2rad(cfg.tierod_merge_angle_deg),
    )

    rods = TieRods(lines=lines)

    # ── 3) Surfaces and cells for each unique rod ───────────────────────────────
    for idx, line in enumerate(lines):
        g0, gL = line.g0, line.gL
        dx, dy, dz = line.direction

        # Per-rod cap planes, perpendicular to the rod axis.
        d_bot = dx * g0[0] + dy * g0[1] + dz * g0[2]
        d_top = dx * gL[0] + dy * gL[1] + dz * gL[2]
        plane_bot = openmc.Plane(a=dx, b=dy, c=dz, d=d_bot, name=f'TieRod_bot_{idx}')
        plane_top = openmc.Plane(a=dx, b=dy, c=dz, d=d_top, name=f'TieRod_top_{idx}')
        rod_axial_bounds = +plane_bot & -plane_top

        # Rotate a ZCylinder so its axis aligns with the ridge direction:
        # first tilt from +z by alpha about x, then spin by phi about z.
        # (arccos/arctan2 pair verified against the rotation convention.)
        alpha = float(np.arccos(np.clip(dz, -1.0, 1.0)))
        phi = float(np.arctan2(dx, -dy))

        surfaces = {}
        for label, radius in (('inner', r_inner), ('mid', r_middle), ('outer', r_outer)):
            cyl = openmc.ZCylinder(r=radius, name=f'TieRod_{label}_{idx}')
            cyl = cyl.rotate(rotation_matrix(xx=alpha), pivot=(0, 0, 0))
            cyl = cyl.rotate(rotation_matrix(zz=phi), pivot=(0, 0, 0))
            cyl = cyl.translate(tuple(g0))
            surfaces[label] = cyl

        inner, middle, outer = surfaces['inner'], surfaces['mid'], surfaces['outer']

        rods.outer_surfaces.append(outer)
        rods.axial_bounds.append((plane_bot, plane_top))

        rods.cells.append(openmc.Cell(
            fill=hydrogen, region=-inner & rod_axial_bounds,
            name=f'tierod_H2_{idx}'))
        rods.cells.append(openmc.Cell(
            fill=Be, region=+inner & -middle & rod_axial_bounds,
            name=f'tierod_Be_{idx}'))
        rods.cells.append(openmc.Cell(
            fill=graphite, region=+middle & -outer & rod_axial_bounds,
            name=f'tierod_C_{idx}'))

        # ── 4) Carve the BOUNDED rod region from every sharing slot ─────────────
        # ~(cylinder AND between-caps)  ==  (+outer | -bot | +top)
        for cavity_idx in line.cavities:
            slot = cavities[cavity_idx].slot_cell
            slot.region &= (+outer | -plane_bot | +plane_top)

    n_shared = sum(1 for line in lines if len(line.cavities) > 1)
    print(f'  {len(lines)} tie rods built ({n_shared} on shared ridges); '
          f'slot cells carved with bounded regions')
    return rods

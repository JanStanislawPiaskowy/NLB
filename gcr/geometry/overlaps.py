"""Resolving double-claimed regions between neighbouring cavities.

Why this module must exist at all: the seven tapered hexagonal prisms
cannot tile exactly (the peripheral-to-peripheral faces mismatch by
~ tilt^2 * hl ~ 3 mm).  Rather than pretending the tiling is perfect,
each neighbouring pair gets a shared midplane, and each cavity keeps
only its own side of it.
"""

import numpy as np
import openmc

from ..config import GCRConfig


def resolve_cavity_overlaps(cfg: GCRConfig, cavities: list) -> None:
    """Insert a shared midplane between every pair of adjacent cavities.

    For each neighbouring pair (centre-to-centre distance below twice the
    hexagon side length), a plane through the midpoint of the two cavity
    centres, normal to the line joining them, trims the slot cell and the
    end-cap / nozzle-moderator cells of both cavities so no region is
    claimed twice.
    """
    hl = cfg.hex_side_length
    n = len(cavities)

    for i in range(n):
        for j in range(i + 1, n):
            t_i = np.asarray(cavities[i].translation, dtype=float)
            t_j = np.asarray(cavities[j].translation, dtype=float)

            dist = float(np.linalg.norm(t_j - t_i))
            if dist > 2 * hl:      # only immediate neighbours
                continue

            midpoint = (t_i + t_j) / 2
            normal = (t_j - t_i) / dist
            nx, ny, nz = normal
            D = nx * midpoint[0] + ny * midpoint[1] + nz * midpoint[2]

            shared_plane = openmc.Plane(a=nx, b=ny, c=nz, d=D,
                                        name=f'shared_boundary_{i}_{j}')

            # The same cell set the original trimmed: the slot itself, both
            # end-moderator caps, and the nozzle BeO (the nozzle H2 cells
            # were deliberately excluded in the original too).
            for cell in (cavities[i].slot_cell,
                         cavities[i].end_beo_cell,
                         cavities[i].end_graphite_cell,
                         cavities[i].nozzle_beo_cell):
                if cell is not None:
                    cell.region &= -shared_plane
            for cell in (cavities[j].slot_cell,
                         cavities[j].end_beo_cell,
                         cavities[j].end_graphite_cell,
                         cavities[j].nozzle_beo_cell):
                if cell is not None:
                    cell.region &= +shared_plane


def resolve_liner_overlaps(liner_cells: list, n: int) -> None:
    """Midplanes between adjacent liner-tube beryllium cells.

    NOTE: this was dead code in the original file -- never called, because
    the `LinerTubeOutsidePrevious` trim inside the liner loop already
    prevents the overlap.  It is kept (as a module function) purely as
    reference in case the liner construction is ever reworked; do not call
    it in the default build.
    """
    n_tubes = 2 * n

    for i in range(n_tubes):
        j = (i + 1) % n_tubes

        # The beryllium cell for tube i is at index i*2 (Be, H2 alternate)
        be_cell_i = liner_cells[i * 2]
        be_cell_j = liner_cells[j * 2]

        angle_i = i * np.pi / n
        angle_j = j * np.pi / n
        mid_angle = (angle_i + angle_j) / 2.0

        nx = np.cos(mid_angle + np.pi / 2)
        ny = np.sin(mid_angle + np.pi / 2)

        shared_plane = openmc.Plane(a=nx, b=ny, c=0.0, d=0.0,
                                    name=f'liner_mid_{i}_{j}')
        be_cell_i.region &= -shared_plane
        be_cell_j.region &= +shared_plane

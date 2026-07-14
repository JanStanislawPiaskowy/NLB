"""Pure mathematics of the seven-cavity hexagonal arrangement.

Everything in this module is plain NumPy -- no OpenMC anywhere.  That is
the point: these functions are the *single source of truth* for

  * the tilted hexagon plane coefficients   (previously written out TWICE
    inside build_cavity: once in the tie-rod helper, once in the wall loop),
  * the ridge (vertex) lines where hexagon faces meet,
  * the placement (rotation + translation) of the seven cavities
    (previously duplicated between build_cavity's assumptions and main()).

Because there is no OpenMC here, all of it is unit-testable in
milliseconds -- see tests/test_hexmaths.py.

Geometry in words
-----------------
Each cavity sits in a hexagonal prism slot.  The propellant channel widens
along +z (r_inlet -> r_outlet), so the hexagon is *tapered*: every one of
its six wall planes is tilted by `cfg.tilt` about the line where that wall
meets the z = 0 plane.  Six such tapered prisms are arranged around a
central seventh, each rotated by phi = 2*tilt about its own wall so the
central interfaces mate exactly.

A consequence worth knowing: the *peripheral-to-peripheral* interfaces
cannot then mate exactly -- tapered hexagonal prisms do not tile.  The
residual mismatch is second order in the tilt (~ tilt^2 * hl ~ 3 mm here).
It is absorbed by resolve_cavity_overlaps() and by the ridge-merging
tolerance in the tie-rod builder.
"""

from dataclasses import dataclass

import numpy as np

from ..config import GCRConfig
from ..transforms import rotation_matrix

#: The six wall angles of the hexagon, in the order the model has always used.
HEX_ANGLES = (0.0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3)

_SIN60 = np.sin(np.pi / 3)


def hex_plane_coefficients(angle: float, tilt: float, hl: float):
    """Coefficients (A, B, C, D) of one tilted hexagon wall plane, A x + B y + C z = D.

    The plane's normal starts as +y, is tilted by `tilt` about x, then
    rotated to `angle` about z.  D is chosen so that, at z = 0, the plane
    passes at the regular-hexagon apothem hl * sin(60 deg) from the axis.
    (At tilt = 0 all six planes therefore sit exactly at the apothem --
    that property is unit-tested.)

    This is the ONE authoritative copy of the D-formula.  Both the wall
    construction and the ridge solver call it, so they can never drift
    apart again.
    """
    n = rotation_matrix(zz=-angle) @ rotation_matrix(xx=tilt) @ np.array([0.0, 1.0, 0.0])
    A, B, C = float(n[0]), float(n[1]), float(n[2])

    if np.isclose(angle, 0.0) or np.isclose(angle, np.pi / 3):
        D = B * hl * _SIN60 + A * hl / 2
    elif np.isclose(angle, 2 * np.pi / 3) or np.isclose(angle, np.pi):
        D = -B * hl * _SIN60 + A * hl / 2
    elif np.isclose(angle, 4 * np.pi / 3) or np.isclose(angle, 5 * np.pi / 3):
        D = -A * (hl * np.cos(np.pi / 3) + hl / 2)
    else:
        raise ValueError(f'Angle {angle} is not one of the six hexagon wall angles.')
    return A, B, C, D


def ridge_vertices(tilt: float, hl: float, L: float):
    """The six hexagon vertex (ridge) lines, in the cavity's local frame.

    Vertex i is the intersection line of wall planes i and i-1.  A line is
    fixed by two points, so we solve the 2x2 system A x + B y = D - C z at
    z = 0 and again at z = L.

    Returns
    -------
    list of 6 tuples (x0, y0, xL, yL):
        (x0, y0) is the vertex at z = 0, (xL, yL) the vertex at z = L.
        Because of the taper, (xL, yL) lies slightly further out than
        (x0, y0) -- the ridges lean outwards with the walls.
    """
    coeffs = [hex_plane_coefficients(a, tilt, hl) for a in HEX_ANGLES]
    vertices = []
    for i in range(6):
        Aa, Ba, Ca, Da = coeffs[i]
        Ab, Bb, Cb, Db = coeffs[(i - 1) % 6]
        M = np.array([[Aa, Ba],
                      [Ab, Bb]])
        x0, y0 = np.linalg.solve(M, np.array([Da, Db]))
        xL, yL = np.linalg.solve(M, np.array([Da - Ca * L, Db - Cb * L]))
        vertices.append((float(x0), float(y0), float(xL), float(yL)))
    return vertices


@dataclass
class Placement:
    """Where one cavity goes: rotate its local frame by `rotation`, then
    shift it by `translation`.

    `angle_xx` (tilt about x, = 2*cfg.tilt for the outer six) and
    `angle_zz` (position around the ring) are kept because build_cavity
    labels cells with them and the source builder uses the translation.
    """
    rotation: np.ndarray        # 3x3
    translation: tuple          # (x, y, z) [cm]
    angle_xx: float             # [rad]
    angle_zz: float             # [rad]

    @property
    def is_central(self) -> bool:
        return self.angle_xx == 0.0 and self.angle_zz == 0.0


def cavity_placements(cfg: GCRConfig) -> list:
    """Rotation + translation for all seven cavities (central first).

    This reproduces, in one place, the placement loop that used to live in
    main():  each outer cavity is tilted by phi = 2*tilt about x (so its
    wall mates flat against the central cavity's wall) and stationed at

        y0 = hl sin60 (1 + cos phi),      z0 = hl sin60 sin phi,

    then rotated by theta = -i*pi/3 about z into its ring position.
    """
    hl = cfg.hex_side_length
    phi = 2 * cfg.tilt

    placements = [Placement(rotation=np.eye(3), translation=(0.0, 0.0, 0.0),
                            angle_xx=0.0, angle_zz=0.0)]

    for i in range(6):
        x0 = 0.0
        y0 = hl * _SIN60 * (1 + np.cos(phi))
        z0 = hl * _SIN60 * np.sin(phi)
        theta = -i * np.pi / 3

        xp = x0 * np.cos(theta) - y0 * np.sin(theta)
        yp = x0 * np.sin(theta) + y0 * np.cos(theta)

        R = rotation_matrix(zz=theta) @ rotation_matrix(xx=phi)
        placements.append(Placement(rotation=R, translation=(xp, yp, z0),
                                    angle_xx=phi, angle_zz=theta))
    return placements


def bounding_sphere_offset(cfg: GCRConfig) -> float:
    """Axial offset used when positioning the end-curvature bounding sphere.

    Note tan(phi) < 0 here (the channel widens), so the offset is negative.
    """
    phi = 2 * cfg.tilt
    return float(_SIN60 * cfg.hex_side_length / np.tan(phi))

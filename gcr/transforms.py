"""Small geometric transform helpers.



Convention: right-handed, active rotations.  rotation_matrix(zz=a) rotates
a vector by +a radians about +z (anticlockwise looking down the axis).
"""

import numpy as np


def rotation_matrix(xx: float = 0.0, yy: float = 0.0, zz: float = 0.0) -> np.ndarray:
    """Return a 3x3 rotation matrix about x, y and/or z.

    All call sites in this package pass exactly one axis.  If more than one
    is given, the composition order is Rz @ Ry @ Rx (x applied first).

    Parameters
    ----------
    xx, yy, zz :
        Rotation angles in radians about the x, y, z axes.

    Examples
    --------
    >>> rotation_matrix(zz=np.pi / 2) @ np.array([1.0, 0.0, 0.0])
    array([0., 1., 0.])   # +x rotates onto +y
    """
    cx, sx = np.cos(xx), np.sin(xx)
    cy, sy = np.cos(yy), np.sin(yy)
    cz, sz = np.cos(zz), np.sin(zz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

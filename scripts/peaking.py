"""
peaking.py  --  power peaking for the seven-cavity nuclear light bulb
=====================================================================

Pure post-processing of the ``fission-q-recoverable`` mesh tally.  Nothing
here imports OpenMC at module level, so the whole analysis chain can be
exercised on a synthetic field without a nuclear-data installation (see
``peaking_selftest.py``).

WHAT CHANGED RELATIVE TO THE PREVIOUS VERSION
---------------------------------------------
1.  **Everything is binned in the cavity's own frame.**  The old code
    described each cavity by a single point ``(x, y)`` and built masks,
    axial profiles and azimuthal harmonics about that point.  The six
    peripheral cavities are canted: their axes sweep radially outwards by
    tens of centimetres over the fuel height, which is a large fraction of
    a fuel radius.  A fixed-point mask therefore (a) truncates the outer
    fuel near the exit plane, (b) smears the radial profile, and (c)
    injects a spurious first azimuthal harmonic whose size is set by the
    axial asymmetry of the power.  Here each cavity carries an *axis*
    (origin + direction) and every voxel is expressed as (r, phi, s) about
    that axis.

2.  **No power is silently discarded.**  ``compute_peaking`` returns a
    closed power balance: in-cavity, upstream of the inlet plane,
    downstream of the exit plane, and beyond the capture radius.  The
    previous "~5 % unaccounted" is now an itemised number rather than a
    residual.

3.  **F_q no longer depends on an arbitrary power contour.**  The old fuel
    mask was ``P > 2 % of the peak voxel`` -- a contour of the very
    quantity being normalised, anchored to a statistically noisy maximum.
    The fuel mask here is geometric: ``r <= R_fuel`` and ``0 <= s <= L``.
    The denominator of F_q is then the honest volumetric mean,
    ``P_fuel / V_fuel``.

4.  **The hot-spot factor is reported with its selection bias.**  A voxel
    maximum drawn from noisy bins overshoots the true maximum, because the
    bin that wins is the one that fluctuated upwards.  With ~5e7 fission
    events spread over ~1e7 fuel voxels this is not a footnote: on a
    synthetic field with an exactly known answer, a per-voxel sigma of 15 %
    inflates F_q by 38 %.  The bias is estimated by a parametric bootstrap
    (add a second independent copy of the tally's own noise and watch how
    far the maximum moves), which recovers the truth to about 10 % over
    per-voxel sigmas from 5 % to 30 %.  ``F_q_vs_rebin`` replaces the old
    "choose the factor where it goes flat" advice, which cannot work: F_q
    falls monotonically with coarsening because bias and genuine resolution
    are removed together.  Use two statepoints and ``F_q_split_half`` for a
    number that is unbiased by construction rather than by model.

5.  **The peaking factors form an exact telescoping product.**  F_cav,
    F_s and F_rphi are defined conditionally so that
    ``F_q = F_cav * F_s * F_rphi`` identically.  The separability question
    is then asked properly, by comparing F_q against the prediction of a
    separable model fitted to the marginal profiles, rather than against a
    product of quantities that double-count each other.  (The old F_xy was
    a max/mean over a whole horizontal plane containing all seven
    cavities, so it already contained F_cav.)

CONVENTIONS
-----------
Local cylindrical coordinates about cavity i:

    s     arc length along the cavity axis from the inlet plane, cm
    r     perpendicular distance from the axis, cm
    phi   azimuth, 0 = radially outward (away from the machine axis),
          180 deg = towards the machine centre

so a positive first-harmonic phase near 180 deg means the fission power
leans towards the core centre.

STATISTICS HEALTH WARNING
-------------------------
Integral quantities (cavity powers, axial and radial profiles, azimuthal
harmonics) average over 1e4-1e6 voxels and are statistically sound.  The
voxel maximum is not: it is an extreme-value statistic on a mesh whose
individual bins carry tens of per cent uncertainty.  Treat F_q as the
weakest number in the table and quote the voxel size alongside it -- a
hot-spot factor without a stated averaging volume is not a physical
quantity.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# =============================================================================
# SETTINGS  (hand-edit; no argparse, per house style)
# =============================================================================

POWER_TALLY_NAME = "power_distribution"

#: Block-sum factor applied on load, (nx, ny, nz).  (1, 1, 1) = no rebinning.
REBIN = (4, 4, 4)
LOAD_STD_DEV = True

#: Perpendicular distance from a cavity axis within which a voxel is
#: attributed to that cavity.  Must be < half the minimum axis separation.
CAPTURE_RADIUS_CM = 30.0

#: Outer fuel radius, cm (GCRConfig.R2).  Defines the geometric fuel mask.
R_FUEL_CM = 20.75

#: Fuel length along the cavity axis, cm (GCRConfig.L = 6 ft).
L_FUEL_CM = 182.88

#: Bootstrap replicates used to estimate the hot-spot selection bias.
#: 0 disables it (and saves one RNG draw per fuel voxel per replicate).
N_BOOTSTRAP = 6

N_AZIMUTHAL_BINS = 36
N_AXIAL_BINS = 60
N_RADIAL_BINS = 40

#: Chunking for the local-coordinate pass, in axial planes.  Keeps peak
#: memory to roughly  nx*ny*chunk*n_cav*8 bytes.
Z_CHUNK = 24


# =============================================================================
# Part 1 -- the power field
# =============================================================================

@dataclass
class PowerField:
    """A scalar field on a regular Cartesian mesh, indexed ``[ix, iy, iz]``.

    ``data`` is the tally mean per voxel, i.e. eV per source particle, an
    *extensive* per-voxel quantity.  Block-summing therefore preserves its
    meaning; block-averaging would not.
    """
    data: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dx: float
    dy: float
    dz: float
    sigma: Optional[np.ndarray] = None
    n_realizations: Optional[int] = None
    rebin: Tuple[int, int, int] = (1, 1, 1)

    @property
    def voxel_volume(self) -> float:
        return self.dx * self.dy * self.dz

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    def total(self) -> float:
        return float(self.data.sum())

    def scaled(self, power_W: float) -> "PowerField":
        """Copy normalised so the whole-mesh integral equals ``power_W``.

        The mesh must enclose every fissionable region, otherwise this
        normalisation quietly redistributes the missing power onto what is
        left.  ``compute_peaking`` checks the enclosure and complains.
        """
        f = power_W / self.total()
        return PowerField(
            data=self.data * f, x=self.x, y=self.y, z=self.z,
            dx=self.dx, dy=self.dy, dz=self.dz,
            sigma=None if self.sigma is None else self.sigma * f,
            n_realizations=self.n_realizations, rebin=self.rebin,
        )


def block_sum(a: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """Block-*sum* over integer factors, trimming any remainder."""
    fx, fy, fz = (int(f) for f in factors)
    if (fx, fy, fz) == (1, 1, 1):
        return a
    nx, ny, nz = a.shape
    nx -= nx % fx
    ny -= ny % fy
    nz -= nz % fz
    a = a[:nx, :ny, :nz]
    return a.reshape(nx // fx, fx, ny // fy, fy, nz // fz, fz).sum(axis=(1, 3, 5))


def load_power_mesh(
    statepoint: str,
    tally: object = POWER_TALLY_NAME,
    rebin: Sequence[int] = REBIN,
    load_std_dev: bool = LOAD_STD_DEV,
    expect_score: Optional[str] = "fission-q-recoverable",
) -> PowerField:
    """Read a mesh tally into a ``[ix, iy, iz]`` field.

    OpenMC writes mesh filter bins with x varying fastest and z slowest, so
    a C-order reshape to ``(nz, ny, nx)`` followed by one transpose gives
    ``[ix, iy, iz]``.  Exactly one transpose -- the historical bug was an
    extra ``.T``.

    A single-bin ParticleFilter (added by ``apply_particle_filter`` when
    photon transport is on) does not change the flattened ordering, so it is
    tolerated; anything that multiplies the bin count is rejected loudly.
    """
    import openmc  # deliberately lazy: keeps the maths importable anywhere

    with openmc.StatePoint(statepoint) as sp:
        t = _find_tally(sp, tally)

        if expect_score is not None and list(t.scores) != [expect_score]:
            raise ValueError(
                f"tally {t.name!r} scores {list(t.scores)}, expected "
                f"[{expect_score!r}].  Peaking factors are defined on the "
                f"fission energy-release density; pass expect_score=None only "
                f"if you know what you are substituting.")

        mf = None
        for f in t.filters:
            if isinstance(f, openmc.MeshFilter):
                mf = f
        if mf is None:
            raise KeyError(f"tally {t.name!r} carries no MeshFilter")

        mesh = mf.mesh
        nx, ny, nz = (int(v) for v in mesh.dimension)
        ll = np.asarray(mesh.lower_left, dtype=float)
        ur = np.asarray(mesh.upper_right, dtype=float)
        n_real = int(getattr(t, "num_realizations", 0)) or None

        mean = np.asarray(t.mean, dtype=np.float64).ravel()
        if mean.size != nx * ny * nz:
            raise ValueError(
                f"tally has {mean.size} bins but the mesh is {nx}x{ny}x{nz}; "
                f"filters = {[type(f).__name__ for f in t.filters]}.  Slice the "
                f"extra filter bins out before calling this.")
        data = block_sum(mean.reshape((nz, ny, nx)).transpose(2, 1, 0), rebin)
        del mean

        sigma = None
        if load_std_dev:
            sd = np.asarray(t.std_dev, dtype=np.float64).ravel()
            sd = sd.reshape((nz, ny, nx)).transpose(2, 1, 0)
            # Quadrature: correct only for independent bins.  Neighbouring
            # mesh bins in an eigenvalue run share histories and share the
            # batch-to-batch source correlation, so this UNDERESTIMATES the
            # uncertainty of a block sum.  Treat the result as a lower bound.
            sigma = np.sqrt(block_sum(sd ** 2, rebin))
            del sd

    fx, fy, fz = (int(f) for f in rebin)
    dx = (ur[0] - ll[0]) / nx * fx
    dy = (ur[1] - ll[1]) / ny * fy
    dz = (ur[2] - ll[2]) / nz * fz
    x = ll[0] + dx * (np.arange(data.shape[0]) + 0.5)
    y = ll[1] + dy * (np.arange(data.shape[1]) + 0.5)
    z = ll[2] + dz * (np.arange(data.shape[2]) + 0.5)

    print(f"[peaking] {nx}x{ny}x{nz} -> {data.shape}  (rebin {fx}x{fy}x{fz})")
    print(f"[peaking] voxel {dx:.3f} x {dy:.3f} x {dz:.3f} cm "
          f"= {dx*dy*dz:.4f} cm3;  mesh z = [{ll[2]:.1f}, {ur[2]:.1f}] cm")
    return PowerField(data=np.ascontiguousarray(data), x=x, y=y, z=z,
                      dx=dx, dy=dy, dz=dz,
                      sigma=None if sigma is None else np.ascontiguousarray(sigma),
                      n_realizations=n_real, rebin=(fx, fy, fz))


def _find_tally(sp, name_or_id):
    if isinstance(name_or_id, (int, np.integer)):
        return sp.tallies[int(name_or_id)]
    for t in sp.tallies.values():
        if t.name == name_or_id:
            return t
    raise KeyError(f"no tally named {name_or_id!r}; available: "
                   f"{[t.name for t in sp.tallies.values()]}")


# =============================================================================
# Part 2 -- cavity axes
# =============================================================================

@dataclass
class CavityAxis:
    """A straight cavity axis and the frame used to report azimuth.

    origin      point on the axis in the inlet plane (s = 0), cm
    direction   unit vector along the axis, downstream positive
    e_out       unit vector, perpendicular to ``direction``, pointing away
                from the machine axis; defines phi = 0
    e_tan       ``direction x e_out``, completing a right-handed frame
    """
    origin: np.ndarray
    direction: np.ndarray
    e_out: np.ndarray
    e_tan: np.ndarray
    label: str = ""

    @property
    def cant_deg(self) -> float:
        """Angle between the cavity axis and the machine axis, degrees."""
        return float(np.degrees(np.arccos(np.clip(self.direction[2], -1.0, 1.0))))

    @property
    def r_machine(self) -> float:
        """Radial position of the axis in the inlet plane, cm."""
        return float(np.hypot(self.origin[0], self.origin[1]))

    def point_at(self, s: float) -> np.ndarray:
        return self.origin + s * self.direction

    def to_local(self, X, Y, Z):
        """Return ``(r, phi, s)`` for arrays of global coordinates."""
        dx = X - self.origin[0]
        dy = Y - self.origin[1]
        dz = Z - self.origin[2]
        s = dx * self.direction[0] + dy * self.direction[1] + dz * self.direction[2]
        px = dx - s * self.direction[0]
        py = dy - s * self.direction[1]
        pz = dz - s * self.direction[2]
        r = np.sqrt(px * px + py * py + pz * pz)
        a = px * self.e_out[0] + py * self.e_out[1] + pz * self.e_out[2]
        b = px * self.e_tan[0] + py * self.e_tan[1] + pz * self.e_tan[2]
        return r, np.arctan2(b, a), s


def _make_axis(origin, direction, label="") -> CavityAxis:
    origin = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    if d[2] < 0:
        d = -d
    # Outward radial reference, orthogonalised against the axis.
    radial = np.array([origin[0], origin[1], 0.0])
    if np.linalg.norm(radial) < 1e-9:
        radial = np.array([1.0, 0.0, 0.0])   # central cavity: phi = 0 is +x
    e_out = radial - np.dot(radial, d) * d
    n = np.linalg.norm(e_out)
    if n < 1e-9:                              # degenerate; pick any perpendicular
        e_out = np.cross(d, [0.0, 0.0, 1.0])
        n = np.linalg.norm(e_out)
        if n < 1e-9:
            e_out = np.array([1.0, 0.0, 0.0])
            n = 1.0
    e_out = e_out / n
    e_tan = np.cross(d, e_out)
    return CavityAxis(origin=origin, direction=d, e_out=e_out, e_tan=e_tan,
                      label=label)


def axes_from_placements(placements, z0: float = 0.0) -> List[CavityAxis]:
    """Build axes from the model's own cavity placements.

    This is the preferred route: it uses the geometry you actually built
    instead of inferring it from the tally.

        from gcr.geometry.hexmaths import cavity_placements
        axes = peaking.axes_from_placements(cavity_placements(cfg))

    Each placement must expose ``.translation`` (3-vector) and ``.rotation``
    (3x3 matrix mapping local to global), matching what
    ``GCR.fissile_envelope`` already assumes.  Plain ``(translation,
    rotation)`` tuples are also accepted.
    """
    axes = []
    for k, p in enumerate(placements):
        if hasattr(p, "translation"):
            t = np.asarray(p.translation, dtype=float)
            R = np.asarray(p.rotation, dtype=float)
        else:
            t, R = p
            t = np.asarray(t, dtype=float)
            R = np.asarray(R, dtype=float)
        d = R @ np.array([0.0, 0.0, 1.0])
        origin = t + (z0 - t[2]) / d[2] * d if abs(d[2]) > 1e-9 else t
        label = "central" if np.hypot(t[0], t[1]) < 1.0 else f"ring {k}"
        axes.append(_make_axis(origin, d, label))
    axes.sort(key=lambda a: (a.r_machine > 1.0,
                             math.atan2(a.origin[1], a.origin[0])))
    return axes


@dataclass
class AxisFit:
    """Diagnostics from :func:`fit_cavity_axes`."""
    slab_z: np.ndarray                 # slab centres used, cm
    track_xy: np.ndarray               # [cavity, slab, 2] fitted centroids
    residual_cm: np.ndarray            # [cavity] rms of the straight-line fit
    support_radius_cm: np.ndarray      # [cavity] mean equivalent fuel radius
    cant_deg: np.ndarray               # [cavity]
    ring_radius_inlet: float
    ring_radius_exit: float
    ring_phase_deg: float
    symmetrised: bool


def fit_cavity_axes(
    pf: PowerField,
    n_slabs: int = 16,
    z_window: Optional[Tuple[float, float]] = None,
    support_frac: float = 0.02,
    symmetrise: bool = True,
    n_ring: int = 6,
    max_residual_cm: float = 0.5,
    reject_outliers: bool = True,
) -> Tuple[List[CavityAxis], AxisFit]:
    """Recover the cavity axes from the tally itself.

    Method, and why each step is what it is:

    * The ``fission-q-recoverable`` score is identically zero outside
      fissionable material, so the *support* of the field is the fuel
      boundary -- a hard geometric surface.  ``support_frac`` is set very
      low (2 % of a robust in-fuel level) precisely so the contour hugs
      that surface rather than tracking the power.

    * Within each axial slab, each cavity's support is a disc (an ellipse,
      for a canted cavity) centred on the axis.  Its **unweighted**
      centroid is therefore the axis position, whatever the power does
      inside it.  A *power-weighted* centroid would be displaced towards
      the hot side, and using that as the azimuthal reference annihilates
      the first harmonic identically -- the trap this module exists to
      avoid.

    * Fitting a straight line to the slab centroids gives origin and
      direction, and hence the cant angle, which you should check against
      the geometry.

    * The abscissa of each slab point is the *support's own* mean z, not
      the slab centre.  For an interior slab the two coincide (the cylinder
      segment is point-symmetric about the axis point at the slab's
      mid-plane), but for a slab in which the fuel starts or ends they do
      not, and a partially filled end slab sitting at a long lever arm is
      the single most destructive thing that can happen to a straight-line
      fit.

    * ``reject_outliers`` then drops slabs whose residual exceeds
      3 x MAD, refits, and repeats.  This is what removes the partially
      filled end slabs.  Without it, one bad slab at z = 214 cm pulled a
      9.39 deg cant down to 7.53 deg on the synthetic test field, silently.

    * ``symmetrise`` enforces the six-fold symmetry of the design (common
      cant, common inlet radius, common splay, 60 deg spacing), which
      removes the residual contour bias that survives per cavity.

    * If the final residual still exceeds ``max_residual_cm`` the fit is
      NOT to be trusted and says so loudly.  A good fit on real data is
      sub-millimetre.
    """
    if z_window is None:
        z_window = (float(pf.z.min()), float(pf.z.max()))
    kz = (pf.z >= z_window[0]) & (pf.z <= z_window[1])
    if kz.sum() < n_slabs:
        n_slabs = max(2, int(kz.sum()) // 2)

    zi = np.nonzero(kz)[0]
    edges = np.linspace(zi[0], zi[-1] + 1, n_slabs + 1).astype(int)

    X, Y = np.meshgrid(pf.x, pf.y, indexing="ij")
    R_mach = np.hypot(X, Y)

    # ---- global support: split central from ring, find the lattice phase ----
    flat = pf.data[:, :, kz].sum(axis=2)
    lvl = support_frac * np.percentile(flat[flat > 0], 90) if (flat > 0).any() else 0.0
    sup = flat > lvl
    if not sup.any():
        raise RuntimeError("no support found -- is this the right tally?")

    r_sup = R_mach[sup]
    r_gap = _largest_gap(r_sup)
    theta = np.arctan2(Y[sup], X[sup])
    ring_sel = r_sup >= r_gap
    if ring_sel.sum() < 100:
        raise RuntimeError("could not separate the ring cavities from the "
                           "central one; pass axes explicitly")
    phase = float(np.angle(np.mean(np.exp(1j * n_ring * theta[ring_sel]))) / n_ring)

    print(f"[peaking] support split at r = {r_gap:.1f} cm, "
          f"lattice phase = {np.degrees(phase):.2f} deg")

    # ---- per-slab, per-cavity unweighted centroid of the support ------------
    n_cav = 1 + n_ring
    track = np.full((n_cav, n_slabs, 2), np.nan)
    track_z = np.full((n_cav, n_slabs), np.nan)
    area = np.zeros((n_cav, n_slabs))
    slab_z = np.zeros(n_slabs)

    for j in range(n_slabs):
        a_, b_ = edges[j], edges[j + 1]
        blk = pf.data[:, :, a_:b_]
        slab_z[j] = pf.z[a_:b_].mean()
        pos = blk[blk > 0]
        if pos.size == 0:
            continue
        m3 = blk > support_frac * np.percentile(pos, 90)
        if not m3.any():
            continue
        ix, iy, iz = np.nonzero(m3)
        xs, ys, zs = pf.x[ix], pf.y[iy], pf.z[a_:b_][iz]
        rr = np.hypot(xs, ys)
        th = np.arctan2(ys, xs)
        idx = np.where(rr < r_gap, 0,
                       1 + (np.round((th - phase) / (2 * np.pi / n_ring))
                            .astype(int) % n_ring))
        thick = (b_ - a_) * pf.dz
        for i in range(n_cav):
            k = idx == i
            if k.sum() < 8:
                continue
            track[i, j] = (xs[k].mean(), ys[k].mean())
            track_z[i, j] = zs[k].mean()
            # Mean cross-sectional area = (support volume)/(slab thickness).
            # For a cylinder canted by t this is exactly pi R^2 / cos(t) --
            # no stadium term, because we are averaging cross-sections
            # rather than taking their union in projection.
            area[i, j] = (k.sum() * pf.dx * pf.dy * pf.dz) / thick

    # ---- straight-line fit per cavity --------------------------------------
    axes: List[CavityAxis] = []
    resid = np.zeros(n_cav)
    cant = np.zeros(n_cav)
    r_eq = np.zeros(n_cav)
    n_dropped = np.zeros(n_cav, dtype=int)
    for i in range(n_cav):
        keep = np.isfinite(track[i, :, 0])
        if keep.sum() < 3:
            raise RuntimeError(f"cavity {i}: only {keep.sum()} usable slabs")

        for _ in range(4):
            zz = track_z[i, keep]
            M = np.vstack([zz, np.ones_like(zz)]).T
            (mx, cx), *_ = np.linalg.lstsq(M, track[i, keep, 0], rcond=None)
            (my, cy), *_ = np.linalg.lstsq(M, track[i, keep, 1], rcond=None)
            dev = np.hypot(track[i, keep, 0] - M @ [mx, cx],
                           track[i, keep, 1] - M @ [my, cy])
            if not reject_outliers or keep.sum() <= 5:
                break
            mad = float(np.median(np.abs(dev - np.median(dev)))) or 1e-9
            bad = dev > max(3.0 * 1.4826 * mad, 0.05)
            if not bad.any():
                break
            # Drop the worst one at a time: a partially filled end slab at a
            # long lever arm can make its neighbours look like the outliers.
            drop = np.nonzero(keep)[0][int(np.argmax(dev))]
            keep[drop] = False
            n_dropped[i] += 1

        resid[i] = float(np.sqrt(np.mean(dev ** 2)))
        d = np.array([mx, my, 1.0])
        d /= np.linalg.norm(d)
        cant[i] = np.degrees(np.arccos(d[2]))
        # area is already the mean cross-section = pi R^2 / cos(cant)
        r_eq[i] = float(np.sqrt(np.mean(area[i, keep]) * d[2] / np.pi))
        axes.append(_make_axis(np.array([cx, cy, 0.0]), d))

    axes.sort(key=lambda a: (a.r_machine > 1.0,
                             math.atan2(a.origin[1], a.origin[0])))

    ring = [a for a in axes if a.r_machine > 1.0]
    r_in = float(np.mean([a.r_machine for a in ring]))
    top = [np.hypot(*a.point_at(L_FUEL_CM)[:2]) for a in ring]
    r_ex = float(np.mean(top))

    if symmetrise and len(ring) == n_ring:
        cant_ring = float(np.mean([a.cant_deg for a in ring]))
        angs = np.array([math.atan2(a.origin[1], a.origin[0]) for a in ring])
        ph = float(np.angle(np.mean(np.exp(1j * n_ring * angs))) / n_ring)
        tan_c = math.tan(math.radians(cant_ring))
        new = [_make_axis([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], "central")]
        for k in range(n_ring):
            a = ph + k * 2 * np.pi / n_ring
            u = np.array([np.cos(a), np.sin(a), 0.0])
            new.append(_make_axis(r_in * u, u * tan_c + np.array([0, 0, 1.0]),
                                  f"ring {k}"))
        # Pair by nearest azimuth: the raw list is sorted from -180 deg while
        # the idealised one starts at the fitted phase, so zip() would compare
        # different cavities and report a meaningless "shift".
        dev = max(min(np.linalg.norm(n_.origin - o.origin) for n_ in new[1:])
                  for o in ring)
        print(f"[peaking] symmetrised: inlet radius {r_in:.2f} cm, "
              f"cant {cant_ring:.3f} deg, phase {np.degrees(ph):.2f} deg, "
              f"max origin shift {dev:.2f} cm")
        axes = new
    else:
        for k, a in enumerate(axes):
            a.label = "central" if a.r_machine < 1.0 else f"ring {k}"

    print(f"[peaking] axis fit: cant = "
          f"{', '.join(f'{c:.2f}' for c in cant)} deg;  "
          f"straight-line residual <= {resid.max():.3f} cm"
          + (f";  dropped {int(n_dropped.sum())} outlier slab(s)"
             if n_dropped.sum() else ""))
    if resid.max() > max_residual_cm:
        print("[peaking] " + "!" * 62)
        print(f"[peaking] WARNING: residual {resid.max():.2f} cm exceeds "
              f"{max_residual_cm:.2f} cm.  The axes are NOT reliable and")
        print("[peaking] every factor downstream inherits the error.  Usual")
        print("[peaking] causes: the z_window includes partially filled end")
        print("[peaking] slabs or a downstream exhaust region with different")
        print("[peaking] geometry.  Restrict z_window to the fuel, or pass")
        print("[peaking] the exact axes via axes_from_placements().")
        print("[peaking] " + "!" * 62)
    dilation = 0.25 * (pf.dx + pf.dy)
    print(f"[peaking] equivalent fuel radius from the support: "
          f"{np.mean(r_eq):.2f} +/- {np.std(r_eq):.2f} cm;  expect about "
          f"{R_FUEL_CM + dilation:.2f} cm")
    print(f"[peaking]   (R2 = {R_FUEL_CM:.2f} cm dilated by ~half a voxel: a "
          f"boundary voxel scores if ANY part of it")
    print(f"[peaking]   lies in the fuel, so a thresholded support always "
          f"reads large on a coarse mesh.")
    print(f"[peaking] ring axis radius: {r_in:.2f} cm at the inlet plane, "
          f"{r_ex:.2f} cm at s = {L_FUEL_CM:.1f} cm")

    fit = AxisFit(slab_z=slab_z, track_xy=track, residual_cm=resid,
                  support_radius_cm=r_eq, cant_deg=cant,
                  ring_radius_inlet=r_in, ring_radius_exit=r_ex,
                  ring_phase_deg=float(np.degrees(phase)),
                  symmetrised=bool(symmetrise))
    return axes, fit


def _largest_gap(r: np.ndarray) -> float:
    """Radius of the widest empty annulus in a set of radii.

    Used to separate the central cavity's support from the ring's without
    assuming a lattice pitch.
    """
    h, e = np.histogram(r, bins=80)
    empty, best, run_start = 0, (0, 0), None
    for i, c in enumerate(h):
        if c == 0:
            run_start = i if run_start is None else run_start
        else:
            if run_start is not None:
                if i - run_start > best[1] - best[0]:
                    best = (run_start, i)
                run_start = None
    if best[1] == best[0]:
        return float(np.median(r))
    return float(0.5 * (e[best[0]] + e[best[1]]))


# =============================================================================
# Part 3 -- binning in the cavity frame
# =============================================================================

@dataclass
class CavityStats:
    """Per-cavity profiles, all in the cavity's own frame."""
    power_W: np.ndarray               # [cav] total inside the capture volume
    cap_power_W: np.ndarray           # [cav] complete fuel power in the s window
    cap_axial_W: np.ndarray           # [cav, n_s] ditto, per axial bin
    cap_azim_W: np.ndarray            # [cav, n_phi] ditto, per azimuthal bin
    cap_axial_vox: np.ndarray         # [cav, n_s]   capture-volume voxel counts
    cap_azim_vox: np.ndarray          # [cav, n_phi] ditto
    fuel_power_W: np.ndarray          # [cav] inside the geometric fuel mask
    fuel_voxels: np.ndarray           # [cav] voxel count in the fuel mask
    axial_W: np.ndarray               # [cav, n_s] power per s-bin
    axial_vox: np.ndarray             # [cav, n_s] fuel voxels per s-bin
    radial_W: np.ndarray              # [cav, n_r]
    radial_vox: np.ndarray            # [cav, n_r]
    azim_W: np.ndarray                # [cav, n_phi] fuel mask only
    azim_vox: np.ndarray              # [cav, n_phi]
    azim_outer_W: np.ndarray          # [cav, n_phi] outer third of the fuel
    azim_outer_vox: np.ndarray
    s_edges: np.ndarray
    r_edges: np.ndarray
    phi_edges: np.ndarray
    peak_W: np.ndarray                # [cav] hottest fuel voxel
    peak_sigma_W: np.ndarray          # [cav] its absolute sigma (NaN if absent)
    boot_max_W: np.ndarray            # [cav, n_rep] max after adding fresh noise
    peak_index: np.ndarray            # [cav, 3] voxel index of that peak
    peak_local: np.ndarray            # [cav, 3] (r, phi_deg, s) of that peak
    balance: Dict[str, float]         # closed power balance, W


def bin_in_cavity_frame(
    pf: PowerField,
    axes: Sequence[CavityAxis],
    power_W: float,
    capture_radius: float = CAPTURE_RADIUS_CM,
    r_fuel: float = R_FUEL_CM,
    s_extent: Tuple[float, float] = (0.0, L_FUEL_CM),
    n_s: int = N_AXIAL_BINS,
    n_r: int = N_RADIAL_BINS,
    n_phi: int = N_AZIMUTHAL_BINS,
    z_chunk: int = Z_CHUNK,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 20240,
) -> CavityStats:
    """One chunked pass over the mesh, accumulating everything at once.

    Voxels are attributed to the cavity whose axis they are nearest to in
    the *perpendicular* sense, subject to ``r <= capture_radius``.  Because
    the assignment follows the canted axis, no fuel is lost near the exit
    plane -- which a fixed-point mask cannot manage without a capture
    radius so large that neighbouring cavities start to overlap.
    """
    w = pf.scaled(power_W)
    n_cav = len(axes)
    nx, ny, nz = w.shape

    s_edges = np.linspace(s_extent[0], s_extent[1], n_s + 1)
    r_edges = np.linspace(0.0, r_fuel, n_r + 1)
    phi_edges = np.linspace(-np.pi, np.pi, n_phi + 1)
    r_outer = 2.0 / 3.0 * r_fuel

    power = np.zeros(n_cav)
    cap_power = np.zeros(n_cav)
    cap_axial = np.zeros((n_cav, n_s))
    cap_azim = np.zeros((n_cav, n_phi))
    cap_axial_v = np.zeros((n_cav, n_s))
    cap_azim_v = np.zeros((n_cav, n_phi))
    fuel_power = np.zeros(n_cav)
    fuel_vox = np.zeros(n_cav)
    axial = np.zeros((n_cav, n_s))
    axial_v = np.zeros((n_cav, n_s))
    radial = np.zeros((n_cav, n_r))
    radial_v = np.zeros((n_cav, n_r))
    azim = np.zeros((n_cav, n_phi))
    azim_v = np.zeros((n_cav, n_phi))
    azim_o = np.zeros((n_cav, n_phi))
    azim_ov = np.zeros((n_cav, n_phi))
    peak = np.full(n_cav, -np.inf)
    peak_sig = np.full(n_cav, np.nan)
    n_rep = int(n_bootstrap) if w.sigma is not None else 0
    boot = np.full((n_cav, max(n_rep, 1)), -np.inf)
    rng = np.random.default_rng(seed)
    peak_idx = np.zeros((n_cav, 3), dtype=int)
    peak_loc = np.zeros((n_cav, 3))

    bal = dict(total=w.total(), captured=0.0, upstream=0.0, downstream=0.0,
               outside=0.0)

    X2, Y2 = np.meshgrid(w.x, w.y, indexing="ij")

    for z0 in range(0, nz, z_chunk):
        z1 = min(z0 + z_chunk, nz)
        blk = w.data[:, :, z0:z1]
        sig = None if w.sigma is None else w.sigma[:, :, z0:z1]
        Z3 = np.broadcast_to(w.z[z0:z1], blk.shape)
        X3 = X2[:, :, None]
        Y3 = Y2[:, :, None]

        rr = np.empty((n_cav,) + blk.shape, dtype=np.float32)
        ss = np.empty_like(rr)
        pp = np.empty_like(rr)
        for i, ax in enumerate(axes):
            r_, p_, s_ = ax.to_local(X3, Y3, Z3)
            rr[i], pp[i], ss[i] = r_, p_, s_

        near = np.argmin(rr, axis=0)
        r_near = np.take_along_axis(rr, near[None], 0)[0]
        s_near = np.take_along_axis(ss, near[None], 0)[0]
        inside = r_near <= capture_radius

        bal["outside"] += float(blk[~inside].sum())
        bal["upstream"] += float(blk[inside & (s_near < s_extent[0])].sum())
        bal["downstream"] += float(blk[inside & (s_near > s_extent[1])].sum())

        in_s = inside & (s_near >= s_extent[0]) & (s_near <= s_extent[1])

        for i in range(n_cav):
            sel = inside & (near == i)
            if not sel.any():
                continue
            power[i] += float(blk[sel].sum())

            # GENEROUS MASK: every voxel within the capture radius and the
            # s window, with no r <= r_fuel test.  Because the score is
            # identically zero outside fissionable material and no other
            # cavity comes within 67 cm, this sum is the COMPLETE fuel power
            # of cavity i -- and, unlike the strict mask, it is free of
            # radial discretisation: no voxel containing fuel is ever cut.
            cs = in_s & (near == i)
            if cs.any():
                pc = blk[cs]
                cap_power[i] += float(pc.sum())
                sc = np.clip(np.digitize(ss[i][cs], s_edges) - 1, 0, n_s - 1)
                cap_axial[i] += np.bincount(sc, weights=pc, minlength=n_s)
                cap_axial_v[i] += np.bincount(sc, minlength=n_s)
                ac = np.clip(np.digitize(pp[i][cs], phi_edges) - 1, 0, n_phi - 1)
                cap_azim[i] += np.bincount(ac, weights=pc, minlength=n_phi)
                cap_azim_v[i] += np.bincount(ac, minlength=n_phi)

            fs = sel & (s_near >= s_extent[0]) & (s_near <= s_extent[1]) \
                 & (rr[i] <= r_fuel)
            if not fs.any():
                continue
            p = blk[fs]
            fuel_power[i] += float(p.sum())
            fuel_vox[i] += int(fs.sum())

            si = np.clip(np.digitize(ss[i][fs], s_edges) - 1, 0, n_s - 1)
            axial[i] += np.bincount(si, weights=p, minlength=n_s)
            axial_v[i] += np.bincount(si, minlength=n_s)

            ri = np.clip(np.digitize(rr[i][fs], r_edges) - 1, 0, n_r - 1)
            radial[i] += np.bincount(ri, weights=p, minlength=n_r)
            radial_v[i] += np.bincount(ri, minlength=n_r)

            ai = np.clip(np.digitize(pp[i][fs], phi_edges) - 1, 0, n_phi - 1)
            azim[i] += np.bincount(ai, weights=p, minlength=n_phi)
            azim_v[i] += np.bincount(ai, minlength=n_phi)

            om = rr[i][fs] >= r_outer
            if om.any():
                azim_o[i] += np.bincount(ai[om], weights=p[om], minlength=n_phi)
                azim_ov[i] += np.bincount(ai[om], minlength=n_phi)

            if n_rep:
                ps = sig[fs]
                for k in range(n_rep):
                    m = float((p + ps * rng.standard_normal(p.shape)).max())
                    if m > boot[i, k]:
                        boot[i, k] = m

            j = int(np.argmax(p))
            if p[j] > peak[i]:
                peak[i] = float(p[j])
                loc = np.nonzero(fs)
                ix, iy, iz = loc[0][j], loc[1][j], loc[2][j] + z0
                peak_idx[i] = (ix, iy, iz)
                peak_loc[i] = (float(rr[i][fs][j]),
                               float(np.degrees(pp[i][fs][j])),
                               float(ss[i][fs][j]))
                if sig is not None:
                    peak_sig[i] = float(sig[loc[0][j], loc[1][j], loc[2][j]])

        del rr, ss, pp, near, r_near, s_near, inside

    bal["captured"] = float(power.sum())
    bal["in_fuel"] = float(fuel_power.sum())
    return CavityStats(
        power_W=power, cap_power_W=cap_power, cap_axial_W=cap_axial,
        cap_azim_W=cap_azim, cap_axial_vox=cap_axial_v,
        cap_azim_vox=cap_azim_v, fuel_power_W=fuel_power, fuel_voxels=fuel_vox,
        axial_W=axial, axial_vox=axial_v, radial_W=radial, radial_vox=radial_v,
        azim_W=azim, azim_vox=azim_v, azim_outer_W=azim_o,
        azim_outer_vox=azim_ov, s_edges=s_edges, r_edges=r_edges,
        phi_edges=phi_edges, peak_W=peak, peak_sigma_W=peak_sig,
        boot_max_W=np.where(np.isfinite(boot), boot, np.nan),
        peak_index=peak_idx, peak_local=peak_loc, balance=bal)


# =============================================================================
# Part 4 -- peaking factors
# =============================================================================

@dataclass
class PeakingResult:
    axes: List[CavityAxis]
    stats: CavityStats
    voxel_volume_cm3: float
    rebin: Tuple[int, int, int]

    # cavity split
    cavity_power_MW: np.ndarray
    cavity_fraction: np.ndarray
    F_cav: float

    # conditional, telescoping: F_q = F_cav * F_s * F_rphi (exactly)
    F_s: float
    F_rphi: float
    F_q: float

    # marginal shape factors, per cavity
    F_z_per_cavity: np.ndarray
    F_r_per_cavity: np.ndarray
    hot_cavity: int

    # hot spot statistics
    F_q_rel_sigma: Optional[float]
    F_q_bias_estimate: Optional[float]
    F_q_debiased: Optional[float]
    n_fuel_voxels: int
    peak_location_cm: Tuple[float, float, float]
    peak_local_rphis: Tuple[float, float, float]

    # separability
    F_q_separable: float
    nonseparability: float

    # azimuthal harmonics
    tilt_epsilon: np.ndarray
    tilt_phase_deg: np.ndarray
    tilt_epsilon_outer: np.ndarray
    tilt_phase_outer_deg: np.ndarray
    harmonic_2: np.ndarray
    harmonic_6_central: float

    balance_MW: Dict[str, float]
    mean_fuel_density_W_cm3: float
    cavity_density_W_cm3: np.ndarray
    volume_ratio: float = 1.0


def compute_peaking(
    pf: PowerField,
    power_W: float,
    axes: Optional[Sequence[CavityAxis]] = None,
    capture_radius: float = CAPTURE_RADIUS_CM,
    r_fuel: float = R_FUEL_CM,
    s_extent: Tuple[float, float] = (0.0, L_FUEL_CM),
    **binning,
) -> PeakingResult:
    if axes is None:
        # Fit over the fuel only.  Defaulting to the whole mesh drags in the
        # nozzle/exhaust region and any partially filled end slab with it.
        axes, _ = fit_cavity_axes(pf, z_window=tuple(s_extent))
    axes = list(axes)

    st = bin_in_cavity_frame(pf, axes, power_W, capture_radius=capture_radius,
                             r_fuel=r_fuel, s_extent=s_extent, **binning)
    vv = pf.voxel_volume

    # ---- cavity split -------------------------------------------------------
    # F_cav compares mean fuel power DENSITIES, not raw cavity powers.  The
    # two agree when the cavities have equal fuel volume, which they do by
    # design, but the sampled voxel count differs by ~1 % between a vertical
    # and a canted cavity.  Using densities makes F_cav insensitive to that
    # discretisation artefact AND makes F_q = F_cav * F_s * F_rphi hold
    # exactly, since every factor is then a ratio of densities sharing one
    # denominator chain.
    cav = st.fuel_power_W
    frac = cav / cav.sum()

    # ---- marginal shape factors, as power DENSITIES -------------------------
    # Dividing by the sampled voxel count per bin, not by the nominal bin
    # volume, removes the ragged-boundary artefact that otherwise depresses
    # the end bins and inflates F_z.
    def density(p, v):
        out = np.zeros_like(p)
        np.divide(p, np.where(v > 0, v * vv, np.nan), out=out, where=v > 0)
        return out

    # ANALYTIC volumes.  The fuel is a cylinder of radius r_fuel and length
    # (s_hi - s_lo), identical for every cavity, so the volume is known
    # exactly and is the same for all seven.  Counting voxels instead makes
    # the denominator resolution-dependent AND cavity-dependent: the central
    # cavity is aligned with the mesh, so its boundary aliasing repeats
    # identically in every z-plane and never averages out, while the canted
    # ring cavities sweep across the grid and theirs does.  That asymmetry
    # goes straight into F_cav, and it was worth 4 % across a rebin scan of
    # one single statepoint -- i.e. it was pure analysis artefact.
    s_len = float(s_extent[1] - s_extent[0])
    V_an = float(np.pi * r_fuel ** 2 * s_len)
    mean_q = st.cap_power_W / V_an                 # per-cavity mean density

    # PROFILE SHAPES from voxel counts, ABSOLUTE SCALE from the analytic
    # volume.  Dividing a per-bin power by a constant analytic bin volume
    # looks cleaner but is wrong whenever the bin width is incommensurate
    # with the mesh: an s-bin of 3.05 cm on a 2.33 cm mesh catches one or
    # two planes alternately, and the profile alternates 1:2 with it.  The
    # voxel count carries exactly the same aliasing, so the ratio is smooth;
    # a single scalar then puts it on the analytic scale.
    def _shape(P, vox, target):
        q = np.divide(P, np.where(vox > 0, vox, np.nan))
        tot_v, tot_p = vox.sum(), P.sum()
        return q * (target * tot_v / tot_p) if tot_p > 0 else q

    q_ax = np.array([_shape(st.cap_axial_W[i], st.cap_axial_vox[i], mean_q[i])
                     for i in range(len(axes))])
    q_az_cap = np.array([_shape(st.cap_azim_W[i], st.cap_azim_vox[i], mean_q[i])
                         for i in range(len(axes))])
    q_ra = density(st.radial_W, st.radial_vox)     # annuli: voxel counts are
                                                   # the honest volume here
    mean_all = float(st.cap_power_W.sum() / (len(axes) * V_an))
    F_cav = float(mean_q.max() / mean_all)
    hot = int(np.argmax(mean_q))
    cav = st.cap_power_W
    frac = cav / cav.sum()

    mean_vox = np.divide(st.fuel_power_W,
                         np.where(st.fuel_voxels > 0, st.fuel_voxels * vv, np.nan))
    F_z = np.array([np.nanmax(q_ax[i]) / mean_q[i] if mean_q[i] > 0 else np.nan
                    for i in range(len(axes))])
    # F_r is normalised by the voxel-count mean so that numerator and
    # denominator share one volume convention; the outermost annulus is
    # always partly outside the fuel whatever you do.
    F_r = np.array([np.nanmax(q_ra[i]) / mean_vox[i] if mean_vox[i] > 0 else np.nan
                    for i in range(len(axes))])

    # ---- hot spot, conditional decomposition --------------------------------
    peak = float(st.peak_W[hot])
    F_q = peak / vv / mean_all

    s_star = st.peak_local[hot, 2]
    j = int(np.clip(np.digitize(s_star, st.s_edges) - 1, 0,
                    len(st.s_edges) - 2))
    q_star = q_ax[hot, j]
    F_s = float(q_star / mean_q[hot])
    F_rphi = float(peak / vv / q_star)
    # F_cav * F_s * F_rphi == F_q identically; asserted in the self-test.

    sigma_rel = None
    bias = None
    debiased = None
    n_fuel = int(st.fuel_voxels.sum())
    if np.isfinite(st.peak_sigma_W[hot]) and peak > 0:
        sigma_rel = float(st.peak_sigma_W[hot] / peak)
    b = st.boot_max_W[hot]
    if np.isfinite(b).any() and peak > 0:
        # PARAMETRIC BOOTSTRAP.  Adding a second, independent copy of the
        # tally's own noise raises the total noise by sqrt(2); if the
        # selection bias grows roughly linearly in sigma then
        #     max(v + noise) - max(v)  =  bias * (sqrt(2) - 1),
        # which inverts to give the bias already present in max(v).
        #
        # Calibrated against a synthetic field with an exactly known answer
        # (peaking_selftest.py, per-voxel sigma from 5 % to 50 %): this
        # over-estimates the true bias by a stable ~25 %, so the truth sits
        # between F_q(1 - b) and F_q(1 - 0.8 b).  The sqrt(2 ln N) formula
        # that used to sit here over-estimated it by a factor of 2.5-2.8 and
        # could drive the corrected value negative.  For a number that is
        # unbiased by construction rather than by model, use F_q_split_half.
        bias = float((np.nanmean(b) - peak) / peak / (math.sqrt(2.0) - 1.0))
        bias = max(bias, 0.0)
        # Multiplicative, not subtractive: the measured peak IS the true peak
        # inflated by (1 + bias), so the correction divides.  Subtracting
        # works only while the bias is small and goes negative when it is not.
        debiased = float(F_q / (1.0 + bias))

    # ---- separable model ----------------------------------------------------
    # q_sep(r, phi, s) = q_bar * f_s(s) * f_r(r) * f_phi(phi), each marginal
    # normalised to unit mean, so max(q_sep) is the product of the three
    # marginal peaking factors.  All three must be included: leaving f_phi
    # out makes the azimuthal tilt masquerade as r-s non-separability.
    f_phi = _azimuthal_shape_max(q_az_cap[hot],
                                 np.ones(q_az_cap.shape[1]), st.phi_edges)
    F_q_sep = float(F_cav * (np.nanmax(q_ax[hot]) / mean_q[hot])
                    * (np.nanmax(q_ra[hot]) / mean_vox[hot]) * f_phi)
    nonsep = float(F_q / F_q_sep) if F_q_sep > 0 else float("nan")

    # ---- azimuthal harmonics ------------------------------------------------
    eps1, ph1, eps1o, ph1o, eps2 = (np.zeros(len(axes)) for _ in range(5))
    for i in range(len(axes)):
        eps1[i], ph1[i] = _harmonic(st.azim_W[i], st.azim_vox[i], st.phi_edges, 1)
        eps2[i], _ = _harmonic(st.azim_W[i], st.azim_vox[i], st.phi_edges, 2)
        eps1o[i], ph1o[i] = _harmonic(st.azim_outer_W[i], st.azim_outer_vox[i],
                                      st.phi_edges, 1)
    central = int(np.argmin([a.r_machine for a in axes]))
    e6, _ = _harmonic(st.azim_W[central], st.azim_vox[central], st.phi_edges, 6)

    ix, iy, iz = st.peak_index[hot]
    bal = {k: v / 1e6 for k, v in st.balance.items()}
    bal["in_cavity_fuel"] = float(st.cap_power_W.sum()) / 1e6
    v_ratio = float((st.fuel_voxels.sum() * vv) / (len(axes) * V_an))

    return PeakingResult(
        axes=axes, stats=st, voxel_volume_cm3=vv, rebin=pf.rebin,
        cavity_power_MW=cav / 1e6, cavity_fraction=frac, F_cav=F_cav,
        F_s=F_s, F_rphi=F_rphi, F_q=F_q,
        F_z_per_cavity=F_z, F_r_per_cavity=F_r, hot_cavity=hot,
        F_q_rel_sigma=sigma_rel, F_q_bias_estimate=bias, F_q_debiased=debiased,
        n_fuel_voxels=n_fuel,
        peak_location_cm=(float(pf.x[ix]), float(pf.y[iy]), float(pf.z[iz])),
        peak_local_rphis=tuple(float(v) for v in st.peak_local[hot]),
        F_q_separable=F_q_sep, nonseparability=nonsep,
        tilt_epsilon=eps1, tilt_phase_deg=ph1,
        tilt_epsilon_outer=eps1o, tilt_phase_outer_deg=ph1o,
        harmonic_2=eps2, harmonic_6_central=float(e6),
        balance_MW=bal, mean_fuel_density_W_cm3=mean_all,
        cavity_density_W_cm3=mean_q, volume_ratio=v_ratio,
    )


def _harmonic(p: np.ndarray, vox: np.ndarray, edges: np.ndarray,
              m: int) -> Tuple[float, float]:
    """Amplitude and phase of the m-th azimuthal harmonic.

    Works on the power *density* per azimuthal bin, i.e. power divided by
    the number of contributing voxels.  Equal-angle wedges of a disc have
    equal area only if the disc is centred on the reference point; dividing
    by the sampled voxel count makes the estimator robust to that and to
    the ragged mesh boundary.

    For  q(phi) = q0 (1 + eps cos(m phi - phi_m)),  <cos(m phi)>_q = eps/2,
    hence the factor of two.
    """
    good = vox > 0
    if good.sum() < 4:
        return float("nan"), float("nan")
    q = p[good] / vox[good]
    c = 0.5 * (edges[:-1] + edges[1:])[good]
    tot = q.sum()
    if tot <= 0:
        return float("nan"), float("nan")
    a = float(np.sum(q * np.cos(m * c)) / tot)
    b = float(np.sum(q * np.sin(m * c)) / tot)
    return 2.0 * math.hypot(a, b), float(np.degrees(math.atan2(b, a)) / m)


# =============================================================================
# Part 5 -- hot-spot statistics
# =============================================================================

def _azimuthal_shape_max(p: np.ndarray, vox: np.ndarray, edges: np.ndarray,
                         m_max: int = 6) -> float:
    """Peak of the SMOOTH azimuthal shape, reconstructed from its first
    ``m_max`` harmonics and normalised to unit mean.

    Taking max/mean of the raw phi-histogram instead picks up the aliasing
    of a Cartesian voxel grid against the circular fuel boundary: wedges
    that happen to catch more partially filled edge voxels get a different
    mean density.  On the real 4x-rebinned mesh that alone is worth ~14 %
    in a cavity whose azimuthal harmonics are all below 0.1 % -- and it
    propagates straight into the separability test as a fake deficit.
    Projecting onto the first few harmonics discards it, because the
    aliasing is spread over all orders, most of them high.
    """
    good = vox > 0
    if good.sum() < 2 * m_max + 2:
        return float("nan")
    q = p[good] / vox[good]
    c = 0.5 * (edges[:-1] + edges[1:])[good]
    q0 = float(q.mean())
    if q0 <= 0:
        return float("nan")
    phi = np.linspace(-np.pi, np.pi, 361)
    shape = np.ones_like(phi)
    for m in range(1, m_max + 1):
        a = 2.0 * float(np.mean(q * np.cos(m * c))) / q0
        b = 2.0 * float(np.mean(q * np.sin(m * c))) / q0
        shape += a * np.cos(m * phi) + b * np.sin(m * phi)
    return float(shape.max())


def F_q_vs_rebin(statepoint: str, tally, power_W: float,
                 factors: Sequence[int] = (1, 2, 4, 6, 8, 12),
                 axes: Optional[Sequence[CavityAxis]] = None,
                 **kw) -> Dict[int, dict]:
    """F_q as a function of the averaging volume.

    F_q is *not* expected to plateau.  Coarsening removes the extreme-value
    bias and the genuine sub-voxel structure at the same time, so the curve
    falls monotonically.  What you are looking for is the point at which
    the bias-corrected value stops changing while the peak-bin sigma is
    still small -- and the honest report quotes the averaging volume.
    """
    out = {}
    for f in factors:
        pf = load_power_mesh(statepoint, tally, rebin=(f, f, f),
                             load_std_dev=True)
        ax = axes if axes is not None else fit_cavity_axes(pf)[0]
        res = compute_peaking(pf, power_W, axes=ax, **kw)
        out[f] = dict(F_q=res.F_q, F_q_debiased=res.F_q_debiased,
                      sigma=res.F_q_rel_sigma, F_cav=res.F_cav,
                      voxel_cm3=res.voxel_volume_cm3,
                      n_fuel_voxels=res.n_fuel_voxels)
        del pf, res
    print()
    print(f"  {'rebin':>6}{'voxel cm3':>12}{'F_q':>9}{'sigma_pk':>10}"
          f"{'bias':>8}{'F_q corr':>10}{'F_cav':>9}")
    print("  " + "-" * 64)
    for f, d in out.items():
        s = "-" if d["sigma"] is None else f"{d['sigma']*100:.1f}%"
        b = "-" if d["F_q_debiased"] is None else f"{1-d['F_q_debiased']/d['F_q']:.2f}"
        c = "-" if d["F_q_debiased"] is None else f"{d['F_q_debiased']:.3f}"
        print(f"  {f:>6}{d['voxel_cm3']:>12.3f}{d['F_q']:>9.3f}{s:>10}"
              f"{b:>8}{c:>10}{d['F_cav']:>9.4f}")
    print()
    print("  F_cav is the column to trust: it is an integral over ~1e5 voxels")
    print("  and should be flat across the whole scan.  If it is not, the")
    print("  fission source is not converged and no peaking factor from this")
    print("  run means anything.")
    return out


def F_q_split_half(sp_early: str, sp_late: str, tally, power_W: float,
                   axes: Optional[Sequence[CavityAxis]] = None,
                   rebin: Sequence[int] = REBIN, **kw) -> dict:
    """Bias-free hot spot from two statepoints of the same run.

    Locate the peak on the early batches, evaluate it on the later ones.
    Because the two samples are independent, the selection bias vanishes by
    construction -- no ``sqrt(2 ln N)`` model required.

    Requires intermediate statepoints, i.e. in ``_build_settings``:

        settings.statepoint = {'batches': [cfg.inactive +
                                           (cfg.batches - cfg.inactive)//2,
                                           cfg.batches]}
    """
    import openmc

    def _sums(path):
        with openmc.StatePoint(path) as sp:
            t = _find_tally(sp, tally)
            return (np.asarray(t.sum, dtype=np.float64).ravel(),
                    int(t.num_realizations))

    s_e, n_e = _sums(sp_early)
    s_l, n_l = _sums(sp_late)
    if n_l <= n_e:
        raise ValueError("sp_late must contain more realizations than sp_early")

    early = load_power_mesh(sp_early, tally, rebin=rebin, load_std_dev=False)

    # Same reshape-then-single-transpose as load_power_mesh: OpenMC writes
    # mesh bins with x fastest, so (nz, ny, nx) -> transpose(2, 1, 0).
    nz, ny, nx = _orig_shape(sp_late, tally)
    inc = ((s_l - s_e) / (n_l - n_e)).reshape((nz, ny, nx)).transpose(2, 1, 0)
    late = PowerField(
        data=np.ascontiguousarray(block_sum(inc, rebin)),
        x=early.x, y=early.y, z=early.z, dx=early.dx, dy=early.dy, dz=early.dz,
        n_realizations=n_l - n_e, rebin=early.rebin)
    if late.data.shape != early.data.shape:
        raise ValueError(f"shape mismatch {late.data.shape} vs "
                         f"{early.data.shape}: are both statepoints from the "
                         f"same run?")

    ax = axes if axes is not None else fit_cavity_axes(early)[0]
    r_e = compute_peaking(early, power_W, axes=ax, **kw)
    ix, iy, iz = r_e.stats.peak_index[r_e.hot_cavity]

    w = late.scaled(power_W)
    peak_indep = float(w.data[ix, iy, iz]) / late.voxel_volume
    F_q_indep = peak_indep / r_e.mean_fuel_density_W_cm3

    print(f"[peaking] split-half hot spot: located on {n_e} realizations, "
          f"evaluated on {n_l - n_e} independent ones")
    print(f"          F_q(selected sample) = {r_e.F_q:.3f}, "
          f"F_q(independent) = {F_q_indep:.3f}  "
          f"-> selection bias {100*(1 - F_q_indep/r_e.F_q):.1f} %")
    return dict(F_q_biased=r_e.F_q, F_q_unbiased=F_q_indep,
                voxel=(int(ix), int(iy), int(iz)))


def _orig_shape(path, tally):
    import openmc
    with openmc.StatePoint(path) as sp:
        t = _find_tally(sp, tally)
        for f in t.filters:
            if isinstance(f, openmc.MeshFilter):
                nx, ny, nz = (int(v) for v in f.mesh.dimension)
                return (nz, ny, nx)
    raise KeyError("no mesh filter")


# =============================================================================
# Part 6 -- reporting
# =============================================================================

def print_report(res: PeakingResult, power_W: float) -> None:
    b = res.balance_MW
    print()
    print("  " + "=" * 74)
    print("  POWER BALANCE            (normalisation: whole mesh = "
          f"{power_W/1e6:.1f} MW)")
    print("  " + "-" * 74)
    tot = b["total"]
    for key, label in (("in_cavity_fuel", "in the cavities (0 <= s <= L)"),
                       ("in_fuel", "  of which inside r <= R_fuel"),
                       ("captured", "within the capture radius"),
                       ("downstream", "downstream of the exit plane"),
                       ("upstream", "upstream of the inlet plane"),
                       ("outside", "beyond the capture radius")):
        if key in b:
            print(f"    {label:<34s}{b[key]:>10.1f} MW  "
                  f"{100*b[key]/tot:>6.2f} %")
    resid = tot - b["in_cavity_fuel"] - b["downstream"] - b["upstream"] \
        - b["outside"]
    print(f"    {'unattributed':<34s}{resid:>10.1f} MW  "
          f"{100*resid/tot:>6.2f} %")
    print()
    print("  " + "=" * 74)
    print("  CAVITY SPLIT")
    print("  " + "-" * 74)
    print(f"  {'cav':<9}{'x':>8}{'y':>8}{'cant':>7}{'P (MW)':>10}{'frac':>8}"
          f"{'F_z':>8}{'F_r':>8}{'eps_1':>8}{'phase':>8}")
    print("  " + "-" * 74)
    for i, a in enumerate(res.axes):
        print(f"  {a.label:<9}{a.origin[0]:>8.2f}{a.origin[1]:>8.2f}"
              f"{a.cant_deg:>7.2f}{res.cavity_power_MW[i]:>10.1f}"
              f"{res.cavity_fraction[i]:>8.4f}{res.F_z_per_cavity[i]:>8.3f}"
              f"{res.F_r_per_cavity[i]:>8.3f}{res.tilt_epsilon[i]:>8.4f}"
              f"{res.tilt_phase_deg[i]:>8.1f}")
    print("  " + "-" * 74)
    print()
    print("  PEAKING FACTORS")
    print(f"    F_cav   cavity-to-cavity          : {res.F_cav:.4f}")
    print(f"    F_s     axial, within cavity {res.hot_cavity:<2d}    : {res.F_s:.4f}")
    print(f"    F_rphi  in-plane, at the hot plane: {res.F_rphi:.4f}")
    print(f"    ------------------------------------------")
    print(f"    F_q = F_cav * F_s * F_rphi        : {res.F_q:.4f}")
    print(f"            (product check {res.F_cav*res.F_s*res.F_rphi:.4f})")
    print()
    print(f"    averaging volume                  : "
          f"{res.voxel_volume_cm3:.3f} cm3  (rebin {res.rebin})")
    print(f"    fuel voxels in the denominator    : {res.n_fuel_voxels:,}")
    print(f"    mean fuel power density           : "
          f"{res.mean_fuel_density_W_cm3:.1f} W/cm3  "
          f"(analytic volume {np.pi*20.75**2*182.88/1e3:.1f}e3 cm3/cavity)")
    print(f"    sampled/analytic fuel volume      : {res.volume_ratio:.4f}"
          f"   <- how much the voxel grid distorts the mask")
    if res.F_q_rel_sigma is not None:
        print(f"    peak-bin relative sigma           : "
              f"{res.F_q_rel_sigma*100:.2f} %")
    if res.F_q_bias_estimate is not None:
        print(f"    selection bias (bootstrap)        : "
              f"{res.F_q_bias_estimate*100:.1f} %  [conservative by ~25 %]")
        print(f"    F_q, bias-corrected               : {res.F_q_debiased:.4f}"
              f"  ... {res.F_q/(1+0.75*res.F_q_bias_estimate):.4f}")
        if res.F_q_rel_sigma > 0.02:
            print()
            print("    The peak bin carries more than 2 % uncertainty, so the")
            print("    raw F_q above is materially biased high.  Quote the")
            print("    corrected value, or better, use F_q_split_half().")
    print()
    print(f"    peak at global (x, y, z)  : "
          f"({res.peak_location_cm[0]:.1f}, {res.peak_location_cm[1]:.1f}, "
          f"{res.peak_location_cm[2]:.1f}) cm")
    print(f"    peak at local (r, phi, s) : "
          f"({res.peak_local_rphis[0]:.1f} cm, "
          f"{res.peak_local_rphis[1]:.0f} deg, "
          f"{res.peak_local_rphis[2]:.1f} cm)")
    print()
    print(f"    separable-model prediction        : {res.F_q_separable:.4f}")
    print(f"    non-separability  F_q / F_q^sep   : {res.nonseparability:.4f}")
    print("      Ratio of the true hot spot to the product of the three")
    print("      marginal (axial, radial, azimuthal) peaking factors of the")
    print("      hot cavity.  > 1 means the shapes co-peak -- the radial")
    print("      profile sharpens exactly where the axial profile peaks, i.e.")
    print("      genuine corner peaking.  Subtract the hot-spot bias above")
    print("      before attributing any of it to physics.")
    print()
    outer = [i for i, a in enumerate(res.axes) if a.r_machine > 1.0]
    if outer:
        eo = res.tilt_epsilon[outer]
        po = res.tilt_phase_deg[outer]
        eb = res.tilt_epsilon_outer[outer]
        pm = float(np.degrees(np.angle(np.nanmean(
            np.exp(1j * np.radians(po))))))
        print("  AZIMUTHAL TILT, RING CAVITIES")
        print(f"    eps_1 (whole fuel)   : {np.nanmean(eo):.4f} "
              f"+/- {np.nanstd(eo):.4f}   phase {pm:.1f} deg")
        print(f"    eps_1 (outer third)  : {np.nanmean(eb):.4f} "
              f"+/- {np.nanstd(eb):.4f}")
        print(f"    eps_2 (whole fuel)   : "
              f"{np.nanmean(res.harmonic_2[outer]):.4f}")
        print(f"    eps_6, central cavity: {res.harmonic_6_central:.4f}  "
              f"(the six-fold lattice signature; a useful symmetry check)")
        print("    phase 180 deg -> the power leans towards the machine axis;")
        print("    phase   0 deg -> towards the radial reflector.  The outer-")
        print("    third value is the one the transparent wall responds to.")
    print()


def write_csv(res: PeakingResult, path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cavity", "x_inlet_cm", "y_inlet_cm", "cant_deg",
                    "power_MW", "power_fraction", "F_z", "F_r",
                    "eps1", "phase_deg", "eps1_outer", "phase_outer_deg",
                    "eps2"])
        for i, a in enumerate(res.axes):
            w.writerow([a.label, f"{a.origin[0]:.3f}", f"{a.origin[1]:.3f}",
                        f"{a.cant_deg:.3f}",
                        f"{res.cavity_power_MW[i]:.4f}",
                        f"{res.cavity_fraction[i]:.6f}",
                        f"{res.F_z_per_cavity[i]:.4f}",
                        f"{res.F_r_per_cavity[i]:.4f}",
                        f"{res.tilt_epsilon[i]:.4f}",
                        f"{res.tilt_phase_deg[i]:.2f}",
                        f"{res.tilt_epsilon_outer[i]:.4f}",
                        f"{res.tilt_phase_outer_deg[i]:.2f}",
                        f"{res.harmonic_2[i]:.4f}"])
    print(f"[peaking] cavity table -> {path}")


def write_json(res: PeakingResult, path: str) -> None:
    payload = {
        "F_cav": res.F_cav, "F_s": res.F_s, "F_rphi": res.F_rphi,
        "F_q": res.F_q, "F_q_rel_sigma": res.F_q_rel_sigma,
        "F_q_bias_estimate": res.F_q_bias_estimate,
        "F_q_debiased": res.F_q_debiased,
        "F_q_separable": res.F_q_separable,
        "nonseparability": res.nonseparability,
        "hot_cavity": res.hot_cavity,
        "voxel_volume_cm3": res.voxel_volume_cm3,
        "rebin": list(res.rebin),
        "n_fuel_voxels": res.n_fuel_voxels,
        "mean_fuel_density_W_cm3": res.mean_fuel_density_W_cm3,
        "peak_location_cm": list(res.peak_location_cm),
        "peak_local_r_phi_s": list(res.peak_local_rphis),
        "cavity_power_MW": res.cavity_power_MW.tolist(),
        "cavity_fraction": res.cavity_fraction.tolist(),
        "F_z_per_cavity": res.F_z_per_cavity.tolist(),
        "F_r_per_cavity": res.F_r_per_cavity.tolist(),
        "tilt_epsilon": res.tilt_epsilon.tolist(),
        "tilt_phase_deg": res.tilt_phase_deg.tolist(),
        "tilt_epsilon_outer": res.tilt_epsilon_outer.tolist(),
        "harmonic_2": res.harmonic_2.tolist(),
        "harmonic_6_central": res.harmonic_6_central,
        "power_balance_MW": res.balance_MW,
        "axes": [{"label": a.label, "origin": a.origin.tolist(),
                  "direction": a.direction.tolist(),
                  "cant_deg": a.cant_deg} for a in res.axes],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[peaking] factors -> {path}")


def write_latex_table(res: PeakingResult, path: str) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Cavity-wise power distribution and peaking factors at "
        r"\qty{4.6}{\giga\watt}.  All quantities are evaluated in each "
        r"cavity's own frame; \(\varepsilon_1\) is the amplitude of the first "
        r"azimuthal harmonic of the fission power density about the cavity "
        r"axis, with phase measured from the outward radial direction.}",
        r"  \label{tab:peaking}",
        r"  \begin{tabular}{l S[table-format=4.1] S[table-format=1.4] "
        r"S[table-format=1.3] S[table-format=1.3] S[table-format=1.4] "
        r"S[table-format=3.1]}",
        r"    \toprule",
        r"    Cavity & {\(P\), \unit{\mega\watt}} & {Fraction} & "
        r"{\(F_z\)} & {\(F_r\)} & {\(\varepsilon_1\)} & "
        r"{Phase, \unit{\degree}} \\",
        r"    \midrule",
    ]
    for i, a in enumerate(res.axes):
        lines.append(
            f"    {a.label.capitalize()} & {res.cavity_power_MW[i]:.1f} & "
            f"{res.cavity_fraction[i]:.4f} & {res.F_z_per_cavity[i]:.3f} & "
            f"{res.F_r_per_cavity[i]:.3f} & {res.tilt_epsilon[i]:.4f} & "
            f"{res.tilt_phase_deg[i]:.1f} \\\\")
    sig = ("" if res.F_q_rel_sigma is None else
           rf"  The peak bin carries \qty{{{res.F_q_rel_sigma*100:.1f}}}"
           rf"{{\percent}} relative uncertainty; the bias-corrected hot-spot "
           rf"factor is \num{{{res.F_q_debiased:.2f}}}.")
    lines += [
        r"    \midrule",
        rf"    \multicolumn{{7}}{{l}}{{\(F_\text{{cav}} = "
        rf"\num{{{res.F_cav:.3f}}}\), \(F_s = \num{{{res.F_s:.3f}}}\), "
        rf"\(F_{{r\phi}} = \num{{{res.F_rphi:.3f}}}\), "
        rf"\(F_q = \num{{{res.F_q:.3f}}}\)}} \\",
        rf"    \multicolumn{{7}}{{l}}{{\footnotesize Hot spot averaged over "
        rf"\qty{{{res.voxel_volume_cm3:.2f}}}{{\cubic\centi\metre}}."
        rf"{sig}}} \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[peaking] LaTeX table -> {path}")


# =============================================================================
# Part 7 -- plots
# =============================================================================

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_cavity_bars(res: PeakingResult, path: str) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([a.label for a in res.axes], res.cavity_power_MW, color="#3465a4")
    ax.axhline(res.cavity_power_MW.mean(), ls="--", c="k", lw=0.9,
               label=f"mean = {res.cavity_power_MW.mean():.1f} MW")
    ax.set_ylabel("Cavity power (MW)")
    ax.set_title(rf"Cavity power distribution ($F_{{\rm cav}}$ = {res.F_cav:.3f})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"[peaking] figure -> {path}")


def plot_axial_profiles(res: PeakingResult, path: str) -> None:
    plt = _mpl()
    st = res.stats
    s = 0.5 * (st.s_edges[:-1] + st.s_edges[1:])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, a in enumerate(res.axes):
        v, P = st.cap_axial_vox[i], st.cap_axial_W[i]
        q = np.divide(P, np.where(v > 0, v, np.nan))
        q = q * (res.cavity_density_W_cm3[i] * v.sum() / P.sum())
        ax.plot(s, q, lw=2.0 if a.r_machine < 1.0 else 1.0, label=a.label)
    ax.set_xlabel("Distance along the cavity axis, $s$ (cm)")
    ax.set_ylabel(r"Fission power density (W/cm$^3$)")
    ax.set_title("Axial power density in the cavity frame")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"[peaking] figure -> {path}")


def plot_radial_profiles(res: PeakingResult, path: str) -> None:
    plt = _mpl()
    st = res.stats
    r = 0.5 * (st.r_edges[:-1] + st.r_edges[1:])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, a in enumerate(res.axes):
        v = st.radial_vox[i]
        q = np.divide(st.radial_W[i], np.where(v > 0, v * res.voxel_volume_cm3,
                                               np.nan))
        ax.plot(r, q, lw=2.0 if a.r_machine < 1.0 else 1.0, label=a.label)
    ax.set_xlabel("Distance from the cavity axis, $r$ (cm)")
    ax.set_ylabel(r"Fission power density (W/cm$^3$)")
    ax.set_title("Radial power density in the cavity frame")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"[peaking] figure -> {path}")


def plot_azimuthal(res: PeakingResult, path: str) -> None:
    plt = _mpl()
    st = res.stats
    c = 0.5 * (st.phi_edges[:-1] + st.phi_edges[1:])
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="polar")
    for i, a in enumerate(res.axes):
        if a.r_machine < 1.0:
            continue
        v = st.azim_outer_vox[i]
        q = np.divide(st.azim_outer_W[i], np.where(v > 0, v, np.nan))
        q = q / np.nanmean(q)
        ax.plot(np.append(c, c[0]), np.append(q, q[0]), lw=1.0, label=a.label)
    ax.set_theta_zero_location("E")
    ax.set_title("Azimuthal power density, outer third of the fuel\n"
                 r"($\phi=0$ outward, $\phi=180^\circ$ towards the machine axis)",
                 pad=20, fontsize=10)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.28, 1.10))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"[peaking] figure -> {path}")


def plot_axis_fit(fit: AxisFit, axes: Sequence[CavityAxis], path: str) -> None:
    """Slab centroids against the fitted straight lines -- the check that the
    cavity frame is right before any factor is believed."""
    plt = _mpl()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for i in range(fit.track_xy.shape[0]):
        r = np.hypot(fit.track_xy[i, :, 0], fit.track_xy[i, :, 1])
        a1.plot(fit.slab_z, r, "o", ms=3)
    for a in axes:
        zz = np.linspace(fit.slab_z.min(), fit.slab_z.max(), 20)
        pts = np.array([a.origin + (z - a.origin[2]) / a.direction[2] * a.direction
                        for z in zz])
        a1.plot(zz, np.hypot(pts[:, 0], pts[:, 1]), "-", lw=1.0)
    a1.set_xlabel("z (cm)"); a1.set_ylabel("axis radius from the machine axis (cm)")
    a1.set_title("Cavity axes: slab centroids vs straight-line fit")
    a1.grid(alpha=0.3)
    for i in range(fit.track_xy.shape[0]):
        a2.plot(fit.track_xy[i, :, 0], fit.track_xy[i, :, 1], "o-", ms=3, lw=0.7)
    a2.set_aspect("equal"); a2.grid(alpha=0.3)
    a2.set_xlabel("x (cm)"); a2.set_ylabel("y (cm)")
    a2.set_title("Axis tracks in plan view")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"[peaking] figure -> {path}")
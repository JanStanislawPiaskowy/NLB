"""
peaking_selftest.py
===================

Builds a synthetic seven-cavity power field with EXACTLY known answers and
checks that peaking.py recovers them.  No OpenMC, no statepoint: this runs
on a laptop in a few seconds and is the thing to run after any edit to
peaking.py.

The synthetic field is deliberately nasty in the same way the real one is:
the six ring cavities are canted so their axes sweep outwards, the axial
profile is top-peaked (so any fixed-point analysis picks up a spurious
outward dipole), the radial profile is edge-peaked, and a genuine inward
azimuthal tilt of known amplitude is imposed.

Truth values are printed next to the recovered ones.
"""

import sys
import types

import numpy as np

import peaking as pk

# ----------------------------------------------------------------------------
# Truth
# ----------------------------------------------------------------------------
L = 182.88
R2 = 20.75
RING_INLET = 67.33          # ring axis radius in the inlet plane
RING_EXIT = 97.56           # ring axis radius at s = L
EPS_TRUE = 0.20             # first azimuthal harmonic amplitude
PHI0_TRUE = 180.0           # degrees, i.e. leaning towards the machine axis
A_CENTRAL = 1.25            # central cavity amplitude, ring = 1.0
EXHAUST_FRAC = 0.05         # fraction of the fuel power placed downstream

#: Path to the PREVIOUS (fixed-point) peaking.py, for the head-to-head in
#: section 4.  None skips that section.  Once you have overwritten
#: scripts/peaking.py, pull the old one out of git rather than pointing this
#: at the new file:
#:     git show <old-commit>:scripts/peaking.py > /tmp/old_peaking.py
OLD_PEAKING_PATH = None       # e.g. "/tmp/old_peaking.py"

NX = NY = 180
NZ = 120
XY_MAX = 135.0
Z_LO, Z_HI = -39.0, 240.88

CANT = np.arctan((RING_EXIT - RING_INLET) / L)


def f_s(s):
    """Top-peaked axial shape."""
    return 1.0 + 0.40 * np.exp(-((s - 0.75 * L) / (0.35 * L)) ** 2)


def f_r(r):
    """Edge-peaked radial shape."""
    return 1.0 + 1.20 * (r / R2) ** 4


def build_field(noise_rel=0.0, seed=0):
    x = np.linspace(-XY_MAX, XY_MAX, NX + 1)
    y = np.linspace(-XY_MAX, XY_MAX, NY + 1)
    z = np.linspace(Z_LO, Z_HI, NZ + 1)
    dx, dy, dz = np.diff(x)[0], np.diff(y)[0], np.diff(z)[0]
    xc, yc, zc = (0.5 * (a[:-1] + a[1:]) for a in (x, y, z))

    axes = [pk._make_axis([0, 0, 0], [0, 0, 1], "central")]
    for k in range(6):
        a = k * np.pi / 3.0
        u = np.array([np.cos(a), np.sin(a), 0.0])
        axes.append(pk._make_axis(RING_INLET * u,
                                  u * np.tan(CANT) + np.array([0, 0, 1.0]),
                                  f"ring {k}"))

    data = np.zeros((NX, NY, NZ))
    X2, Y2 = np.meshgrid(xc, yc, indexing="ij")
    phi0 = np.radians(PHI0_TRUE)

    for j0 in range(0, NZ, 20):
        j1 = min(j0 + 20, NZ)
        Z3 = np.broadcast_to(zc[j0:j1], (NX, NY, j1 - j0))
        blk = np.zeros((NX, NY, j1 - j0))
        for i, ax in enumerate(axes):
            r, phi, s = ax.to_local(X2[:, :, None], Y2[:, :, None], Z3)
            amp = A_CENTRAL if i == 0 else 1.0
            m = (r <= R2) & (s >= 0) & (s <= L)
            q = amp * f_s(s) * f_r(r) * (1.0 + EPS_TRUE * np.cos(phi - phi0))
            blk += np.where(m, q, 0.0)
            # a downstream "exhaust" tail so the balance has something to find
            me = (r <= R2) & (s > L) & (s <= L + 25.0)
            blk += np.where(me, amp * EXHAUST_FRAC * 8.0, 0.0)
        data[:, :, j0:j1] = blk

    if noise_rel > 0:
        rng = np.random.default_rng(seed)
        data = np.where(data > 0,
                        data * (1.0 + noise_rel * rng.standard_normal(data.shape)),
                        0.0)
        data = np.clip(data, 0.0, None)

    sigma = noise_rel * data if noise_rel > 0 else None
    return pk.PowerField(data=data, x=xc, y=yc, z=zc, dx=dx, dy=dy, dz=dz,
                         sigma=sigma, rebin=(1, 1, 1)), axes


def _load_old_module():
    """Import the previous peaking.py, if one has been pointed at.

    Refuses to run if the file turns out to be the CURRENT module -- which
    is what happens once scripts/peaking.py has been overwritten, and which
    otherwise fails deep inside compute_peaking with a confusing TypeError
    about an unexpected keyword argument.
    """
    import os
    if not OLD_PEAKING_PATH or not os.path.exists(OLD_PEAKING_PATH):
        return None
    sys.modules.setdefault("openmc", types.ModuleType("openmc"))
    import importlib.util
    import inspect
    spec = importlib.util.spec_from_file_location("old_peaking",
                                                  OLD_PEAKING_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["old_peaking"] = mod       # dataclass needs it registered
    spec.loader.exec_module(mod)
    if "centres" not in inspect.signature(mod.compute_peaking).parameters:
        print(f"      {OLD_PEAKING_PATH} is not the old module (no `centres`")
        print("      argument) -- it looks like the current one.")
        return None
    return mod


def _finish(fails):
    print()
    print("=" * 78)
    print(f"  {'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    print("=" * 78)
    return fails


def ok(label, got, want, tol):
    if isinstance(got, bool) or isinstance(want, bool):
        good = bool(got) == bool(want)
        print(f"   {'PASS' if good else 'FAIL'}  {label:<42s} "
              f"got {str(got):>10s}   want {str(want):>10s}")
        return good
    good = abs(got - want) <= tol
    print(f"   {'PASS' if good else 'FAIL'}  {label:<42s} "
          f"got {got:10.4f}   want {want:10.4f}   tol {tol:g}")
    return good


def main():
    print("=" * 78)
    print("  SYNTHETIC FIELD:  cant = %.3f deg, ring radius %.2f -> %.2f cm"
          % (np.degrees(CANT), RING_INLET, RING_EXIT))
    print("=" * 78)
    pf, true_axes = build_field()
    P = 4.6e9
    fails = 0

    # --- 1. axis recovery ---------------------------------------------------
    print("\n1. AXIS RECOVERY FROM THE FIELD ALONE")
    axes, fit = pk.fit_cavity_axes(pf, n_slabs=12,
                                   z_window=(0.0, L * 0.98))
    ring = [a for a in axes if a.r_machine > 1.0]
    fails += not ok("ring cant angle, deg",
                    float(np.mean([a.cant_deg for a in ring])),
                    np.degrees(CANT), 0.35)
    fails += not ok("ring inlet radius, cm",
                    float(np.mean([a.r_machine for a in ring])),
                    RING_INLET, 1.5)
    fails += not ok("central cavity cant, deg", axes[0].cant_deg, 0.0, 0.2)
    fails += not ok("axis fit residual, cm", float(fit.residual_cm.max()),
                    0.0, 0.10)

    # REGRESSION LOCK.  With no z_window the fit sees the downstream exhaust
    # and the partially filled end slabs.  Before outlier rejection, one bad
    # slab at z = 214 cm dragged a 9.39 deg cant down to 7.53 deg -- and the
    # only sign of it was a residual nobody was checking.
    print("   (repeat with NO z_window, i.e. every end slab included)")
    axes_nw, fit_nw = pk.fit_cavity_axes(pf)
    ring_nw = [a for a in axes_nw if a.r_machine > 1.0]
    fails += not ok("ring cant, no z_window, deg",
                    float(np.mean([a.cant_deg for a in ring_nw])),
                    np.degrees(CANT), 0.05)
    fails += not ok("residual, no z_window, cm",
                    float(fit_nw.residual_cm.max()), 0.0, 0.10)
    fails += not ok("fuel radius from the support, cm",
                    float(np.mean(fit.support_radius_cm)), R2, 0.15)

    # --- 2. peaking factors with the correct (true) axes --------------------
    print("\n2. PEAKING FACTORS, CAVITY FRAME, TRUE AXES")
    res = pk.compute_peaking(pf, P, axes=true_axes, capture_radius=30.0,
                             r_fuel=R2, s_extent=(0.0, L))
    F_cav_true = A_CENTRAL / ((A_CENTRAL + 6.0) / 7.0)
    fails += not ok("F_cav (discretisation-limited)", res.F_cav, F_cav_true, 0.01)
    fails += not ok("F_q = F_cav * F_s * F_rphi (identity)",
                    res.F_cav * res.F_s * res.F_rphi, res.F_q, 1e-9)
    fails += not ok("eps_1, ring mean",
                    float(np.mean(res.tilt_epsilon[1:])), EPS_TRUE, 0.01)
    ph = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(
        res.tilt_phase_deg[1:])))))
    fails += not ok("eps_1 phase, ring mean, deg", abs(ph), PHI0_TRUE, 3.0)
    fails += not ok("eps_2, ring mean (should vanish)",
                    float(np.mean(res.harmonic_2[1:])), 0.0, 0.02)
    fails += not ok("non-separability (field IS separable)",
                    res.nonseparability, 1.0, 0.02)

    # --- 3. power balance ---------------------------------------------------
    print("\n3. POWER BALANCE")
    b = res.balance_MW
    lost = b["outside"] / b["total"]
    down = b["downstream"] / b["total"]
    fails += not ok("power beyond the capture radius, frac", lost, 0.0, 1e-6)
    print(f"         downstream of the exit plane: {down*100:.2f} % "
          f"(synthetic exhaust; the real run should show something similar)")
    print(f"         in the fuel mask            : "
          f"{b['in_fuel']/b['total']*100:.2f} %")

    # --- 4. head to head against the old fixed-point algorithm --------------
    print("\n4. THE OLD FIXED-POINT ALGORITHM ON THE SAME FIELD")
    old = _load_old_module()
    if old is None:
        print("      skipped -- set OLD_PEAKING_PATH to compare against the")
        print("      previous fixed-point implementation.")
    else:
        fails += _compare_with_old(old, pf, res, b, P, F_cav_true)

    # --- 5. hot-spot selection bias -----------------------------------------
    print("\n5. HOT-SPOT SELECTION BIAS (noise added, physics unchanged)")
    F_q_clean = res.F_q
    for nz_ in (0.05, 0.15, 0.30):
        pfn, _ = build_field(noise_rel=nz_, seed=1)
        rn = pk.compute_peaking(pfn, P, axes=true_axes, capture_radius=30.0,
                                r_fuel=R2, s_extent=(0.0, L))
        print(f"      per-voxel sigma {nz_*100:4.0f} %  ->  F_q = {rn.F_q:.3f} "
              f"(noise-free {F_q_clean:.3f}, inflated "
              f"{100*(rn.F_q/F_q_clean-1):+.1f} %), "
              f"corrected {rn.F_q_debiased:.3f}")
    print(f"      F_cav is untouched by the same noise: {rn.F_cav:.4f} "
          f"vs {res.F_cav:.4f} noise-free")

    return _finish(fails)


def _compare_with_old(old, pf, res, b, P, F_cav_true):
    fails = 0
    old_pf = old.PowerField(data=pf.data, x=pf.x, y=pf.y, z=pf.z,
                            dx=pf.dx, dy=pf.dy, dz=pf.dz)
    old_res = old.compute_peaking(old_pf, P, centres=None,
                                  capture_radius=35.0,
                                  z_active=(0.0, L),
                                  threshold_frac=0.02)
    eps_old = float(np.mean(old_res.tilt_epsilon[1:]))
    ph_old = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(
        old_res.tilt_phase_deg[1:])))))
    cap = old_res.cavity_power_MW.sum() / (P / 1e6)
    print(f"      old F_cav                 : {old_res.F_cav:.4f}   "
          f"(truth {F_cav_true:.4f}, error {100*(old_res.F_cav/F_cav_true-1):+.1f} %)")
    print(f"      old eps_1, ring mean      : {eps_old:.4f}   "
          f"(truth {EPS_TRUE:.4f}, error {100*(eps_old/EPS_TRUE-1):+.1f} %)")
    print(f"      old eps_1 phase           : {ph_old:+.1f} deg  "
          f"(truth {PHI0_TRUE:.0f} deg)")
    print(f"      old power accounted for   : {cap*100:.2f} % of the total")
    print(f"      new F_cav                 : {res.F_cav:.4f}")
    print(f"      new eps_1, ring mean      : "
          f"{float(np.mean(res.tilt_epsilon[1:])):.4f}")
    print(f"      new power accounted for   : "
          f"{(b['in_fuel']+b['downstream'])/b['total']*100:.2f} % "
          f"(itemised, nothing dropped)")
    fails += not ok("new eps_1 beats old eps_1",
                    abs(float(np.mean(res.tilt_epsilon[1:])) - EPS_TRUE)
                    < abs(eps_old - EPS_TRUE), True, 0)
    return fails


if __name__ == "__main__":
    sys.exit(main())
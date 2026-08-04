"""
run_peaking.py
==============

Post-processing only.  Point it at a statepoint containing the
``fission-q-recoverable`` mesh tally and it produces the power balance, the
cavity split, the peaking factors, the azimuthal harmonics and five
figures.  No re-run, no new tallies.

ORDER OF OPERATIONS -- do these once, in this order, before quoting anything

  1. AXES=... from the geometry (see below) or CHECK_AXES = True, and look at
     ``axis_fit.pdf``.  If the fitted cant angle does not match the geometry
     to a few hundredths of a degree, nothing downstream is meaningful.

  2. REBIN_SCAN = True.  F_cav must be flat across the whole scan; it is an
     integral over ~1e5 voxels and has no business moving.  If it moves, the
     fission source is not converged and no peaking factor from this run
     means anything -- check the Shannon entropy trace first.

  3. Only then the main run, and quote F_q with its averaging volume and its
     bias correction.
"""

from __future__ import annotations

import os

import numpy as np

import peaking

# =============================================================================
# SETTINGS
# =============================================================================

STATEPOINT = "reference_runs/critical_nophoton/statepoint.250.h5"
POWER_TALLY = "power_distribution"          # tally name, or an integer id
POWER_W = 4.6e9
OUT_DIR = "peaking_run"

REBIN = (2, 2, 2)                # block-sum factor applied on load
LOAD_STD_DEV = True              # needed for the hot-spot bias estimate

# --- Cavity frame -------------------------------------------------------------
# Preferred: take the axes straight from the geometry you built.  Set
# USE_MODEL_AXES = True and this uses cavity_placements(cfg), which is exact.
# Otherwise the axes are fitted from the tally support, which is accurate to
# about a millimetre but should be checked against the geometry once.
USE_MODEL_AXES = False
CONFIG_JSON = None               # e.g. ".../baseline/gcr_config.json"

R_FUEL_CM = 20.75                # GCRConfig.R2
L_FUEL_CM = 182.88               # GCRConfig.L
CAPTURE_RADIUS_CM = 30.0         # perpendicular to the cavity axis, so this
                                 # no longer has to grow to cover the cant
S_EXTENT_CM = (0.0, L_FUEL_CM)   # along the cavity axis, not global z

CHECK_AXES = True                # write axis_fit.pdf and stop to look at it
REBIN_SCAN = False               # F_q and F_cav versus averaging volume
REBIN_SCAN_FACTORS = (2, 4, 6, 8, 12)

# --- Optional: bias-free hot spot from two statepoints of the same run --------
SPLIT_HALF_EARLY = None          # e.g. ".../statepoint.137.h5"

MAKE_PLOTS = True


# =============================================================================

def _axes_from_model():
    """Exact axes from the model's own placement function."""
    from gcr.config import GCRConfig
    from gcr.geometry.hexmaths import cavity_placements
    cfg = (GCRConfig.from_json(CONFIG_JSON) if CONFIG_JSON else GCRConfig())
    return peaking.axes_from_placements(cavity_placements(cfg))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if REBIN_SCAN:
        peaking.F_q_vs_rebin(
            STATEPOINT, POWER_TALLY, POWER_W,
            factors=REBIN_SCAN_FACTORS,
            axes=_axes_from_model() if USE_MODEL_AXES else None,
            capture_radius=CAPTURE_RADIUS_CM, r_fuel=R_FUEL_CM,
            s_extent=S_EXTENT_CM)
        return

    pf = peaking.load_power_mesh(
        STATEPOINT, POWER_TALLY, rebin=REBIN, load_std_dev=LOAD_STD_DEV)

    if USE_MODEL_AXES:
        axes = _axes_from_model()
        print("[run] using exact axes from cavity_placements()")
    else:
        axes, fit = peaking.fit_cavity_axes(pf, z_window=(0.0, L_FUEL_CM))
        if CHECK_AXES:
            peaking.plot_axis_fit(fit, axes,
                                  os.path.join(OUT_DIR, "axis_fit.pdf"))

    res = peaking.compute_peaking(
        pf, POWER_W, axes=axes,
        capture_radius=CAPTURE_RADIUS_CM,
        r_fuel=R_FUEL_CM,
        s_extent=S_EXTENT_CM,
    )

    peaking.print_report(res, POWER_W)
    peaking.write_csv(res, os.path.join(OUT_DIR, "cavity_summary.csv"))
    peaking.write_json(res, os.path.join(OUT_DIR, "peaking_factors.json"))
    peaking.write_latex_table(res, os.path.join(OUT_DIR, "peaking_table.tex"))

    if MAKE_PLOTS:
        peaking.plot_cavity_bars(res, os.path.join(OUT_DIR, "cavity_power.pdf"))
        peaking.plot_axial_profiles(res, os.path.join(OUT_DIR, "cavity_axial.pdf"))
        peaking.plot_radial_profiles(res, os.path.join(OUT_DIR, "cavity_radial.pdf"))
        peaking.plot_azimuthal(res, os.path.join(OUT_DIR, "cavity_azimuthal.pdf"))

    if SPLIT_HALF_EARLY:
        peaking.F_q_split_half(
            SPLIT_HALF_EARLY, STATEPOINT, POWER_TALLY, POWER_W,
            axes=axes, rebin=REBIN,
            capture_radius=CAPTURE_RADIUS_CM, r_fuel=R_FUEL_CM,
            s_extent=S_EXTENT_CM)


if __name__ == "__main__":
    main()

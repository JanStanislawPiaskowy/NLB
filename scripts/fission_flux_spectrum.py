"""
fission_flux_spectrum.py
========================

Tally and plot the energy distribution of fissions inside the GCR fuel
-- "the flux that contributes to fission" -- comparing two nuclear data
libraries (JEFF-4.0 and TENDL-2025) at several BeO reflector temperatures.

We score `flux` and `fission` on a 500-bin log-spaced energy filter from
1e-5 eV to 20 MeV, restricted to the fuel materials via a MaterialFilter.
The 'fission' score is integral of phi(E) * Sigma_f(E) over the fuel
volume in each energy bin -- i.e. the energy distribution of fission
events themselves.

The script runs every (library, temperature) combination:
    libraries    : JEFF-4.0, TENDL-2025
    temperatures : baseline, BeO at 1000 K, BeO at 2000 K
Each combination is written to its own subdirectory
(fission_spectrum_run/<temp_tag>/<lib_tag>) so runs can be re-plotted
without re-running.

Post-processing produces ONE figure per BeO temperature. Each figure has
two stacked panels (fission rate per lethargy on top, in-fuel flux per
lethargy below) with one curve per library, so JEFF-4.0 and TENDL-2025 are
overlaid directly for that temperature.

Note on 2000 K: JEFF-4.0 / TENDL BeO S(alpha, beta) tables typically top
out at ~1200 K. Above that the cross-section interpolator either accepts
the top tabulated value (if your tolerance permits) or warns. The 2000 K
curve is therefore physically extrapolated and shows the *trend*, not a
quantitatively faithful 2000 K result.

Drop next to GCR.py and sensitivity_analysis.py and run
    python fission_flux_spectrum.py             # run all 6 + plot
    python fission_flux_spectrum.py --plot-only # re-plot from statepoints
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import openmc

from GCR import GCR, GCRConfig


# ---------------------------------------------------------------------------
# Configuration -- mirrors sensitivity_analysis.py for consistency.
# ---------------------------------------------------------------------------

# cross_sections_dir is now injected per-library in run_one_case(), so it is
# deliberately NOT listed here.
BASE_CONFIG_KWARGS = dict(
    n_axial_layers=10,
    h2_density_profile_path='settings/h2_density_profile.npz',
)

OUTPUT_DIR  = 'fission_spectrum_run'
N_BATCHES   = 150
N_INACTIVE  = 25
N_PARTICLES = 150_000

# Libraries to compare. Each entry is (lib_tag, xs_dir, label, color).
# The colour distinguishes the two libraries *within* a single figure.
# NOTE: set the TENDL path to wherever your TENDL-2025 HDF5 build lives.
LIBRARIES = [
    ('jeff40', 'libraries_xs/jeff40_hdf5',    'JEFF-4.0',   'C0'),
    ('tendl',  'libraries_xs/tendl2025_hdf5', 'TENDL-2025', 'C3'),
]

# BeO reflector temperatures -- one *figure* per entry.
# Each entry is (temp_tag, beo_T_kelvin_or_None, label).
# beo_T = None means "leave BeO at whatever set_materials() chose" -- this
# is the unmodified baseline. A numeric value overrides BeO's temperature
# in-place after the materials are built.
TEMPERATURES = [
    ('baseline',  None,  'baseline (unmodified)'),
    ('beo_1000K', 1000., r'BeO at 1000 K'),
    ('beo_2000K', 2000., r'BeO at 2000 K'),
]

# Energy grid for the spectrum: log-spaced, 1e-5 eV to 20 MeV, 500 bins.
N_E_BINS = 500
E_MIN    = 1.0e-5     # eV
E_MAX    = 2.0e7      # eV  (20 MeV)

# Boundaries for the thermal / intermediate / fast bookkeeping
E_THERMAL_HI = 0.625    # eV  (cadmium cutoff)
E_FAST_LO    = 1.0e5    # eV


# ---------------------------------------------------------------------------
# Material classification (matches sensitivity_analysis.py)
# ---------------------------------------------------------------------------

def _is_fuel(mat):
    name = mat.name or ''
    return (name == 'fuel'
            or name.startswith('fuel_inner')
            or name.startswith('fuel_outer'))


def _is_beo_reflector(mat):
    # set_materials() creates a single material called 'BeO' that fills
    # the inter-cavity moderator slabs and the top cap.
    return (mat.name or '') == 'BeO'


def _set_beo_temperature(core, T_kelvin):
    """Override BeO material temperature in-place. Density unchanged --
    solid BeO's thermal expansion (~10^-5 /K) is small compared with the
    S(alpha, beta) temperature effect that dominates moderator feedback.
    """
    n_set = 0
    for mat in _unique_materials(core):
        if _is_beo_reflector(mat):
            mat.temperature = float(T_kelvin)
            n_set += 1
    if n_set == 0:
        raise RuntimeError('No BeO material found to override temperature.')
    print(f'  -> BeO temperature set to {T_kelvin:g} K on {n_set} material(s).')


def _unique_materials(core):
    """Yield each material once, de-duplicating by id().

    _create_layered_fuel_materials() aliases 'fuel_inner' / 'fuel_outer' to
    the layer-0 objects, so the same object appears twice in core.materials.
    """
    seen = set()
    for mat in core.materials.values():
        if id(mat) in seen:
            continue
        seen.add(id(mat))
        yield mat


# ---------------------------------------------------------------------------
# Build core + geometry. Mirrors sensitivity_analysis.build_core /
# build_geometry_and_export, with the critical-state fuel scaling baked in.
# ---------------------------------------------------------------------------

def build_core(config, output_dir, beo_T=None):
    """Build the GCR core, apply critical-state fuel scaling, and optionally
    override the BeO reflector temperature."""
    core = GCR(config)
    core.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    core.set_materials()
    if config.n_axial_layers > 1:
        core._create_layered_propellant_materials()
        core._create_layered_fuel_materials()

    # Same FUEL_DENSITY_ALPHA as sensitivity_analysis.build_core -- gets us
    # to (near-)criticality before we tally.
    FUEL_DENSITY_ALPHA = 0.2586
    for mat in _unique_materials(core):
        if _is_fuel(mat):
            mat.set_density('g/cm3', mat.density * FUEL_DENSITY_ALPHA)

    if beo_T is not None:
        _set_beo_temperature(core, beo_T)
    core.apply_beo_sab()

    return core


def build_geometry(core):
    cfg = core.config

    core.build_cavity(tilt=cfg.tilt)

    sixty_deg = np.pi / 3
    hex_side_to_radius = 1.4
    hl  = cfg.r_inlet * hex_side_to_radius
    phi = 2 * cfg.tilt

    for i in range(6):
        y0 = hl * np.sin(sixty_deg) * (1 + np.cos(phi))
        z0 = hl * np.sin(sixty_deg) * np.sin(phi)
        theta = -i * np.pi / 3
        xp = -y0 * np.sin(theta)
        yp =  y0 * np.cos(theta)
        zp = z0
        core.build_cavity(
            x0=xp, y0=yp, z0=zp,
            tilt=cfg.tilt,
            cavity_angle_zz=theta,
            cavity_angle_xx=phi,
        )

    z_off = np.sin(sixty_deg) * hl / np.tan(phi)
    core.create_bounding_sphere(offset=z_off)
    core.build_moderator()
    core.build_end_moderator()
    core.build_nozzle_end()
    core.resolve_cavity_overlaps()
    core.set_source(batches=N_BATCHES, inactive=N_INACTIVE, n=N_PARTICLES)
    core.settings.temperature = {
        'method': 'interpolation',
        'tolerance': 300.0,
        'multipole': False,
    }
    core.settings.export_to_xml(os.path.join(core.output_dir, 'settings.xml'))
    core.export_geometry()


# ---------------------------------------------------------------------------
# Tally setup
# ---------------------------------------------------------------------------

def add_fission_spectrum_tally(core):
    """Energy-binned tally of flux and fission, restricted to fuel materials."""
    energy_bins = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_E_BINS + 1)

    fuel_mats = [m for m in _unique_materials(core) if _is_fuel(m)]
    if not fuel_mats:
        raise RuntimeError('No fuel materials found -- nothing to tally.')

    energy_filter   = openmc.EnergyFilter(energy_bins)
    material_filter = openmc.MaterialFilter(fuel_mats)

    tally = openmc.Tally(name='fuel_spectrum')
    tally.filters = [material_filter, energy_filter]
    tally.scores  = ['flux', 'fission']

    openmc.Tallies([tally]).export_to_xml(
        os.path.join(core.output_dir, 'tallies.xml')
    )
    print(f'Spectrum tally added: {N_E_BINS} log bins from '
          f'{E_MIN:.1e} to {E_MAX:.1e} eV across {len(fuel_mats)} fuel materials.')
    return energy_bins


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _energy_grid(energy_bins):
    E_lo  = energy_bins[:-1]
    E_hi  = energy_bins[1:]
    E_mid = np.sqrt(E_lo * E_hi)         # geometric mid for log-spaced bins
    du    = np.log(E_hi / E_lo)          # lethargy width per bin
    return E_lo, E_hi, E_mid, du


def _read_spectrum(output_dir, energy_bins, batches=N_BATCHES):
    """Read fuel-spectrum tally from the statepoint in `output_dir` and
    return per-bin (flux, fission, flux_se, fission_se) summed over fuel
    materials. This is generic enough to point at any of the case
    directories produced by main()."""
    sp_path = os.path.join(output_dir, f'statepoint.{batches}.h5')
    sp = openmc.StatePoint(sp_path)
    tally = sp.get_tally(name='fuel_spectrum')

    # tally.mean shape: (n_filter_bins, n_nuclides, n_scores). Filter axis
    # iterates outer-to-inner in the order of tally.filters, so with
    # filters = [material, energy] the layout is (n_mat * n_E, ...).
    mean   = tally.mean   [:, 0, :]      # collapse the (single) nuclide axis
    stddev = tally.std_dev[:, 0, :]

    score_idx = {s: i for i, s in enumerate(tally.scores)}
    flux_col    = score_idx['flux']
    fission_col = score_idx['fission']

    n_E   = len(energy_bins) - 1
    n_mat = mean.shape[0] // n_E
    if n_mat * n_E != mean.shape[0]:
        raise RuntimeError(
            f'Tally row count {mean.shape[0]} not a multiple of n_E={n_E}.'
        )
    mean_3d   = mean.  reshape(n_mat, n_E, -1)
    stddev_3d = stddev.reshape(n_mat, n_E, -1)

    flux_E      = mean_3d  [:, :, flux_col   ].sum(axis=0)
    fission_E   = mean_3d  [:, :, fission_col].sum(axis=0)
    flux_E_se   = np.sqrt((stddev_3d[:, :, flux_col   ] ** 2).sum(axis=0))
    fission_se  = np.sqrt((stddev_3d[:, :, fission_col] ** 2).sum(axis=0))

    sp.close()
    return flux_E, fission_E, flux_E_se, fission_se


def _bookkeeping(E_mid, fission_E):
    """Print fraction of fissions falling in thermal / epithermal / fast."""
    total = fission_E.sum()
    if total <= 0:
        print('  (no fissions tallied -- something is wrong)')
        return
    therm = fission_E[E_mid <  E_THERMAL_HI].sum()
    fast  = fission_E[E_mid >= E_FAST_LO  ].sum()
    epi   = total - therm - fast
    print(f'  thermal      (E < {E_THERMAL_HI} eV)            : {therm/total*100:6.2f} %')
    print(f'  epithermal ({E_THERMAL_HI} eV <= E < {E_FAST_LO:.0e} eV): '
          f'{epi/total*100:6.2f} %')
    print(f'  fast         (E >= {E_FAST_LO:.0e} eV)         : {fast/total*100:6.2f} %')


def _plot_one_figure(series, energy_bins, title, pdf_path, dynamic_range=1e-6):
    """Render one two-panel figure (fission rate per lethargy on top,
    in-fuel flux per lethargy below) from a list of
    (label, color, fission_per_u, flux_per_u) tuples -- one curve per entry.
    """
    E_lo, E_hi, E_mid, du = _energy_grid(energy_bins)

    f_peak   = max(np.max(s[2]) for s in series)
    phi_peak = max(np.max(s[3]) for s in series)
    f_ymin,   f_ymax   = f_peak   * dynamic_range, f_peak   * 2
    phi_ymin, phi_ymax = phi_peak * dynamic_range, phi_peak * 2

    fig, (ax_f, ax_phi) = plt.subplots(
        2, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw=dict(hspace=0.06),
    )

    # --- Top: fission rate per lethargy ----------------------------------
    for label, color, fission_per_u, _ in series:
        ax_f.step(E_mid, fission_per_u, where='mid',
                  color=color, lw=1.5, label=label)
    ax_f.set_xscale('log')
    ax_f.set_yscale('log')
    ax_f.set_ylim(f_ymin, f_ymax)
    ax_f.set_ylabel('Fission rate per lethargy\n(per source neutron)')
    ax_f.set_title(title)
    ax_f.grid(which='both', alpha=0.3)
    ax_f.legend(loc='lower center', ncol=len(series))

    # --- Bottom: in-fuel flux per lethargy -------------------------------
    for label, color, _, flux_per_u in series:
        ax_phi.step(E_mid, flux_per_u, where='mid',
                    color=color, lw=1.4, label=label)
    ax_phi.set_xscale('log')
    ax_phi.set_yscale('log')
    ax_phi.set_ylim(phi_ymin, phi_ymax)
    ax_phi.set_xlabel('Neutron energy (eV)')
    ax_phi.set_ylabel(r'Flux per lethargy' '\n' r'(cm$^{-2}$ per source)')
    ax_phi.grid(which='both', alpha=0.3)
    ax_phi.legend(loc='lower center', ncol=len(series))

    # --- Thermal / fast guide bands on both panels ------------------------
    for ax in (ax_f, ax_phi):
        ax.axvspan(E_lo[0],   E_THERMAL_HI, color='gray', alpha=0.07, zorder=0)
        ax.axvspan(E_FAST_LO, E_hi[-1],     color='gray', alpha=0.07, zorder=0)
    ax_f.text(np.sqrt(E_lo[0] * E_THERMAL_HI), 0.96, 'thermal',
              transform=ax_f.get_xaxis_transform(), ha='center', va='top',
              fontsize=9, color='gray')
    ax_f.text(np.sqrt(E_FAST_LO * E_hi[-1]), 0.96, 'fast',
              transform=ax_f.get_xaxis_transform(), ha='center', va='top',
              fontsize=9, color='gray')

    fig.tight_layout()
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight')
    print(f'  written {pdf_path}')
    plt.close(fig)
    return fig


def plot_per_temperature(parent_dir, libraries, temperatures, energy_bins,
                         dynamic_range=1e-6):
    """One figure per BeO temperature, each overlaying both libraries.

    Reads each (temperature, library) statepoint from
    `parent_dir/<temp_tag>/<lib_tag>/statepoint.<N_BATCHES>.h5`.

    `libraries`    is a list of (lib_tag, xs_dir, label, color) tuples
                   (xs_dir is ignored here -- only tag/label/color are used).
    `temperatures` is a list of (temp_tag, beo_T, label) tuples
                   (beo_T is ignored here -- only tag/label are used).
    """
    _, _, E_mid, du = _energy_grid(energy_bins)

    figs = []
    for temp_tag, _beo_T, temp_label in temperatures:
        print(f'\n{"="*70}\nFigure: {temp_label}  ({temp_tag})\n{"="*70}')

        series = []
        for lib_tag, _xs_dir, lib_label, lib_color in libraries:
            case_dir = os.path.join(parent_dir, temp_tag, lib_tag)
            flux_E, fission_E, _, _ = _read_spectrum(case_dir, energy_bins)
            flux_per_u    = flux_E    / du
            fission_per_u = fission_E / du
            print(f'\nLibrary "{lib_label}" ({lib_tag}):')
            _bookkeeping(E_mid, fission_E)
            series.append((lib_label, lib_color, fission_per_u, flux_per_u))

        title    = f'GCR fission spectrum -- {temp_label} (library comparison)'
        pdf_path = os.path.join(parent_dir,
                                f'fission_flux_spectrum_{temp_tag}.pdf')
        figs.append(_plot_one_figure(series, energy_bins, title, pdf_path,
                                     dynamic_range))

    print(f'\n{len(figs)} figure(s) written to {parent_dir}/')
    return figs


# ---------------------------------------------------------------------------

def run_one_case(lib_tag, xs_dir, temp_tag, beo_T):
    """Build, set up tally, run OpenMC for a single (library, temperature)
    case. Returns the energy-bin array used (identical for every case)."""
    case_dir = os.path.join(OUTPUT_DIR, temp_tag, lib_tag)
    print(f'\n{"="*70}\n'
          f'Case: lib={lib_tag}  temp={temp_tag}  (BeO T = {beo_T!r})\n'
          f'      xs_dir = {xs_dir}\n'
          f'{"="*70}')

    config = GCRConfig(cross_sections_dir=xs_dir, **BASE_CONFIG_KWARGS)
    core   = build_core(config, case_dir, beo_T=beo_T)
    build_geometry(core)
    energy_bins = add_fission_spectrum_tally(core)
    core.run(dry_run=False, map_geometry=False)
    return energy_bins


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    energy_bins = None
    for lib_tag, xs_dir, _lib_label, _lib_color in LIBRARIES:
        for temp_tag, beo_T, _temp_label in TEMPERATURES:
            energy_bins = run_one_case(lib_tag, xs_dir, temp_tag, beo_T)
    plot_per_temperature(OUTPUT_DIR, LIBRARIES, TEMPERATURES, energy_bins)


def plot_only(parent_dir=OUTPUT_DIR):
    """Re-do only the plots from existing statepoints -- no new OpenMC runs."""
    energy_bins = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_E_BINS + 1)
    plot_per_temperature(parent_dir, LIBRARIES, TEMPERATURES, energy_bins)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='GCR fission flux spectrum: JEFF-4.0 vs TENDL across '
                    'BeO temperatures.')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip the OpenMC runs; re-plot from existing '
                             'statepoints in '
                             'fission_spectrum_run/<temp_tag>/<lib_tag>/.')
    args = parser.parse_args()
    if args.plot_only:
        plot_only()
    else:
        main()
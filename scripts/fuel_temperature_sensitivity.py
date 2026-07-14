"""
fuel_temperature_sensitivity_delta.py
=====================================

Delta-based fuel-temperature sensitivity analysis for the GCR model in
GCR.py.

This is a variant of fuel_temperature_sensitivity.py. Instead of the
anchor + fractional-ratio mechanism (set every fuel material to the same
absolute T_a, then perturb by +-p*T_a), this script:

  * keeps the model's *native* per-layer fuel-temperature profile, i.e.
    the radiation-equilibrium temperatures written by
    GCR._create_layered_fuel_materials (fuel_temperature_from_h2 per
    layer), and
  * rigidly shifts that whole profile by a single additive offset
    Delta T -- the SAME delta applied to every fuel layer -- one delta
    per OpenMC run.

For each delta in FUEL_T_DELTAS_K, every fuel material's temperature is
set to

        T_layer_new = T_layer_baseline + Delta T

so each layer retains its distinct baseline temperature and only a common
offset is applied. The delta = 0 case reproduces the unshifted production
model exactly and serves as the reactivity reference.

The reactivity of each case is measured relative to the delta = 0 baseline,

        rho(Delta T) = ( k(Delta T) - k_0 ) / ( k(Delta T) * k_0 )   [pcm]

and the reactivity-vs-delta curve is fit with a linear and a quadratic
model. The linear slope

        alpha = d rho / d(Delta T)                                   [pcm / K]

is the uniform-shift fuel-temperature coefficient (a single global number,
since the perturbation is a rigid shift of the real profile rather than a
collapse to one temperature). A local central-difference slope at the
smallest symmetric +-delta is also reported.

Default workload = len(FUEL_T_DELTAS_K) OpenMC runs (one per delta),
NOT 3 runs per anchor.

Differences from the anchor version
-----------------------------------
* Perturbation is additive (a rigid Delta T on the whole profile), not a
  multiplicative fraction of a single anchor temperature.
* The per-layer profile is preserved; the fuel is NOT flattened to a
  uniform temperature.
* One run per delta; reactivity is global (relative to the single
  delta = 0 baseline), so alpha is one number with a real multi-point fit
  (7 default points -> genuine degrees of freedom), not a per-anchor slope.

Doppler vs. combined feedback
-----------------------------
Set COUPLE_DENSITY_TO_FUEL_T = True to rescale each fuel layer's density
by T_old/T_new (ideal-gas, constant pressure -- the physical gas-core
feedback). NOTE: the baseline layer densities come from the real-gas EOS
in GCR; this coupling applies an *ideal-gas* 1/T correction on top of the
shift, matching the anchor script's semantics. For a fully consistent
real-gas density at the shifted temperature you would re-query
gcnr.eos.UraniumEOS at (T_baseline + delta); that is intentionally NOT done
here so the toggle stays comparable to the anchor study. The default
(False) holds density fixed and isolates the cross-section temperature
response (the "Doppler-like" coefficient from the library's sqrt(T)
interpolation).

Cross-section caveat
--------------------
OpenMC interpolates cross sections in sqrt(T) between processed library
temperatures. A delta of -5000 K lowers the coolest layer by 5000 K and a
delta of +5000 K raises the hottest layer by 5000 K, so the union of
sampled temperatures is roughly (T_min - 5000) ... (T_max + 5000). Make
sure the cross-section library covers this band -- otherwise OpenMC will
raise (the script sets a 300 K interpolation tolerance, as in GCR.main()).
The script prints each case's baseline min/max and the shifted mean so you
can see where you are.

Correlated-error caveat
-----------------------
Because every rho is measured against the same baseline run, the points
share the baseline's statistical error and are therefore correlated. The
straight-line/quadratic fits below ignore that correlation (as the anchor
script did); treat the coefficient errors as indicative.

Usage
-----
    python fuel_temperature_sensitivity.py                # full sweep
    python fuel_temperature_sensitivity_delta.py --reprocess    # re-plot only
    python fuel_temperature_sensitivity_delta.py --rebuild-from-statepoints
"""

import os
import re
import csv
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import openmc

from GCR import GCR, GCRConfig


# ---------------------------------------------------------------------------
# Scan configuration
# ---------------------------------------------------------------------------

# Additive fuel-temperature shifts (K) applied uniformly to the native
# per-layer profile. One OpenMC run per entry. 0.0 (the unshifted model)
# is the reactivity reference and is inserted automatically if omitted.
FUEL_T_DELTAS_K = [-5000.0, -2000.0, -1000.0, 0.0, 1000.0, 2000.0, 5000.0]

# Couple fuel density to fuel temperature?
#   False -> rho_fuel held fixed -> isolates cross-section / Doppler component.
#   True  -> rho_fuel * T = const (ideal gas at constant pressure)
#            -> combined Doppler + density feedback (see module docstring).
COUPLE_DENSITY_TO_FUEL_T = False

# True -> collect IFP-based kinetic parameters (beta_eff, Lambda_eff) per case.
# Requires the cross-section library to carry delayed-neutron data.
COLLECT_KINETIC_PARAMS = True

# Mirror main() in GCR.py so geometry, materials and cross sections match
# the production model exactly.
BASE_CONFIG_KWARGS = dict(
    cross_sections_dir='libraries_xs/jeff40_hdf5',
    n_axial_layers=10,
    h2_density_profile_path='settings/h2_density_profile.npz',
)

# Critical-state fuel-density scaling, as in GCR.main().
FUEL_DENSITY_ALPHA = 2.0240

# Monte Carlo statistics applied to every case.
N_BATCHES   = 251
N_INACTIVE  = 25
N_PARTICLES = 500_000

# Output directories (suffixed _delta so this study does not overwrite the
# anchor-based fuel_temperature_sensitivity.py outputs).
RESULTS_DIR = 'sensitivity_results_fuel_T_delta'
RUNS_DIR    = 'sensitivity_runs_fuel_T_delta'


# ---------------------------------------------------------------------------
# Material classification (mirrors fuel_temperature_sensitivity.py)
# ---------------------------------------------------------------------------

def _is_fuel(mat):
    name = mat.name or ''
    return (name == 'fuel'
            or name.startswith('fuel_inner')
            or name.startswith('fuel_outer'))


def _unique_materials(core):
    """Yield each material object once; layered fuel is aliased under two keys."""
    seen = set()
    for mat in core.materials.values():
        if id(mat) in seen:
            continue
        seen.add(id(mat))
        yield mat


# ---------------------------------------------------------------------------
# Shifting (not setting) the fuel temperature
# ---------------------------------------------------------------------------

def shift_fuel_temperature(core, delta_K):
    """Add delta_K to every fuel material's *native* (per-layer) temperature.

    Unlike the anchor approach, this preserves the radiation-equilibrium
    per-layer temperature profile set by
    GCR._create_layered_fuel_materials and rigidly shifts the whole profile
    by a single common offset: each layer keeps its distinct baseline
    temperature, only delta_K is added.

    If COUPLE_DENSITY_TO_FUEL_T is True, each layer's density is rescaled by
    T_old / T_new (ideal gas, constant pressure) on top of the shift. See
    the module docstring for the real-gas-EOS caveat. Otherwise density is
    left untouched, which isolates the cross-section component.
    """
    delta = float(delta_K)
    for mat in _unique_materials(core):
        if not _is_fuel(mat):
            continue
        T_old = float(mat.temperature)
        T_new = T_old + delta
        if T_new <= 0.0:
            raise ValueError(
                f'Fuel material {mat.name!r}: T_old={T_old:.1f} K + '
                f'delta={delta:+.1f} K = {T_new:.1f} K is non-physical.'
            )
        mat.temperature = T_new
        if COUPLE_DENSITY_TO_FUEL_T:
            if mat.density_units != 'g/cm3':
                raise RuntimeError(
                    f'Fuel material {mat.name!r} has density_units='
                    f'{mat.density_units!r}; expected g/cm3.'
                )
            mat.set_density('g/cm3', mat.density * (T_old / T_new))


# ---------------------------------------------------------------------------
# Baseline (native) fuel-temperature statistics, captured pre-shift
# ---------------------------------------------------------------------------

def _fuel_layer_temperatures(core):
    """Native per-layer fuel temperatures actually used in the geometry
    (inner layers when layered, else the canonical inner material)."""
    layers = getattr(core, '_fuel_inner_layer_materials', None)
    if layers:
        return [float(m.temperature) for m in layers]
    return [float(core.materials['fuel_inner'].temperature)]


def _fuel_temperature_stats(core):
    T = np.asarray(_fuel_layer_temperatures(core), float)
    return {
        'T_base_mean': float(T.mean()),
        'T_base_min':  float(T.min()),
        'T_base_max':  float(T.max()),
        'n_layers':    int(T.size),
    }


# ---------------------------------------------------------------------------
# Build / run / extract  (identical helpers to the anchor script, retargeted
# at our own RUNS_DIR)
# ---------------------------------------------------------------------------

def build_core(config, run_subdir):
    core = GCR(config)
    core.output_dir = os.path.join(RUNS_DIR, run_subdir)
    os.makedirs(core.output_dir, exist_ok=True)
    core.set_materials()
    if config.n_axial_layers > 1:
        core._create_layered_propellant_materials()
        core._create_layered_fuel_materials()

    # Critical-state fuel density (matches GCR.main()).
    seen = set()
    for mat in core.materials.values():
        if id(mat) in seen:
            continue
        seen.add(id(mat))
        if _is_fuel(mat):
            mat.set_density('g/cm3', mat.density * FUEL_DENSITY_ALPHA)
    return core


def build_geometry_and_export(core):
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
    if COLLECT_KINETIC_PARAMS:
        core.add_kinetics_tally(num_groups=6)


def run_and_extract_keff(core):
    core.run(dry_run=False, map_geometry=False)
    sp_path = os.path.join(core.output_dir,
                           f'statepoint.{core.settings.batches}.h5')
    sp = openmc.StatePoint(sp_path)
    keff = sp.keff
    nominal = float(keff.nominal_value)
    stddev  = float(keff.std_dev)

    kinetics = {}
    if COLLECT_KINETIC_PARAMS:
        try:
            kin = sp.get_kinetics_parameters()
            beta = kin.beta_effective
            gen  = kin.generation_time
            kinetics['beta_eff']         = float(beta.nominal_value)
            kinetics['sigma_beta_eff']   = float(beta.std_dev)
            kinetics['gen_time_s']       = float(gen.nominal_value)
            kinetics['sigma_gen_time_s'] = float(gen.std_dev)
        except Exception as exc:
            print(f'    [warn] kinetic parameters not available: {exc}')

    sp.close()
    return nominal, stddev, kinetics


# ---------------------------------------------------------------------------
# One case = one OpenMC run at a single delta
# ---------------------------------------------------------------------------

def _tag(delta_K):
    sign = 'm' if delta_K < 0 else 'p'
    return f'fuelT_delta_{sign}{int(round(abs(delta_K))):05d}K'


def run_case(delta_K, base_config):
    tag = _tag(delta_K)
    print(f'\n=== {tag} ===')

    core = build_core(base_config, run_subdir=tag)
    stats = _fuel_temperature_stats(core)        # native profile, pre-shift
    shift_fuel_temperature(core, delta_K)
    build_geometry_and_export(core)

    T_eff_mean = stats['T_base_mean'] + float(delta_K)
    print(f'    baseline fuel T: mean {stats["T_base_mean"]:.0f} K '
          f'(min {stats["T_base_min"]:.0f}, max {stats["T_base_max"]:.0f}); '
          f'shift {delta_K:+.0f} K -> mean {T_eff_mean:.0f} K, '
          f'coolest layer -> {stats["T_base_min"] + float(delta_K):.0f} K')

    keff, sigma, kinetics = run_and_extract_keff(core)
    print(f'    --> k_eff = {keff:.5f} +/- {sigma:.5f}')
    if kinetics:
        print(f'    --> beta_eff = {kinetics["beta_eff"]:.5f} '
              f'+/- {kinetics["sigma_beta_eff"]:.5f}')
        print(f'    --> Lambda_eff = {kinetics["gen_time_s"]:.4e} '
              f'+/- {kinetics["sigma_gen_time_s"]:.4e} s')

    result = {
        'delta_K':     float(delta_K),
        'T_base_mean': stats['T_base_mean'],
        'T_base_min':  stats['T_base_min'],
        'T_base_max':  stats['T_base_max'],
        'n_layers':    stats['n_layers'],
        'T_eff_mean':  float(T_eff_mean),
        'keff':        float(keff),
        'sigma':       float(sigma),
        'tag':         tag,
        'coupled_fuel_rho': bool(COUPLE_DENSITY_TO_FUEL_T),
    }
    result.update(kinetics)
    return result


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def _ordered_unique_deltas(deltas):
    """Sorted, de-duplicated deltas with 0.0 guaranteed present."""
    vals = sorted({float(d) for d in deltas})
    if not any(abs(v) < 1e-9 for v in vals):
        vals = sorted(vals + [0.0])
    return vals


def run_delta_scan():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR,    exist_ok=True)

    base_config = GCRConfig(**BASE_CONFIG_KWARGS)
    deltas = _ordered_unique_deltas(FUEL_T_DELTAS_K)
    print(f'Fuel-temperature delta scan over {len(deltas)} cases: '
          + ', '.join(f'{d:+.0f}' for d in deltas) + ' K')

    results = [run_case(d, base_config) for d in deltas]

    out = os.path.join(RESULTS_DIR, 'keff_results.json')
    with open(out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nRaw k_eff results written to {out}')
    return results


# ---------------------------------------------------------------------------
# Post-processing: reactivity vs baseline + global/local coefficient
# ---------------------------------------------------------------------------

def _reactivity_pcm(k, sk, k_ref, sk_ref):
    """Reactivity rho = (k - k_ref) / (k * k_ref) in pcm, with sigma."""
    rho   = 1e5 * (k - k_ref) / (k * k_ref)
    sigma = 1e5 * np.hypot(sk / k**2, sk_ref / k_ref**2)
    return float(rho), float(sigma)


def compute_reactivity_and_alpha(results):
    """Attach rho (relative to the delta = 0 baseline) to every case."""
    cases = sorted(results, key=lambda r: r['delta_K'])
    base = next((r for r in cases if abs(r['delta_K']) < 1e-9), None)
    if base is None:
        raise RuntimeError(
            'No delta = 0 baseline case found; reactivity is defined '
            'relative to the unshifted profile. Add 0.0 to FUEL_T_DELTAS_K.'
        )

    k_ref, s_ref = base['keff'], base['sigma']
    for r in cases:
        r['rho_pcm'], r['sigma_rho_pcm'] = _reactivity_pcm(
            r['keff'], r['sigma'], k_ref, s_ref
        )

    summary = {
        'cases':        cases,
        'k_ref':        float(k_ref),
        'sigma_k_ref':  float(s_ref),
        'T_base_mean':  float(base.get('T_base_mean', float('nan'))),
        'T_base_min':   float(base.get('T_base_min',  float('nan'))),
        'T_base_max':   float(base.get('T_base_max',  float('nan'))),
        'coupled_fuel_rho': bool(base.get('coupled_fuel_rho',
                                          COUPLE_DENSITY_TO_FUEL_T)),
    }
    return summary


def _polyfit_with_errors(x, y, sy, degree):
    """Weighted polynomial fit. Returns coeffs and 1-sigma errors low->high order."""
    x  = np.asarray(x,  float)
    y  = np.asarray(y,  float)
    sy = np.asarray(sy, float)
    w  = 1.0 / sy**2
    X  = np.vander(x, degree + 1, increasing=True)
    cov  = np.linalg.inv(X.T @ np.diag(w) @ X)
    beta = cov @ X.T @ (w * y)
    err  = np.sqrt(np.diag(cov))
    return beta, err, cov


def _local_central_alpha(cases):
    """Central-difference slope at the smallest symmetric +-delta present."""
    by_delta = {round(c['delta_K'], 6): c for c in cases}
    mags = sorted({abs(d) for d in by_delta if abs(d) > 1e-9})
    for m in mags:
        plus  = by_delta.get(round(+m, 6))
        minus = by_delta.get(round(-m, 6))
        if plus and minus:
            two_h = plus['delta_K'] - minus['delta_K']
            alpha = (plus['rho_pcm'] - minus['rho_pcm']) / two_h
            err = np.hypot(plus['sigma_rho_pcm'],
                           minus['sigma_rho_pcm']) / two_h
            return {
                'delta_mag_K':          float(m),
                'alpha_pcm_per_K':      float(alpha),
                'alpha_pcm_per_K_err':  float(err),
            }
    return None


def fit_rho_vs_delta(summary):
    """Fit linear and quadratic models to rho(Delta T).

    The linear slope a1 is the global uniform-shift fuel-temperature
    coefficient alpha [pcm/K]. The quadratic a2 captures how alpha varies
    with the magnitude/sign of the shift.
    """
    cases = summary['cases']
    d  = np.array([c['delta_K']       for c in cases], float)
    y  = np.array([c['rho_pcm']       for c in cases], float)
    sy = np.array([c['sigma_rho_pcm'] for c in cases], float)

    # Floor any non-positive sigma (e.g. a degenerate baseline) so weights
    # stay finite.
    pos = sy[sy > 0]
    floor = float(pos.max()) if pos.size else 1.0
    sy = np.where(sy > 0, sy, floor)

    out = {
        'delta':         d.tolist(),
        'rho_pcm':       y.tolist(),
        'sigma_rho_pcm': sy.tolist(),
    }

    if d.size >= 2:
        c1, e1, _ = _polyfit_with_errors(d, y, sy, degree=1)
        out['linear'] = {
            'a0': float(c1[0]), 'a1': float(c1[1]),
            'a0_err': float(e1[0]), 'a1_err': float(e1[1]),
            'alpha_pcm_per_K':     float(c1[1]),
            'alpha_pcm_per_K_err': float(e1[1]),
        }
    if d.size >= 3:
        c2, e2, _ = _polyfit_with_errors(d, y, sy, degree=2)
        out['quadratic'] = {
            'a0': float(c2[0]), 'a1': float(c2[1]), 'a2': float(c2[2]),
            'a0_err': float(e2[0]), 'a1_err': float(e2[1]),
            'a2_err': float(e2[2]),
        }

    out['alpha_local'] = _local_central_alpha(cases)
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _subtitle(coupled):
    return (r'  ($\rho\,T=$ const)' if coupled
            else r'  (Doppler only, $\rho_\mathrm{fuel}$ fixed)')


def plot_sensitivity(summary, fit, save=True):
    cases   = summary['cases']
    coupled = summary['coupled_fuel_rho']
    subtitle = _subtitle(coupled)

    d    = np.array([c['delta_K'] for c in cases], float)
    rho  = np.array([c['rho_pcm'] for c in cases], float)
    srho = np.array([c['sigma_rho_pcm'] for c in cases], float)
    keff = np.array([c['keff']  for c in cases], float)
    skeff = np.array([c['sigma'] for c in cases], float)
    T_mean = summary['T_base_mean']

    # ---------------- Figure 1: reactivity vs delta (with fit) ----------- #
    fig1, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.axhline(0.0, color='k', lw=0.8, ls=':')
    ax1.axvline(0.0, color='k', lw=0.8, ls=':')
    ax1.errorbar(d, rho, yerr=srho, marker='o', ls='none', capsize=3,
                 color='C0', label='OpenMC', zorder=3)

    dd = np.linspace(d.min(), d.max(), 200)
    if 'linear' in fit:
        L = fit['linear']
        ax1.plot(dd, L['a0'] + L['a1'] * dd, '-', color='C3',
                 label=rf"linear: $\alpha={L['a1']:+.2f}\pm{L['a1_err']:.2f}$ pcm/K")
    if 'quadratic' in fit:
        Q = fit['quadratic']
        ax1.plot(dd, Q['a0'] + Q['a1'] * dd + Q['a2'] * dd**2, '--',
                 color='C2', label='quadratic')

    ax1.set_xlabel(r'Uniform fuel-temperature shift $\Delta T$ (K)')
    ax1.set_ylabel(r'Reactivity vs baseline $\rho$ (pcm)')
    ax1.set_title('GCR reactivity vs uniform fuel-T shift' + subtitle)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc='best')

    if np.isfinite(T_mean):
        ax1t = ax1.secondary_xaxis(
            'top',
            functions=(lambda x: x + T_mean, lambda x: x - T_mean),
        )
        ax1t.set_xlabel(r'Mean fuel temperature $\bar{T}_\mathrm{fuel}$ (K)')

    fig1.tight_layout()
    if save:
        pdf1 = os.path.join(RESULTS_DIR, 'fuel_T_delta_reactivity.pdf')
        fig1.savefig(pdf1, dpi=150)
        print(f'Reactivity plot written to {pdf1}')

    # ---------------- Figure 2: k_eff vs delta --------------------------- #
    fig2, ax2 = plt.subplots(figsize=(7.5, 5))
    ax2.errorbar(d, keff, yerr=skeff, marker='s', ls='-', capsize=3, color='C0')
    ax2.axhline(1.0, color='k', lw=0.8, ls='--', label=r'$k_\mathrm{eff}=1$')
    ax2.axvline(0.0, color='k', lw=0.8, ls=':')
    ax2.set_xlabel(r'Uniform fuel-temperature shift $\Delta T$ (K)')
    ax2.set_ylabel(r'$k_\mathrm{eff}$')
    ax2.set_title(r'GCR $k_\mathrm{eff}$ vs uniform fuel-T shift' + subtitle)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    fig2.tight_layout()
    if save:
        pdf2 = os.path.join(RESULTS_DIR, 'fuel_T_delta_keff.pdf')
        fig2.savefig(pdf2, dpi=150)
        print(f'k_eff plot written to {pdf2}')

    plt.show()
    return fig1, fig2


def plot_kinetics(summary, save=True):
    """Plot beta_eff and Lambda_eff vs delta."""
    cases = [c for c in summary['cases'] if 'beta_eff' in c]
    if not cases:
        print('No kinetic parameters found in summary; skipping kinetics plot.')
        return None

    subtitle = _subtitle(summary['coupled_fuel_rho'])

    d  = np.array([c['delta_K'] for c in cases], float)
    b  = np.array([c['beta_eff'] for c in cases], float)
    sb = np.array([c.get('sigma_beta_eff', 0.0) for c in cases], float)
    g  = np.array([c['gen_time_s'] for c in cases], float)
    sg = np.array([c.get('sigma_gen_time_s', 0.0) for c in cases], float)

    fig, (ax_b, ax_g) = plt.subplots(2, 1, figsize=(7.5, 9), sharex=True)

    ax_b.errorbar(d, b * 1e3, yerr=sb * 1e3, marker='o', ls='-',
                  capsize=3, color='C0')
    ax_b.axvline(0.0, color='k', lw=0.8, ls=':')
    ax_b.set_ylabel(r'$\beta_\mathrm{eff}$ ($\times 10^{-3}$)')
    ax_b.set_title(r'Effective delayed-neutron fraction vs fuel-T shift'
                   + subtitle)
    ax_b.grid(alpha=0.3)

    ax_g.errorbar(d, g * 1e6, yerr=sg * 1e6, marker='s', ls='-',
                  capsize=3, color='C0')
    ax_g.axvline(0.0, color='k', lw=0.8, ls=':')
    ax_g.set_xlabel(r'Uniform fuel-temperature shift $\Delta T$ (K)')
    ax_g.set_ylabel(r'$\Lambda_\mathrm{eff}$ (µs)')
    ax_g.set_title(r'Prompt neutron generation time vs fuel-T shift' + subtitle)
    ax_g.grid(alpha=0.3)

    fig.tight_layout()
    if save:
        pdf = os.path.join(RESULTS_DIR, 'fuel_T_delta_kinetics.pdf')
        fig.savefig(pdf, dpi=150)
        print(f'Kinetic parameters plot written to {pdf}')
    return fig


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def write_csv(summary, fit):
    cases = summary['cases']
    csv_path = os.path.join(RESULTS_DIR, 'fuel_T_delta_sensitivity.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([
            'delta_K', 'T_base_mean_K', 'T_eff_mean_K',
            'keff', 'sigma_keff',
            'rho_pcm_vs_baseline', 'sigma_rho_pcm',
            'beta_eff', 'sigma_beta_eff',
            'gen_time_s', 'sigma_gen_time_s',
        ])
        for c in cases:
            w.writerow([
                c['delta_K'], c.get('T_base_mean', ''), c.get('T_eff_mean', ''),
                c['keff'], c['sigma'],
                c.get('rho_pcm', 0.0), c.get('sigma_rho_pcm', 0.0),
                c.get('beta_eff', ''), c.get('sigma_beta_eff', ''),
                c.get('gen_time_s', ''), c.get('sigma_gen_time_s', ''),
            ])
    print(f'Per-case table written to {csv_path}')

    fit_path = os.path.join(RESULTS_DIR, 'alpha_fit.json')
    with open(fit_path, 'w') as fh:
        json.dump(fit, fh, indent=2)
    print(f'Fit coefficients written to {fit_path}')


def print_summary(summary, fit):
    tb = summary['T_base_mean']
    tb_str = (f'{tb:.0f} K (min {summary["T_base_min"]:.0f}, '
              f'max {summary["T_base_max"]:.0f})'
              if np.isfinite(tb) else 'n/a (rebuilt from statepoints)')
    print('\nFuel-temperature (uniform-shift) sensitivity:')
    print(f'  baseline mean fuel T = {tb_str}')
    print(f'  k_ref (delta=0) = {summary["k_ref"]:.5f} '
          f'+/- {summary["sigma_k_ref"]:.5f}')

    print('\n   dT (K)     k_eff                rho vs baseline (pcm)')
    print('   ------     -----------------    ---------------------')
    for c in summary['cases']:
        print(f'  {c["delta_K"]:+7.0f}    {c["keff"]:.5f} +/- {c["sigma"]:.5f}    '
              f'{c.get("rho_pcm", 0.0):+9.1f} +/- '
              f'{c.get("sigma_rho_pcm", 0.0):5.1f}')

    if 'linear' in fit:
        L = fit['linear']
        print(f'\n  Global linear coefficient  alpha = d(rho)/dT = '
              f'{L["a1"]:+.3f} +/- {L["a1_err"]:.3f} pcm/K')
        print(f'    (fit intercept {L["a0"]:+.2f} +/- {L["a0_err"]:.2f} pcm)')
    if fit.get('alpha_local'):
        A = fit['alpha_local']
        print(f'  Local coefficient at +/-{A["delta_mag_K"]:.0f} K = '
              f'{A["alpha_pcm_per_K"]:+.3f} +/- '
              f'{A["alpha_pcm_per_K_err"]:.3f} pcm/K')
    if 'quadratic' in fit:
        Q = fit['quadratic']
        print(f'  Quadratic curvature  a2 = {Q["a2"]:+.4g} +/- {Q["a2_err"]:.2g} '
              f'pcm/K^2   (rho = a0 + a1*dT + a2*dT^2)')


# ---------------------------------------------------------------------------
# Re-run post-processing without redoing OpenMC
# ---------------------------------------------------------------------------

def load_and_postprocess(results_path=None):
    if results_path is None:
        results_path = os.path.join(RESULTS_DIR, 'keff_results.json')
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f'No cached results at {results_path!r}. '
            f'Run run_delta_scan() first.'
        )
    with open(results_path) as fh:
        results = json.load(fh)
    summary = compute_reactivity_and_alpha(results)
    fit     = fit_rho_vs_delta(summary)
    write_csv(summary, fit)
    print_summary(summary, fit)
    plot_sensitivity(summary, fit)
    plot_kinetics(summary)
    return summary, fit


def rebuild_results_from_statepoints(runs_dir=None, save=True):
    """Walk RUNS_DIR, parse delta tags, read every statepoint and rebuild
    keff_results.json from what's actually on disk.

    Tags must follow the _tag() convention (fuelT_delta_{p|m}NNNNNK) so the
    delta can be recovered. Baseline temperature statistics cannot be
    recovered from a statepoint and are set to NaN (the absolute-temperature
    top axis on the reactivity plot is then omitted).
    """
    if runs_dir is None:
        runs_dir = RUNS_DIR
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f'No runs directory at {runs_dir!r}.')

    tag_re = re.compile(r'^fuelT_delta_([pm])(\d{5})K$')
    results = []

    for tag in sorted(os.listdir(runs_dir)):
        m = tag_re.match(tag)
        if not m:
            continue
        sign = -1.0 if m.group(1) == 'm' else 1.0
        delta = sign * float(m.group(2))
        case_dir = os.path.join(runs_dir, tag)

        sp_files = glob.glob(os.path.join(case_dir, 'statepoint.*.h5'))
        if not sp_files:
            print(f'[skip] {tag}: no statepoint found')
            continue
        sp_path = max(sp_files, key=os.path.getmtime)  # most recent

        try:
            sp = openmc.StatePoint(sp_path)
            keff = sp.keff
            nominal = float(keff.nominal_value)
            stddev  = float(keff.std_dev)
            kin = {}
            if COLLECT_KINETIC_PARAMS:
                try:
                    kp = sp.get_kinetics_parameters()
                    kin['beta_eff']         = float(kp.beta_effective.nominal_value)
                    kin['sigma_beta_eff']   = float(kp.beta_effective.std_dev)
                    kin['gen_time_s']       = float(kp.generation_time.nominal_value)
                    kin['sigma_gen_time_s'] = float(kp.generation_time.std_dev)
                except Exception:
                    pass
            sp.close()
        except Exception as e:
            print(f'[skip] {tag}: {e}')
            continue

        rec = {
            'delta_K':     delta,
            'T_base_mean': float('nan'),
            'T_base_min':  float('nan'),
            'T_base_max':  float('nan'),
            'T_eff_mean':  float('nan'),
            'keff':        nominal,
            'sigma':       stddev,
            'tag':         tag,
            'coupled_fuel_rho': bool(COUPLE_DENSITY_TO_FUEL_T),
        }
        rec.update(kin)
        results.append(rec)
        print(f'  {tag}: delta={delta:+.0f} K  keff={nominal:.5f} +/- {stddev:.5f}')

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, 'keff_results.json')
        with open(out, 'w') as fh:
            json.dump(results, fh, indent=2)
        print(f'\nMerged {len(results)} runs to {out}')

    return results


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Delta-based (rigid-shift) fuel-temperature sensitivity '
                    'analysis for the GCR model.',
    )
    parser.add_argument(
        '--reprocess', action='store_true',
        help='Skip OpenMC and re-do post-processing on cached '
             'keff_results.json only.',
    )
    parser.add_argument(
        '--rebuild-from-statepoints', action='store_true',
        help='Walk RUNS_DIR, read every statepoint on disk and rebuild '
             'keff_results.json from the delta tags.',
    )
    args = parser.parse_args()

    if args.rebuild_from_statepoints:
        rebuild_results_from_statepoints()
        load_and_postprocess()
    elif args.reprocess:
        load_and_postprocess()
    else:
        results = run_delta_scan()
        summary = compute_reactivity_and_alpha(results)
        fit     = fit_rho_vs_delta(summary)
        write_csv(summary, fit)
        print_summary(summary, fit)
        plot_sensitivity(summary, fit)
        plot_kinetics(summary)
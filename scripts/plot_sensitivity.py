"""
plot_sensitivity.py
===================

Produce sensitivity figures from a saved keff_results JSON file **without**
running any OpenMC simulations.  No GCR or OpenMC imports are required.

Usage
-----
# Plot all variables present in the JSON:
python plot_sensitivity.py

# Plot a single mode (only that variable's panels are shown):
python plot_sensitivity.py --mode fuel_T
python plot_sensitivity.py --mode h2_rho
python plot_sensitivity.py --mode power
python plot_sensitivity.py --mode fuel_rho
python plot_sensitivity.py --mode beo_T
python plot_sensitivity.py --mode h2_T

# Combine several modes into one figure:
python plot_sensitivity.py --mode fuel_T h2_rho beo_T

# Non-default JSON or output directory:
python plot_sensitivity.py --results path/to/keff_results.json --out-dir my_figures

# Suppress the interactive plt.show() window (useful on headless servers):
python plot_sensitivity.py --no-show

# Override coupling-assumption labels (purely cosmetic, affects figure title):
python plot_sensitivity.py --couple-fuel --couple-h2

Input JSON format
-----------------
Each entry in the JSON list must contain at least:
    variable  : str    – 'baseline' | 'fuel_T' | 'h2_rho' | 'power'
                                    | 'fuel_rho' | 'beo_T' | 'h2_T'
    delta     : float  – perturbation value (K or fractional, as produced by
                         sensitivity_analysis.py)
    keff      : float
    sigma     : float  – 1-sigma Monte Carlo uncertainty on keff

Optional kinetics fields (produced when COLLECT_KINETIC_PARAMS = True):
    beta_eff            : float
    sigma_beta_eff      : float
    gen_time_s          : float
    sigma_gen_time_s    : float

The baseline entry (variable='baseline', delta=0) must always be present.

Outputs
-------
Three PDF files are written to --out-dir (default: sensitivity_results/):
    reactivity[_<modes>].pdf   – ρ = (k−1)/k  (pcm) vs perturbation
    keff[_<modes>].pdf         – k_eff vs perturbation
    kinetics[_<modes>].pdf     – β_eff and Λ_eff (only if kinetics data present)
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Weighted linear fit
# ---------------------------------------------------------------------------

def _weighted_linear_fit(x, y, sy):
    """Weighted least-squares y = slope*x + intercept (weights = 1/sy²).

    Returns (slope, intercept, slope_err, intercept_err).
    Raises ValueError if the system is under-determined or any sy is zero.
    """
    x  = np.asarray(x,  float)
    y  = np.asarray(y,  float)
    sy = np.asarray(sy, float)
    if np.any(sy <= 0):
        raise ValueError('All sigma values must be strictly positive for a '
                         'weighted fit.')
    w   = 1.0 / sy**2
    X   = np.vander(x, 2)                    # columns: [x, 1]
    cov = np.linalg.inv(X.T @ np.diag(w) @ X)
    beta = cov @ X.T @ (w * y)
    slope, intercept = float(beta[0]), float(beta[1])
    slope_err        = float(np.sqrt(cov[0, 0]))
    intercept_err    = float(np.sqrt(cov[1, 1]))
    return slope, intercept, slope_err, intercept_err


# ---------------------------------------------------------------------------
# Figure-layout helper (shared by all three figures)
# ---------------------------------------------------------------------------

def _make_fig(n_panels, rows_per_panel=1):
    """Return (fig, axes_2d, axes_flat) for n_panels panels.

    rows_per_panel=1  -> standard single-row-of-axes layout
    rows_per_panel=2  -> used by the kinetics figure (β + Λ rows)
    """
    cols = min(n_panels, 3)
    if n_panels <= 3:
        grid_rows = rows_per_panel
        grid_cols = n_panels
    elif n_panels == 4:
        grid_rows = 2 * rows_per_panel
        grid_cols = 2
    else:
        grid_rows = 2 * rows_per_panel
        grid_cols = 3

    w = 5.5 * grid_cols
    h = 4.5 * grid_rows
    fig, axes = plt.subplots(grid_rows, grid_cols,
                             figsize=(w, h), squeeze=False)

    # Hide surplus cells (can happen when n_panels == 5)
    flat = list(axes.flatten())
    surplus_start = rows_per_panel * n_panels
    for ax in flat[surplus_start:]:
        ax.axis('off')

    return fig, axes, flat


# ---------------------------------------------------------------------------
# Panel metadata
# ---------------------------------------------------------------------------

_PANEL_META = {
    # key: (x_scale, xlabel, marker, colour, title)
    'fuel_T':   (1.0,   r'$\Delta T_{\mathrm{fuel}}$ (K)',      'o', 'C0',
                 'Fuel temperature'),
    'h2_rho':   (100.0, r'$\Delta \rho_{\mathrm{H}_2}$ (%)',    's', 'C1',
                 r'H$_2$ propellant density'),
    'power':    (100.0, r'$\Delta Q_{\mathrm{cavity}}$ (%)',     '^', 'C2',
                 'Cavity power'),
    'fuel_rho': (100.0, r'$\Delta \rho_{\mathrm{fuel}}$ (%)',   'D', 'C3',
                 'Fuel density (T fixed)'),
    'beo_T':    (1.0,   r'$\Delta T_{\mathrm{BeO}}$ (K)',        'v', 'C4',
                 'BeO reflector temperature'),
    'h2_T':     (1.0,   r'$\Delta T_{\mathrm{H}_2}$ (K)',        'P', 'C5',
                 r'H$_2$ propellant temperature'),
    'fuel_radius': (100.0, r'$\delta R / R_0$ (%)',               'h', 'C6',
                 r'Fuel-cloud radius ($N_U$ const)'),
}


# ---------------------------------------------------------------------------
# Alpha labels for the reactivity fit annotation
# ---------------------------------------------------------------------------

_ALPHA_LABEL = {
    'fuel_T':   (r'$\alpha_T$',                  'pcm/K'),
    'h2_rho':   (r'$\alpha_\rho$',               'pcm/%'),
    'power':    (r'$\alpha_Q$',                  'pcm/%'),
    'fuel_rho': (r'$\alpha_{\rho,F}$',           'pcm/%'),
    'beo_T':    (r'$\alpha_\mathrm{BeO}$',       'pcm/K'),
    'h2_T':     (r'$\alpha_{T,\mathrm{H}_2}$',  'pcm/K'),
    'fuel_radius': (r'$\alpha_R$',              'pcm/%'),
}


# ---------------------------------------------------------------------------
# Core plotting routine
# ---------------------------------------------------------------------------

def make_figures(results, panels,
                 out_dir='sensitivity_results',
                 suptitle_base='GCR reactivity feedback',
                 show=True):
    """Produce all three sensitivity figures from *results* for the given *panels*.

    Parameters
    ----------
    results : list[dict]
        Loaded JSON list.  Must contain a 'baseline' entry.
    panels : list[str]
        Ordered list of variable names to plot, e.g. ['fuel_T', 'h2_rho'].
        Only names present in _PANEL_META are accepted.
    out_dir : str
        Directory for the output PDFs (created if absent).
    suptitle_base : str
        Root figure super-title (coupling notes appended automatically if
        coupling fields are detected in the results).
    show : bool
        Whether to call plt.show() at the end.

    Returns
    -------
    dict
        Mapping panel name -> (slope, slope_err, units) for each fitted panel.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Baseline
    # ------------------------------------------------------------------ #
    baseline = next((r for r in results if r['variable'] == 'baseline'), None)
    if baseline is None:
        raise ValueError("No 'baseline' entry in results.  "
                         "A baseline case (variable='baseline', delta=0) is "
                         "required to compute reactivity.")

    # ------------------------------------------------------------------ #
    # Compute ρ = (k-1)/k  (pcm) for every case
    # ------------------------------------------------------------------ #
    for r in results:
        k, s = r['keff'], r['sigma']
        r['rho_pcm']       = 1e5 * (k - 1.0) / k
        r['sigma_rho_pcm'] = 1e5 * s / k**2

    rho_baseline = baseline['rho_pcm']
    k_baseline   = baseline['keff']
    sk_baseline  = baseline['sigma']

    # ------------------------------------------------------------------ #
    # Subset each panel's data
    # ------------------------------------------------------------------ #
    subs = {p: sorted([r for r in results if r['variable'] == p],
                      key=lambda r: r['delta'])
            for p in panels}

    # ------------------------------------------------------------------ #
    # Weighted linear fits
    # ------------------------------------------------------------------ #
    fits        = {}   # panel -> (slope_in_natural_units, intercept, err, err_int)
    coefficients = {}  # panel -> (slope_display, err_display, units_str)

    for panel in panels:
        sub = subs[panel]
        if len(sub) < 2:
            print(f'  [{panel}] fewer than 2 points — skipping fit.')
            continue
        scale, _, _, _, _ = _PANEL_META[panel]
        x  = [r['delta']         for r in sub]
        y  = [r['rho_pcm']       for r in sub]
        sy = [r['sigma_rho_pcm'] for r in sub]
        try:
            s, i, se, ie = _weighted_linear_fit(x, y, sy)
        except (np.linalg.LinAlgError, ValueError) as exc:
            print(f'  [{panel}] fit failed: {exc}')
            continue
        fits[panel] = (s, i, se, ie)
        # Convert to display units (pcm/% for fractional variables)
        disp_slope = s / scale if scale != 1.0 else s
        disp_err   = se / scale if scale != 1.0 else se
        _, units   = _ALPHA_LABEL[panel]
        coefficients[panel] = (disp_slope, disp_err, units)
        sym, _ = _ALPHA_LABEL[panel]
        print(f'  {sym:40s} = {disp_slope:+8.3f} ± {disp_err:.3f}  {units}')

    # ------------------------------------------------------------------ #
    # Coupling-assumption notes (read from the data if present)
    # ------------------------------------------------------------------ #
    couple_fuel = any(r.get('coupled_fuel_rho') for r in results)
    couple_h2   = any(r.get('coupled_h2_rho')   for r in results)
    notes = []
    if subs.get('fuel_T'):
        if couple_fuel:
            notes.append(r'fuel $T$ couples $\rho_\mathrm{fuel}$ via '
                         r'$\rho T = \mathrm{const}$')
        else:
            notes.append(r'pure Doppler: $\rho_\mathrm{fuel}$ held constant')
    if subs.get('h2_T'):
        if couple_h2:
            notes.append(r'H$_2$ $T$ couples $\rho_{\mathrm{H}_2}$ via '
                         r'$\rho T = \mathrm{const}$')
        else:
            notes.append(r'H$_2$ $T$: $\rho_{\mathrm{H}_2}$ held constant')
    suptitle = suptitle_base
    if notes:
        suptitle += '  (' + '; '.join(notes) + ')'

    # Filename suffix (omit when all panels are plotted)
    plot_suffix = '_' + '_'.join(panels) if len(panels) < 7 else ''
    n_panels = len(panels)

    # ================================================================== #
    # Figure 1 — reactivity ρ (pcm) vs perturbation
    # ================================================================== #
    fig1, axes1, flat1 = _make_fig(n_panels, rows_per_panel=1)

    for ax, panel in zip(flat1, panels):
        scale, xlabel, mk, col, title = _PANEL_META[panel]
        sub = subs[panel]
        x   = np.array([scale * r['delta']         for r in sub])
        y   = np.array([r['rho_pcm']               for r in sub])
        ye  = np.array([r['sigma_rho_pcm']         for r in sub])

        ax.errorbar(x, y, yerr=ye, marker=mk, color=col,
                    capsize=3, linestyle='none', label='data')

        if panel in fits:
            s, i, se, _ = fits[panel]
            xf = np.linspace(x.min(), x.max(), 50)
            sym, units = _ALPHA_LABEL[panel]
            disp_s  = s / scale  if scale != 1.0 else s
            disp_se = se / scale if scale != 1.0 else se
            ax.plot(xf, s * xf + i, 'k--', lw=1,
                    label=rf'{sym} $= {disp_s:+.2f} \pm {disp_se:.2f}$ {units}')

        ax.axhline(rho_baseline, color='grey', lw=0.8, linestyle=':',
                   label=rf'$\rho_{{\mathrm{{ref}}}} = {rho_baseline:.0f}$ pcm')
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ylabel_rho = r'Reactivity $\rho = (k{-}1)/k$ (pcm)'
    if n_panels <= 3:
        flat1[0].set_ylabel(ylabel_rho)
    else:
        axes1[0, 0].set_ylabel(ylabel_rho)
        axes1[1, 0].set_ylabel(ylabel_rho)

    fig1.suptitle(suptitle)
    fig1.tight_layout()
    pdf1 = os.path.join(out_dir, f'reactivity{plot_suffix}.pdf')
    fig1.savefig(pdf1, dpi=150)
    print(f'Reactivity plot  -> {pdf1}')

    # ================================================================== #
    # Figure 2 — k_eff vs perturbation
    # ================================================================== #
    fig2, axes2, flat2 = _make_fig(n_panels, rows_per_panel=1)

    for ax, panel in zip(flat2, panels):
        scale, xlabel, mk, col, title = _PANEL_META[panel]
        sub = subs[panel]
        x   = np.array([scale * r['delta'] for r in sub])
        y   = np.array([r['keff']          for r in sub])
        ye  = np.array([r['sigma']         for r in sub])

        ax.errorbar(x, y, yerr=ye, marker=mk, color=col,
                    capsize=3, linestyle='none', label='data')
        ax.errorbar([0], [k_baseline], yerr=[sk_baseline],
                    marker='*', color='grey', capsize=3, markersize=9,
                    linestyle='none',
                    label=rf'baseline $k = {k_baseline:.5f}$')
        ax.axhline(1.0, color='k', lw=0.8, linestyle='--',
                   label=r'$k_\mathrm{eff} = 1$')
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ylabel_k = r'$k_\mathrm{eff}$'
    if n_panels <= 3:
        flat2[0].set_ylabel(ylabel_k)
    else:
        axes2[0, 0].set_ylabel(ylabel_k)
        axes2[1, 0].set_ylabel(ylabel_k)

    fig2.suptitle(suptitle.replace('reactivity feedback', r'$k_\mathrm{eff}$'))
    fig2.tight_layout()
    pdf2 = os.path.join(out_dir, f'keff{plot_suffix}.pdf')
    fig2.savefig(pdf2, dpi=150)
    print(f'k_eff plot       -> {pdf2}')

    # ================================================================== #
    # Figure 3 — kinetic parameters (β_eff, Λ_eff), if collected
    # ================================================================== #
    panels_with_kin = [p for p in panels
                       if any('beta_eff' in r
                               for r in subs[p])]
    beta_ref = baseline.get('beta_eff')
    gen_ref  = baseline.get('gen_time_s')
    has_kinetics = bool(panels_with_kin) or (beta_ref is not None)

    if has_kinetics:
        nk = len(panels_with_kin)
        if nk == 0:
            print('Kinetics data present in baseline only — skipping kinetics figure.')
        else:
            # Two rows per panel: top = β_eff, bottom = Λ_eff
            fig3, axes3, _ = _make_fig(nk, rows_per_panel=2)

            if nk <= 3:
                beta_axes = [axes3[0, i] for i in range(nk)]
                gen_axes  = [axes3[1, i] for i in range(nk)]
            elif nk == 4:
                beta_axes = [axes3[0, 0], axes3[0, 1],
                             axes3[2, 0], axes3[2, 1]]
                gen_axes  = [axes3[1, 0], axes3[1, 1],
                             axes3[3, 0], axes3[3, 1]]
            else:
                # 5 or 6
                beta_axes = [axes3[0, c] for c in range(3)] + \
                            [axes3[2, c] for c in range(nk - 3)]
                gen_axes  = [axes3[1, c] for c in range(3)] + \
                            [axes3[3, c] for c in range(nk - 3)]

            for i, panel in enumerate(panels_with_kin):
                scale, xlabel, mk, col, title = _PANEL_META[panel]
                sub = subs[panel]
                x   = np.array([scale * r['delta']               for r in sub])
                b   = np.array([r.get('beta_eff', np.nan)         for r in sub])
                sb  = np.array([r.get('sigma_beta_eff', 0.0)      for r in sub])
                g   = np.array([r.get('gen_time_s', np.nan)       for r in sub])
                sg  = np.array([r.get('sigma_gen_time_s', 0.0)    for r in sub])

                # β_eff panel
                ax_b = beta_axes[i]
                ax_b.errorbar(x, b * 1e3, yerr=sb * 1e3,
                              marker=mk, color=col, capsize=3,
                              linestyle='none', label='data')
                if beta_ref is not None:
                    ax_b.axhline(beta_ref * 1e3, color='grey', lw=0.8,
                                 linestyle=':',
                                 label=rf'baseline $\beta = {beta_ref*1e3:.2f}'
                                       rf'\times10^{{-3}}$')
                ax_b.set_xlabel(xlabel)
                ax_b.set_ylabel(r'$\beta_\mathrm{eff}$ ($\times10^{-3}$)')
                ax_b.set_title(f'β_eff — {title}')
                ax_b.grid(alpha=0.3)
                ax_b.legend(fontsize=8)

                # Λ_eff panel
                ax_g = gen_axes[i]
                ax_g.errorbar(x, g * 1e6, yerr=sg * 1e6,
                              marker=mk, color=col, capsize=3,
                              linestyle='none', label='data')
                if gen_ref is not None:
                    ax_g.axhline(gen_ref * 1e6, color='grey', lw=0.8,
                                 linestyle=':',
                                 label=rf'baseline $\Lambda = '
                                       rf'{gen_ref*1e6:.2f}$ µs')
                ax_g.set_xlabel(xlabel)
                ax_g.set_ylabel(r'$\Lambda_\mathrm{eff}$ (µs)')
                ax_g.set_title(rf'$\Lambda_\mathrm{{eff}}$ — {title}')
                ax_g.grid(alpha=0.3)
                ax_g.legend(fontsize=8)

            fig3.suptitle(suptitle.replace('reactivity feedback',
                                           'kinetic parameters'))
            fig3.tight_layout()
            pdf3 = os.path.join(out_dir, f'kinetics{plot_suffix}.pdf')
            fig3.savefig(pdf3, dpi=150)
            print(f'Kinetics plot    -> {pdf3}')
    else:
        print('No kinetic parameters in results — kinetics figure skipped.')

    if show:
        plt.show()

    return coefficients


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_MODES = ['fuel_T', 'h2_rho', 'power', 'fuel_rho', 'beo_T', 'h2_T', 'fuel_radius']

# Map the short mode names used by sensitivity_analysis.py's --mode flag
# to the variable keys stored in the JSON.
_MODE_ALIAS = {
    'fuel':         'fuel_T',
    'h2':           'h2_rho',
    'fuel_rho':     'fuel_rho',
    'power':        'power',
    'beo':          'beo_T',
    'h2_T':         'h2_T',
    'fuel_radius':  'fuel_radius',
    # canonical names pass through unchanged
    'fuel_T':       'fuel_T',
    'h2_rho':       'h2_rho',
    'beo_T':        'beo_T',
}


def main():
    parser = argparse.ArgumentParser(
        description='Plot GCR sensitivity figures from a saved JSON without '
                    'running any OpenMC simulations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--results', default=None,
        help='Path to keff_results JSON.  '
             'Defaults to sensitivity_results/keff_results.json, or '
             'sensitivity_results/keff_results_<mode>.json when a single '
             'mode is given.',
    )
    parser.add_argument(
        '--mode', nargs='+',
        metavar='MODE',
        default=None,
        help='Variable(s) to plot.  Any of: '
             + ', '.join(ALL_MODES)
             + ' (or the short aliases fuel, h2, beo).  '
               'Defaults to all variables present in the JSON.',
    )
    parser.add_argument(
        '--out-dir', default='sensitivity_results',
        help='Directory for output PDFs (default: sensitivity_results).',
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Do not call plt.show() — useful on headless servers.',
    )
    parser.add_argument(
        '--couple-fuel', action='store_true',
        help='Override coupling label: treat fuel T as coupled to density '
             '(ρT = const).  Purely cosmetic — affects the figure title.',
    )
    parser.add_argument(
        '--couple-h2', action='store_true',
        help='Override coupling label: treat H2 T as coupled to density.  '
             'Purely cosmetic.',
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve the JSON path
    # ------------------------------------------------------------------ #
    if args.results is not None:
        json_path = args.results
    elif args.mode and len(args.mode) == 1:
        # Single-mode shortcut: look for the per-mode file first
        key = _MODE_ALIAS.get(args.mode[0], args.mode[0])
        candidate = os.path.join(args.out_dir, f'keff_results_{key}.json')
        default   = os.path.join(args.out_dir, 'keff_results.json')
        json_path = candidate if os.path.exists(candidate) else default
    else:
        json_path = os.path.join(args.out_dir, 'keff_results.json')

    if not os.path.exists(json_path):
        sys.exit(f'ERROR: results file not found: {json_path!r}\n'
                 f'Run sensitivity_analysis.py first, or pass --results '
                 f'<path>.')

    with open(json_path) as f:
        results = json.load(f)

    print(f'Loaded {len(results)} cases from {json_path}')

    # ------------------------------------------------------------------ #
    # Override coupling flags from CLI if requested
    # ------------------------------------------------------------------ #
    if args.couple_fuel:
        for r in results:
            r['coupled_fuel_rho'] = True
    if args.couple_h2:
        for r in results:
            r['coupled_h2_rho'] = True

    # ------------------------------------------------------------------ #
    # Resolve which panels to draw
    # ------------------------------------------------------------------ #
    variables_in_json = {r['variable'] for r in results
                         if r['variable'] != 'baseline'}

    if args.mode:
        requested = []
        for m in args.mode:
            key = _MODE_ALIAS.get(m)
            if key is None:
                sys.exit(f'ERROR: unknown mode {m!r}.  '
                         f'Choose from: {", ".join(_MODE_ALIAS)}')
            requested.append(key)
        # Warn about any requested mode that has no data
        for key in requested:
            if key not in variables_in_json:
                print(f'WARNING: mode {key!r} requested but not found in '
                      f'{json_path!r} — it will be skipped.')
        panels = [k for k in requested if k in variables_in_json]
    else:
        # All modes present in the file, in canonical order
        panels = [m for m in ALL_MODES if m in variables_in_json]

    if not panels:
        sys.exit('ERROR: no plottable data found for the requested mode(s).')

    print(f'Plotting panels: {panels}')
    print('Reactivity coefficients (weighted linear fit):')
    make_figures(results, panels,
                 out_dir=args.out_dir,
                 show=not args.no_show)


if __name__ == '__main__':
    main()

# plot_sensitivity.py --mode fuel_radius --results sensitivity_results/keff_results_fuel_radius.json
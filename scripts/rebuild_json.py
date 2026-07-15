"""
rebuild_json.py
===============

Re-create keff_results.json by reading the OpenMC statepoint files that
sensitivity_analysis.py already produced, without re-running any simulations.

The script scans sensitivity_runs/<tag>/statepoint.*.h5, parses each tag
name back into (variable, delta), extracts k_eff and (optionally) kinetic
parameters, then writes the same JSON format that run_sensitivity_scan()
produces.  The output can be fed directly into plot_sensitivity.py.

Usage
-----
# Rebuild the full JSON from all statepoints found in sensitivity_runs/:
python rebuild_json.py

# Only include specific modes (others are silently ignored):
python rebuild_json.py --mode fuel_T beo_T

# Non-default directories or output path:
python rebuild_json.py --runs-dir my_runs --out sensitivity_results/rebuilt.json

# Print what would be found without writing anything:
python rebuild_json.py --dry-run
"""

import argparse
import glob
import json
import os
import re
import sys

import openmc


# ---------------------------------------------------------------------------
# Tag → (variable, delta) parser
# ---------------------------------------------------------------------------

# Each pattern captures the sign character and the magnitude string.
_TAG_PATTERNS = [
    # baseline (no sign, no magnitude)
    (re.compile(r'^baseline$'),
     lambda m: ('baseline', 0.0)),

    # fuel_T_p050K  /  fuel_T_m200K
    (re.compile(r'^fuel_T_([pm])(\d+)K$'),
     lambda m: ('fuel_T', float(m.group(2)) * (1 if m.group(1) == 'p' else -1))),

    # h2_rho_p50pct  /  h2_rho_m05pct   (stored as fraction in JSON: /100)
    (re.compile(r'^h2_rho_([pm])(\d+(?:\.\d+)?)pct$'),
     lambda m: ('h2_rho', float(m.group(2)) / 100.0 * (1 if m.group(1) == 'p' else -1))),

    # fuel_rho_p10pct
    (re.compile(r'^fuel_rho_([pm])(\d+(?:\.\d+)?)pct$'),
     lambda m: ('fuel_rho', float(m.group(2)) / 100.0 * (1 if m.group(1) == 'p' else -1))),

    # power_p10pct
    (re.compile(r'^power_([pm])(\d+(?:\.\d+)?)pct$'),
     lambda m: ('power', float(m.group(2)) / 100.0 * (1 if m.group(1) == 'p' else -1))),

    # beo_T_p302K  /  beo_T_m698K
    (re.compile(r'^beo_T_([pm])(\d+)K$'),
     lambda m: ('beo_T', float(m.group(2)) * (1 if m.group(1) == 'p' else -1))),

    # h2_T_p1000K  /  h2_T_m100K
    (re.compile(r'^h2_T_([pm])(\d+)K$'),
     lambda m: ('h2_T', float(m.group(2)) * (1 if m.group(1) == 'p' else -1))),

    # fuel_radius_p10pct  /  fuel_radius_m05pct  (stored as fraction: /100)
    (re.compile(r'^fuel_radius_([pm])(\d+(?:\.\d+)?)pct$'),
     lambda m: ('fuel_radius', float(m.group(2)) / 100.0 * (1 if m.group(1) == 'p' else -1))),
]

# Canonical order for the output JSON (baseline always first)
_VARIABLE_ORDER = ['baseline', 'fuel_T', 'h2_rho', 'power',
                   'fuel_rho', 'beo_T', 'h2_T', 'fuel_radius']


def parse_tag(tag):
    """Return (variable, delta) for a run directory tag, or None if unrecognised."""
    for pattern, extractor in _TAG_PATTERNS:
        m = pattern.match(tag)
        if m:
            return extractor(m)
    return None


# ---------------------------------------------------------------------------
# Statepoint extraction
# ---------------------------------------------------------------------------

def extract_from_statepoint(sp_path, collect_kinetics=True):
    """Open *sp_path* and return (keff, sigma, kinetics_dict).

    kinetics_dict is empty if collect_kinetics=False or data are absent.
    """
    sp = openmc.StatePoint(sp_path)
    keff    = sp.keff
    nominal = float(keff.nominal_value)
    stddev  = float(keff.std_dev)

    kinetics = {}
    if collect_kinetics:
        try:
            kin  = sp.get_kinetics_parameters()
            beta = kin.beta_effective   # uncertainties.ufloat
            gen  = kin.generation_time  # uncertainties.ufloat
            kinetics['beta_eff']         = float(beta.nominal_value)
            kinetics['sigma_beta_eff']   = float(beta.std_dev)
            kinetics['gen_time_s']       = float(gen.nominal_value)
            kinetics['sigma_gen_time_s'] = float(gen.std_dev)
        except Exception as exc:
            print(f'    [warn] kinetic parameters not available: {exc}')

    sp.close()
    return nominal, stddev, kinetics


# ---------------------------------------------------------------------------
# Main rebuild logic
# ---------------------------------------------------------------------------

def find_statepoint(run_dir):
    """Return the path to the (unique) statepoint file in *run_dir*, or None."""
    hits = glob.glob(os.path.join(run_dir, 'statepoint.*.h5'))
    if not hits:
        return None
    if len(hits) > 1:
        # Take the one with the highest batch number
        def _batch(p):
            m = re.search(r'statepoint\.(\d+)\.h5$', p)
            return int(m.group(1)) if m else 0
        hits.sort(key=_batch)
        print(f'    [warn] multiple statepoints in {run_dir!r}; '
              f'using highest batch: {os.path.basename(hits[-1])}')
        return hits[-1]
    return hits[0]


def rebuild(runs_dir='sensitivity_runs',
            out_path='sensitivity_results/keff_results.json',
            mode_filter=None,
            collect_kinetics=True,
            dry_run=False):
    """Scan *runs_dir*, read statepoints, write *out_path*.

    Parameters
    ----------
    runs_dir        : str        Root directory of per-case subdirectories.
    out_path        : str        Destination JSON path.
    mode_filter     : list[str] | None
                                 If given, only include these variable names.
    collect_kinetics: bool       Whether to extract β_eff and Λ_eff.
    dry_run         : bool       Print what would be done without writing.
    """
    if not os.path.isdir(runs_dir):
        sys.exit(f'ERROR: runs directory not found: {runs_dir!r}')

    tags = sorted(os.listdir(runs_dir))
    results = []
    skipped = []

    for tag in tags:
        run_dir = os.path.join(runs_dir, tag)
        if not os.path.isdir(run_dir):
            continue

        parsed = parse_tag(tag)
        if parsed is None:
            print(f'  [skip] unrecognised tag: {tag!r}')
            skipped.append(tag)
            continue

        variable, delta = parsed

        if mode_filter and variable not in mode_filter and variable != 'baseline':
            continue

        sp_path = find_statepoint(run_dir)
        if sp_path is None:
            print(f'  [skip] no statepoint in {run_dir!r}')
            skipped.append(tag)
            continue

        if dry_run:
            print(f'  [dry-run] {tag:35s}  variable={variable!r:10s}  '
                  f'delta={delta:+g}  sp={os.path.basename(sp_path)}')
            continue

        print(f'  Reading {tag} ...', end='  ', flush=True)
        try:
            keff, sigma, kinetics = extract_from_statepoint(
                sp_path, collect_kinetics=collect_kinetics)
        except Exception as exc:
            print(f'FAILED ({exc})')
            skipped.append(tag)
            continue

        print(f'k_eff = {keff:.5f} ± {sigma:.5f}')

        record = {
            'variable': variable,
            'delta':    delta,
            'keff':     keff,
            'sigma':    sigma,
            'tag':      tag,
            # Preserve the coupling fields as False by default; the original
            # values are not stored in the statepoint, so we cannot recover
            # them automatically.  Override with --couple-fuel / --couple-h2
            # in plot_sensitivity.py if needed.
            'coupled_fuel_rho': False,
            'coupled_h2_rho':   False,
        }
        record.update(kinetics)
        results.append(record)

    if dry_run:
        print(f'\n{len(tags)} directories scanned (dry run — nothing written).')
        return

    if not results:
        sys.exit('ERROR: no valid statepoints found — nothing to write.')

    # Sort: baseline first, then by variable canonical order, then by delta.
    order = {v: i for i, v in enumerate(_VARIABLE_ORDER)}
    results.sort(key=lambda r: (order.get(r['variable'], 99), r['delta']))

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nWrote {len(results)} cases to {out_path}')
    if skipped:
        print(f'Skipped {len(skipped)} directories: {skipped}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ALL_MODES = ['fuel_T', 'h2_rho', 'power', 'fuel_rho', 'beo_T', 'h2_T', 'fuel_radius']
_MODE_ALIAS = {
    'fuel':         'fuel_T',
    'h2':           'h2_rho',
    'beo':          'beo_T',
    'fuel_rho':     'fuel_rho',
    'power':        'power',
    'h2_T':         'h2_T',
    'fuel_radius':  'fuel_radius',
    # canonical names pass through
    'fuel_T':       'fuel_T',
    'h2_rho':       'h2_rho',
    'beo_T':        'beo_T',
}


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild keff_results.json from existing OpenMC statepoints.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--runs-dir', default='sensitivity_runs',
        help='Root directory containing per-case subdirectories '
             '(default: sensitivity_runs).',
    )
    parser.add_argument(
        '--out', default='sensitivity_results/keff_results.json',
        help='Output JSON path (default: sensitivity_results/keff_results.json).',
    )
    parser.add_argument(
        '--mode', nargs='+', metavar='MODE', default=None,
        help='Only include these variable(s): '
             + ', '.join(_ALL_MODES)
             + '.  Baseline is always included.  '
               'Defaults to all modes found.',
    )
    parser.add_argument(
        '--no-kinetics', action='store_true',
        help='Skip extraction of β_eff and Λ_eff (faster if not needed).',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would be processed without reading statepoints or '
             'writing the JSON.',
    )
    args = parser.parse_args()

    mode_filter = None
    if args.mode:
        mode_filter = []
        for m in args.mode:
            key = _MODE_ALIAS.get(m)
            if key is None:
                sys.exit(f'ERROR: unknown mode {m!r}.  '
                         f'Choose from: {", ".join(_MODE_ALIAS)}')
            mode_filter.append(key)
        mode_filter.append('baseline')   # always keep baseline

    rebuild(
        runs_dir=args.runs_dir,
        out_path=args.out,
        mode_filter=mode_filter,
        collect_kinetics=not args.no_kinetics,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
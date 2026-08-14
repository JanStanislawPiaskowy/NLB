"""
rebuild_json.py
===============

Re-create keff_results.json by reading the OpenMC statepoint files that
sensitivity_analysis.py already produced, without re-running any simulations.

The script scans <RUNS_DIR>/<tag>/statepoint.*.h5, parses each tag name back
into (variable, delta), extracts k_eff, the IFP kinetic parameters and the
four-factor decomposition, then writes the same JSON format that
run_sensitivity_scan() produces.  The output can be fed directly into
plot_sensitivity.py or compute_reactivity_and_plot().

Parity with sensitivity_analysis.py
-----------------------------------
Constants and helpers are imported from sensitivity_analysis rather than
duplicated, so the rebuilder cannot drift away from the scan it mirrors:

    RUNS_DIR / RESULTS_DIR      default input and output locations
    *_DELTAS_* lists            used to restore the exact deltas (see below)
    _extract_keff_h5py()        summary-less k_eff fallback
    _ff_to_dict()               four-factor record serialisation

If the import fails (running outside the repository, no OpenMC/gcr on the
path) the script falls back to local copies and simply omits the four-factor
block, printing a warning rather than dying.

Why deltas are snapped
----------------------
_tag() encodes the perturbation lossily: `int(delta)` for kelvin sweeps and
'%.0f' for percentage sweeps.  A delta of 0.075 becomes 'm08pct' and would
come back as 0.080, putting the point at the wrong abscissa in every fit.
Parsed deltas are therefore snapped onto the canonical delta lists whenever
the parsed value is within the encoding granularity of exactly one canonical
entry; anything else is kept verbatim and reported.

Usage
-----
# Rebuild the full JSON from all statepoints found in RUNS_DIR:
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

# Allow running from the repository root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Import the scan module so constants and helpers cannot drift
# ---------------------------------------------------------------------------

try:
    import sensitivity_analysis as _sa
except Exception as _exc:                                    # noqa: BLE001
    _sa = None
    print(f'[warn] sensitivity_analysis could not be imported ({_exc}).\n'
          f'       Falling back to local defaults; the four-factor block '
          f'cannot be rebuilt.')

try:
    from gcr.analysis.four_factors import compute_four_factors
except Exception:                                            # noqa: BLE001
    compute_four_factors = None

DEFAULT_RUNS_DIR    = getattr(_sa, 'RUNS_DIR',    'sensitivity_runs')
DEFAULT_RESULTS_DIR = getattr(_sa, 'RESULTS_DIR', 'sensitivity_results')

# Canonical delta lists, used only to undo the lossy tag encoding.
_CANONICAL_DELTAS = {
    'fuel_T':      list(getattr(_sa, 'FUEL_T_DELTAS_K',
                                [-5000, -2000, -1000, 1000, 2000, 5000])),
    'h2_rho':      list(getattr(_sa, 'H2_RHO_DELTAS',
                                [-0.10, -0.05, 0.05, 0.10])),
    'fuel_rho':    list(getattr(_sa, 'FUEL_RHO_DELTAS',
                                [-0.10, -0.05, 0.05, 0.10])),
    'beo_T':       list(getattr(_sa, 'BEO_T_DELTAS_K',
                                [-598, -498, -398, -198, -48,
                                 102, 302, 552, 602])),
    'h2_T':        list(getattr(_sa, 'H2_T_DELTAS_K',
                                [-1000, -600, -200, -100, 100,
                                 200, 300, 400, 600, 1000])),
    'fuel_radius': list(getattr(_sa, 'FUEL_RADIUS_DELTAS',
                                [-0.10, -0.05, 0.05, 0.10])),
}

# Variables whose delta is a temperature in kelvin.  _tag() truncates these
# with int(), so the parsed magnitude is at most 1 K below the true one.
_KELVIN_VARIABLES = ('fuel_T', 'beo_T', 'h2_T')

# Variables whose delta is a fraction written as a rounded whole percent.
# '%.0f' rounds, so the parsed value is at most 0.5 pct (0.005) off.
_PERCENT_VARIABLES = ('h2_rho', 'fuel_rho', 'fuel_radius', 'power')

_SNAP_TOLERANCE = {**{v: 1.0   for v in _KELVIN_VARIABLES},
                   **{v: 0.005 for v in _PERCENT_VARIABLES}}


# ---------------------------------------------------------------------------
# Tag -> (variable, delta) parser
# ---------------------------------------------------------------------------

# Each pattern captures the sign character and the magnitude string.  The
# magnitude accepts a decimal part throughout: _tag() does not currently emit
# one, but a hand-made or future directory that does must not be dropped.
_TAG_PATTERNS = [
    # baseline (no sign, no magnitude)
    (re.compile(r'^baseline$'),
     lambda m: ('baseline', 0.0)),

    # fuel_T_p5000K  /  fuel_T_m2000K
    (re.compile(r'^fuel_T_([pm])(\d+(?:\.\d+)?)K$'),
     lambda m: ('fuel_T', float(m.group(2)) * (1 if m.group(1) == 'p' else -1))),

    # h2_rho_p10pct  /  h2_rho_m05pct   (stored as fraction in JSON: /100)
    (re.compile(r'^h2_rho_([pm])(\d+(?:\.\d+)?)pct$'),
     lambda m: ('h2_rho', float(m.group(2)) / 100.0 * (1 if m.group(1) == 'p' else -1))),

    # fuel_rho_p10pct
    (re.compile(r'^fuel_rho_([pm])(\d+(?:\.\d+)?)pct$'),
     lambda m: ('fuel_rho', float(m.group(2)) / 100.0 * (1 if m.group(1) == 'p' else -1))),

    # power_p10pct -- legacy only; run_sensitivity_scan no longer accepts
    # 'power', but old run directories are still readable.
    (re.compile(r'^power_([pm])(\d+(?:\.\d+)?)pct$'),
     lambda m: ('power', float(m.group(2)) / 100.0 * (1 if m.group(1) == 'p' else -1))),

    # beo_T_p302K  /  beo_T_m598K
    (re.compile(r'^beo_T_([pm])(\d+(?:\.\d+)?)K$'),
     lambda m: ('beo_T', float(m.group(2)) * (1 if m.group(1) == 'p' else -1))),

    # h2_T_p1000K  /  h2_T_m100K
    (re.compile(r'^h2_T_([pm])(\d+(?:\.\d+)?)K$'),
     lambda m: ('h2_T', float(m.group(2)) * (1 if m.group(1) == 'p' else -1))),

    # fuel_radius_p10pct  /  fuel_radius_m05pct  (stored as fraction: /100)
    (re.compile(r'^fuel_radius_([pm])(\d+(?:\.\d+)?)pct$'),
     lambda m: ('fuel_radius', float(m.group(2)) / 100.0 * (1 if m.group(1) == 'p' else -1))),
]

# Canonical order for the output JSON (baseline always first).  Matches the
# order run_sensitivity_scan() appends its cases in.
_VARIABLE_ORDER = ['baseline', 'fuel_T', 'h2_rho', 'power',
                   'fuel_rho', 'beo_T', 'h2_T', 'fuel_radius']


def parse_tag(tag):
    """Return (variable, delta) for a run directory tag, or None if unrecognised."""
    for pattern, extractor in _TAG_PATTERNS:
        m = pattern.match(tag)
        if m:
            return extractor(m)
    return None


def snap_delta(variable, delta):
    """Restore the exact delta the scan used, undoing the tag rounding.

    Returns (delta, note).  *note* is None when nothing was changed, otherwise
    a short string describing what happened, for the caller to print.

    Snapping is refused -- and the parsed value kept -- when two canonical
    deltas lie within tolerance of the parsed one, because then the tag is
    genuinely ambiguous and guessing would silently move a data point.
    """
    candidates = _CANONICAL_DELTAS.get(variable)
    if not candidates:
        return delta, None

    tol = _SNAP_TOLERANCE.get(variable, 0.0)
    near = [c for c in candidates if abs(float(c) - delta) <= tol]

    if not near:
        return delta, (f'delta {delta:+g} is not in the canonical '
                       f'{variable} list (extra or hand-made case)')
    if len(near) > 1:
        return delta, (f'delta {delta:+g} is within {tol:g} of several '
                       f'canonical values {near}; kept as parsed')

    exact = float(near[0])
    if exact == delta:
        return delta, None
    return exact, f'delta restored {delta:+g} -> {exact:+g} from the tag rounding'


# ---------------------------------------------------------------------------
# Statepoint extraction
# ---------------------------------------------------------------------------

def _extract_keff_h5py(sp_path):
    """Local copy of the scan's summary-less fallback.

    Used when openmc.StatePoint cannot be opened because summary.h5 is absent
    or corrupt.  Kinetics parameters and four factors need tally linking (and
    so the summary), so only k_eff is recovered -- which is still the whole
    point of the file.  Previously this case raised and the whole run was
    dropped from the rebuilt JSON.
    """
    if _sa is not None and hasattr(_sa, '_extract_keff_h5py'):
        return _sa._extract_keff_h5py(sp_path)

    import h5py
    with h5py.File(sp_path, 'r') as f:
        # 'k_combined' stores [mean, std_dev] of the combined k-eff estimator.
        k = f['k_combined'][()]
        nominal = float(k[0])
        stddev  = float(k[1])
    print('    [warn] k_eff extracted via h5py fallback '
          '(summary.h5 unavailable - kinetics and four factors skipped).')
    return nominal, stddev, {}


def _ff_to_dict(record):
    """Serialise one four-factor record; delegates to the scan when available."""
    if _sa is not None and hasattr(_sa, '_ff_to_dict'):
        return _sa._ff_to_dict(record)

    from dataclasses import asdict, is_dataclass
    if is_dataclass(record):
        return asdict(record)
    return {k: v for k, v in vars(record).items() if not k.startswith('_')}


def extract_four_factors(sp_path):
    """Return the serialised four-factor records, or None if unavailable."""
    if compute_four_factors is None:
        return None
    try:
        return [_ff_to_dict(r) for r in compute_four_factors(sp_path)]
    except Exception as exc:                                 # noqa: BLE001
        print(f'    [warn] four-factor decomposition skipped ({exc})')
        return None


def extract_from_statepoint(sp_path, collect_kinetics=True,
                            collect_four_factors=True):
    """Open *sp_path* and return (keff, sigma, kinetics_dict, four_factors).

    kinetics_dict is empty and four_factors is None when the data are absent
    or the caller asked for them to be skipped.  A statepoint that cannot be
    linked to its summary still yields k_eff via the h5py fallback instead of
    being discarded.
    """
    try:
        sp = openmc.StatePoint(sp_path)
    except (OSError, KeyError) as exc:
        print(f'    [warn] StatePoint could not be opened ({exc}).')
        nominal, stddev, _ = _extract_keff_h5py(sp_path)
        return nominal, stddev, {}, None

    try:
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
            except Exception as exc:                         # noqa: BLE001
                print(f'    [warn] kinetic parameters not available: {exc}')
    finally:
        sp.close()

    four_factors = extract_four_factors(sp_path) if collect_four_factors else None
    return nominal, stddev, kinetics, four_factors


# ---------------------------------------------------------------------------
# Main rebuild logic
# ---------------------------------------------------------------------------

def find_statepoint(run_dir):
    """Return the path to the statepoint file in *run_dir*, or None.

    Looks one level down as well, so a run directory that keeps its OpenMC
    output in a subdirectory is still found instead of being reported as
    having no statepoint at all.
    """
    hits = glob.glob(os.path.join(run_dir, 'statepoint.*.h5'))
    if not hits:
        hits = glob.glob(os.path.join(run_dir, '**', 'statepoint.*.h5'),
                         recursive=True)
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


def coverage_report(results, mode_filter=None):
    """Print which canonical deltas made it into the rebuild and which did not.

    This is the check that says whether the JSON is complete: a case whose
    directory is missing, whose statepoint never got written or whose tag did
    not parse simply is not in *results*, and every downstream fit would
    quietly run on the surviving points.
    """
    found = {}
    for r in results:
        found.setdefault(r['variable'], set()).add(round(float(r['delta']), 6))

    print('\nCoverage against the canonical delta lists:')

    if 'baseline' not in found:
        print('  baseline       MISSING - reactivities cannot be computed '
              'without it')
    else:
        print('  baseline       present')

    for variable in _VARIABLE_ORDER:
        if variable == 'baseline':
            continue
        expected = _CANONICAL_DELTAS.get(variable)
        if not expected:
            continue
        got = found.get(variable, set())
        if not got and not (mode_filter and variable in mode_filter):
            continue                       # this sweep was simply never run

        want = {round(float(d), 6) for d in expected}
        missing = sorted(want - got)
        extra   = sorted(got - want)

        line = f'  {variable:<13s}{len(got & want)}/{len(want)}'
        if missing:
            line += '  missing: ' + ', '.join(f'{d:+g}' for d in missing)
        if extra:
            line += '  extra: ' + ', '.join(f'{d:+g}' for d in extra)
        print(line)


def rebuild(runs_dir=None,
            out_path=None,
            mode_filter=None,
            collect_kinetics=True,
            collect_four_factors=True,
            write_baseline_copy=True,
            dry_run=False):
    """Scan *runs_dir*, read statepoints, write *out_path*.

    Parameters
    ----------
    runs_dir            : str | None  Root directory of per-case subdirectories.
                                      Defaults to sensitivity_analysis.RUNS_DIR.
    out_path            : str | None  Destination JSON path.  Defaults to
                                      <RESULTS_DIR>/keff_results.json.
    mode_filter         : list[str] | None
                                      If given, only include these variables.
    collect_kinetics    : bool        Whether to extract beta_eff and Lambda_eff.
    collect_four_factors: bool        Whether to redo the four-factor decomposition.
    write_baseline_copy : bool        Also write <out dir>/baseline.json, as
                                      run_sensitivity_scan() does, so partial
                                      sweeps can reuse it via --baseline-json.
    dry_run             : bool        Print what would be done without writing.
    """
    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    out_path = out_path or os.path.join(DEFAULT_RESULTS_DIR, 'keff_results.json')

    if not os.path.isdir(runs_dir):
        sys.exit(f'ERROR: runs directory not found: {runs_dir!r}')

    print(f'Scanning {runs_dir!r}')

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
        delta, note = snap_delta(variable, delta)

        if mode_filter and variable not in mode_filter and variable != 'baseline':
            continue

        sp_path = find_statepoint(run_dir)
        if sp_path is None:
            print(f'  [skip] no statepoint in {run_dir!r}')
            skipped.append(tag)
            continue

        if dry_run:
            print(f'  [dry-run] {tag:35s}  variable={variable!r:14s}  '
                  f'delta={delta:+g}  sp={os.path.basename(sp_path)}')
            if note:
                print(f'            [note] {note}')
            continue

        print(f'  Reading {tag} ...', end='  ', flush=True)
        try:
            keff, sigma, kinetics, four_factors = extract_from_statepoint(
                sp_path,
                collect_kinetics=collect_kinetics,
                collect_four_factors=collect_four_factors)
        except Exception as exc:                             # noqa: BLE001
            print(f'FAILED ({exc})')
            skipped.append(tag)
            continue

        print(f'k_eff = {keff:.5f} +/- {sigma:.5f}')
        if note:
            print(f'    [note] {note}')

        # Field set and order mirror run_case() exactly, so a rebuilt file and
        # a freshly scanned one are interchangeable downstream.  The old
        # 'coupled_fuel_rho' / 'coupled_h2_rho' flags are gone: run_case() no
        # longer writes them, and emitting them as False asserted something
        # about the run that the statepoint cannot support.
        record = {
            'variable': variable,
            'delta':    float(delta),
            'keff':     float(keff),
            'sigma':    float(sigma),
            'tag':      tag,
        }
        record.update(kinetics)
        if four_factors:
            record['four_factors'] = four_factors
        results.append(record)

    if dry_run:
        print(f'\n{len(tags)} entries scanned (dry run - nothing written).')
        return

    if not results:
        sys.exit('ERROR: no valid statepoints found - nothing to write.')

    # Sort: baseline first, then by variable canonical order, then by delta.
    order = {v: i for i, v in enumerate(_VARIABLE_ORDER)}
    results.sort(key=lambda r: (order.get(r['variable'], 99), r['delta']))

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nWrote {len(results)} cases to {out_path}')

    if write_baseline_copy:
        baseline = next((r for r in results if r['variable'] == 'baseline'), None)
        if baseline is not None:
            bl_path = os.path.join(os.path.dirname(out_path) or '.',
                                   'baseline.json')
            with open(bl_path, 'w') as f:
                json.dump(baseline, f, indent=2)
            print(f'Baseline copy written to {bl_path}')

    n_kin = sum(1 for r in results if 'beta_eff' in r)
    n_ff  = sum(1 for r in results if 'four_factors' in r)
    print(f'  kinetic parameters recovered for {n_kin}/{len(results)} cases')
    print(f'  four-factor blocks recovered for {n_ff}/{len(results)} cases')

    if skipped:
        print(f'Skipped {len(skipped)} directories: {skipped}')

    coverage_report(results, mode_filter=mode_filter)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ALL_MODES = ['fuel_T', 'h2_rho', 'fuel_rho', 'beo_T', 'h2_T', 'fuel_radius']
_MODE_ALIAS = {
    # sensitivity_analysis.py's own --mode spellings
    'fuel':         'fuel_T',
    'h2':           'h2_rho',
    'beo':          'beo_T',
    'fuel_rho':     'fuel_rho',
    'h2_T':         'h2_T',
    'fuel_radius':  'fuel_radius',
    # canonical names pass through
    'fuel_T':       'fuel_T',
    'h2_rho':       'h2_rho',
    'beo_T':        'beo_T',
    # legacy directories only
    'power':        'power',
}


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild keff_results.json from existing OpenMC statepoints.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--runs-dir', default=DEFAULT_RUNS_DIR,
        help=f'Root directory containing per-case subdirectories '
             f'(default: {DEFAULT_RUNS_DIR}).',
    )
    parser.add_argument(
        '--out', default=None,
        help=f'Output JSON path.  Defaults to '
             f'{DEFAULT_RESULTS_DIR}/keff_results[_MODE].json, following the '
             f'naming run_sensitivity_scan() uses so a partial rebuild cannot '
             f'clobber a full one.',
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
        help='Skip extraction of beta_eff and Lambda_eff (faster if not needed).',
    )
    parser.add_argument(
        '--no-four-factors', action='store_true',
        help='Skip the four-factor decomposition.  It is recomputed from the '
             'statepoint tallies, which costs a few seconds per case.',
    )
    parser.add_argument(
        '--no-baseline-copy', action='store_true',
        help='Do not write the standalone baseline.json next to the output.',
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

    out_path = args.out
    if out_path is None:
        suffix = ''
        if args.mode:
            suffix = '_' + '_'.join(args.mode)
        out_path = os.path.join(DEFAULT_RESULTS_DIR,
                                f'keff_results{suffix}.json')

    rebuild(
        runs_dir=args.runs_dir,
        out_path=out_path,
        mode_filter=mode_filter,
        collect_kinetics=not args.no_kinetics,
        collect_four_factors=not args.no_four_factors,
        write_baseline_copy=not args.no_baseline_copy,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()

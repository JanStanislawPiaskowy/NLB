"""
run_photon_diag.py
==================

Two-minute diagnostic for the empty photon column in the coupled heating run.

State of play before running this
---------------------------------
The local-vs-coupled CSV diff has already established:

    local  (MT 901, photons off)   4639.59 MW
    coupled(MT 301, photons on)    4283.16 MW
    difference                      356.43 MW  = secondary photon production

    local overshoot   4639.59 - 4600 =  39.59 MW  (capture KERMA)
    coupled shortfall 4600 - 4283.16 = 316.84 MW
    39.59 + 316.84 = 356.43                        <- books balance exactly

So the library is fine: MT 301 correctly removes the gamma energy, photons are
born, and the printed leakage fraction of 0.807 cannot be neutrons alone
(a critical system is bounded by 1 - 1/nubar ~ 0.60), so photons are being
transported and are leaking.  Roughly 290 MW of gamma energy is depositing
inside the model and the photon bin of the cell tally saw 0.45 MW of it.

What this script separates
--------------------------
    B  estimator          photon deposition is a collision-site quantity;
                          the cell tally defaulted to tracklength
    C  particle tagging   photons hand their energy to secondary electrons;
                          if the deposition is tagged 'electron'/'positron'
                          a ParticleFilter(['neutron','photon']) misses it

Both are tested at once by scoring 'heating' over FOUR particle bins, twice:
once at the default estimator (reproducing the 15 h run) and once analog.
Global, unfiltered tallies -- no CellFilter, so the region map cannot
interfere and the cost is negligible.

Same hand-edited settings-block convention as run_heating.py.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import openmc

from gcr import GCRConfig, GCR

# =============================================================================
# SETTINGS
# =============================================================================

OUT_DIR = 'heating_runs/photon_diag'

N_PARTICLES = 5_000
N_BATCHES = 30
N_INACTIVE = 10

PHOTON_CUTOFF_EV = 1.0e3
XS_DIR = 'libraries_xs/jeff40_hdf5'
PHOTON_XS_DIR = 'libraries_xs/photon_hdf5'

#: Set False if your OpenMC build rejects electron/positron bins.
INCLUDE_LEPTONS = True

#: heating-local is nuclear DATA, not a transport option, so scoring it in a
#: photons-on run is only a lookup.  Set False if your build refuses it.
SCORE_HEATING_LOCAL = True

#: What the CSV diff already told us, in MW.  Used to size the verdict.
PHOTON_PRODUCTION_MW = 356.43
POWER_MW = 4600.0

MEV = 1.0e6


# =============================================================================

def make_config() -> GCRConfig:
    return GCRConfig(
        cross_sections_dir=XS_DIR,
        photon_cross_sections_dir=PHOTON_XS_DIR,
        n_axial_layers=10,
        h2_density_profile_path='settings/h2_density_profile.npz',
        batches=N_BATCHES,
        inactive=N_INACTIVE,
        particles=N_PARTICLES,
        photon_transport=True,
        photon_cutoff_ev=PHOTON_CUTOFF_EV,
    )


def particle_list() -> list:
    p = ['neutron', 'photon']
    if INCLUDE_LEPTONS:
        p += ['electron', 'positron']
    return p


def build_tallies() -> list:
    parts = particle_list()

    # -- the whole diagnostic, in two tallies --------------------------------
    # Same score, same bins, different estimator.  Whichever column holds the
    # ~290 MW names the bug.
    t_def = openmc.Tally(name='heat_by_particle_default')
    t_def.filters = [openmc.ParticleFilter(parts)]
    t_def.scores = ['heating']
    # estimator deliberately left alone: this is what the 15 h run did

    t_ana = openmc.Tally(name='heat_by_particle_analog')
    t_ana.filters = [openmc.ParticleFilter(list(parts))]
    t_ana.scores = ['heating']
    t_ana.estimator = 'analog'

    out = [t_def, t_ana]

    # -- normalisation and the MT301/MT901 gap -------------------------------
    t_n = openmc.Tally(name='norm')
    t_n.filters = [openmc.ParticleFilter(['neutron'])]
    t_n.scores = ['fission', 'fission-q-recoverable', 'heating']
    out.append(t_n)

    if SCORE_HEATING_LOCAL:
        t_nl = openmc.Tally(name='norm_local')
        t_nl.filters = [openmc.ParticleFilter(['neutron'])]
        t_nl.scores = ['heating-local']
        out.append(t_nl)

    t_pf = openmc.Tally(name='p_flux')
    t_pf.filters = [openmc.ParticleFilter(['photon'])]
    t_pf.scores = ['flux']
    out.append(t_pf)

    return out


def _frame(sp: openmc.StatePoint, name: str):
    for t in sp.tallies.values():
        if t.name == name:
            return t.get_pandas_dataframe()
    return None


def _scalar(sp: openmc.StatePoint, name: str, score: str) -> float:
    df = _frame(sp, name)
    if df is None:
        return float('nan')
    row = df[df['score'] == score]
    return float(row['mean'].sum()) if not row.empty else float('nan')


def _by_particle(sp: openmc.StatePoint, name: str) -> dict:
    """particle -> heating, eV per source particle."""
    df = _frame(sp, name)
    if df is None:
        return {}
    col = 'particle' if 'particle' in df.columns else df.columns[0]
    return {str(p): float(g['mean'].sum()) for p, g in df.groupby(col)}


def report(statepoint: str) -> None:
    with openmc.StatePoint(statepoint) as sp:
        fiss = _scalar(sp, 'norm', 'fission')
        qrec = _scalar(sp, 'norm', 'fission-q-recoverable')
        heat_n = _scalar(sp, 'norm', 'heating')
        heat_loc = _scalar(sp, 'norm_local', 'heating-local')
        p_flux = _scalar(sp, 'p_flux', 'flux')
        default = _by_particle(sp, 'heat_by_particle_default')
        analog = _by_particle(sp, 'heat_by_particle_analog')

    if not fiss > 0:
        raise SystemExit('no fissions scored -- nothing to diagnose')

    # eV/source -> MW, using the same normalisation the heating report uses
    to_mw = POWER_MW / qrec if qrec > 0 else float('nan')
    per_fis = lambda x: x / fiss / MEV

    print()
    print('  Photon-transport diagnostic')
    print(f'  statepoint : {statepoint}')
    print()
    print(f"  {'quantity':<32}{'MeV/fission':>13}{'MW':>12}")
    print('  ' + '-' * 60)
    print(f"  {'Q recoverable':<32}{per_fis(qrec):>13.2f}{qrec*to_mw:>12.1f}")
    print(f"  {'neutron heating (MT 301)':<32}{per_fis(heat_n):>13.2f}"
          f"{heat_n*to_mw:>12.1f}")
    if not np.isnan(heat_loc):
        gap = heat_loc - heat_n
        print(f"  {'neutron heating-local (MT 901)':<32}{per_fis(heat_loc):>13.2f}"
              f"{heat_loc*to_mw:>12.1f}")
        print(f"  {'  MT901 - MT301 = gamma born':<32}{per_fis(gap):>13.2f}"
              f"{gap*to_mw:>12.1f}")
    print()

    print(f"  heating by particle bin      {'default est.':>16}{'analog est.':>16}")
    print('  ' + '-' * 60)
    for p in particle_list():
        d = default.get(p, 0.0) * to_mw
        a = analog.get(p, 0.0) * to_mw
        print(f"  {p:<28}{d:>16.2f}{a:>16.2f}")
    print(f"  {'photon flux (cm/src n)':<28}{p_flux:>16.3e}")
    print()

    # ------------------------------------------------------------- verdict
    lepton_keys = ('electron', 'positron')
    lept_def = sum(default.get(k, 0.0) for k in lepton_keys) * to_mw
    lept_ana = sum(analog.get(k, 0.0) for k in lepton_keys) * to_mw
    ph_def = default.get('photon', 0.0) * to_mw
    ph_ana = analog.get('photon', 0.0) * to_mw
    target = 0.3 * PHOTON_PRODUCTION_MW      # generous: leakage takes some

    print('  VERDICT')
    if p_flux < 1.0e-9 or np.isnan(p_flux):
        print('    No photon flux at all. Contradicts the CSV diff -- check')
        print('    that photon_transport really took effect in this run.')
    elif lept_def > target or lept_ana > target:
        print('    The deposition is tagged to electrons/positrons, not to')
        print('    photons. One-line fix in heating.add_heating_tallies:')
        print("        particles = ['neutron', 'photon', 'electron', 'positron']")
        print('    No estimator change needed. Re-run is required, but the')
        print('    cell tally itself was otherwise correct.')
    elif ph_ana > target and ph_def < 0.1 * target:
        print('    Analog scores the gamma deposition, tracklength does not.')
        print("    Set estimator='analog' on the coupled heating tallies")
        print('    (heating.py lines 396 and 427). Consider keeping the')
        print('    neutron tally on tracklength and registering a separate')
        print('    analog photon-only tally, to protect neutron statistics.')
    elif ph_def > target:
        print('    Photon heating scores correctly here with the SAME estimator')
        print('    the 15 h run used. The fault is therefore in the')
        print('    cell-filtered tally, not the physics: compare cell_ids')
        print('    coverage and check for duplicate filter IDs in tallies.xml.')
    else:
        print(f'    Photons are born ({per_fis(heat_loc - heat_n):.1f} MeV/fission)')
        print('    but nothing deposits them under either estimator or any')
        print('    particle bin. Check settings.cutoff and electron_treatment,')
        print('    then the photon library entries in cross_sections.xml.')
    print()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    config = make_config()

    core = GCR(config, output_dir=OUT_DIR)
    core.build()
    core.register_tally(*build_tallies())
    core.run()

    report(os.path.join(OUT_DIR, f'statepoint.{config.batches}.h5'))


if __name__ == '__main__':
    main()

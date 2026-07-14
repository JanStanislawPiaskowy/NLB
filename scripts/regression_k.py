"""Fixed-seed regression check: is the model STILL the same model?

The whole refactor promises "same physics, different file layout".  This
script is how you HOLD it to that promise, and how you protect every
future change:

  1.  Before touching anything (i.e. on the OLD code), run a tiny
      fixed-seed case and record k_eff to all printed digits.
  2.  After each refactor step, run this script.  With the same seed,
      the same particle count and physically identical inputs, OpenMC is
      bit-reproducible on the same machine/build: k_eff must come out
      IDENTICAL, digit for digit.
  3.  Any deviation means the change altered the MODEL, not just the code
      layout -- stop and find out why before proceeding.

This converts a frightening 2,500-line refactor into a sequence of small
verified steps, and afterwards it doubles as a cheap pre-flight check
before expensive cluster campaigns.

The run is deliberately tiny (~minutes, not hours): statistics are
irrelevant here, only reproducibility.

Usage
-----
    python scripts/regression_k.py
    python scripts/regression_k.py --record   # append result to regression_k.log
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcr import GCRConfig, GCR

REGRESSION_SEED = 1
REGRESSION_BATCHES = 15
REGRESSION_INACTIVE = 5
REGRESSION_PARTICLES = 10_000
LOG_FILE = 'regression_k.log'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true',
                        help=f'Append the result to {LOG_FILE}.')
    args = parser.parse_args()

    config = GCRConfig(
        cross_sections_dir='libraries_xs/jeff40_hdf5',
        n_axial_layers=10,
        h2_density_profile_path='settings/h2_density_profile.npz',
        batches=REGRESSION_BATCHES,
        inactive=REGRESSION_INACTIVE,
        particles=REGRESSION_PARTICLES,
        seed=REGRESSION_SEED,
    )

    core = GCR(config, output_dir='settings_regression')
    core.build()
    core.run()

    import openmc
    sp = openmc.StatePoint(core.statepoint_path)
    k = sp.keff

    line = (f'seed={REGRESSION_SEED} batches={REGRESSION_BATCHES} '
            f'particles={REGRESSION_PARTICLES}  '
            f'k_eff = {k.nominal_value:.10f} +/- {k.std_dev:.2e}')

    print('\n' + '=' * 70)
    print('  REGRESSION RESULT (must match the recorded value EXACTLY):')
    print(f'  {line}')
    print('=' * 70 + '\n')

    if args.record:
        import datetime
        with open(LOG_FILE, 'a') as f:
            f.write(f'{datetime.datetime.now().isoformat()}  {line}\n')
        print(f'Recorded to {LOG_FILE}')


if __name__ == '__main__':
    main()

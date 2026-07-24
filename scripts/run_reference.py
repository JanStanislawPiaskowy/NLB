"""Reference GCR run -- the scenario that used to live in GCR.py's main().

A script's job is to describe ONE scenario: which config, which tallies,
which plots.  All physics, geometry and bookkeeping live in the gcr
package; if you find yourself writing a loop over materials or a formula
in a script, it probably belongs in the package instead.

Usage
-----
    python scripts/run_reference.py                # full build + run + plots
    python scripts/run_reference.py --geo-plot     # geometry plots only, no run
    python scripts/run_reference.py --plot-only    # re-plot from existing statepoint
    python scripts/run_reference.py --dry-run      # geometry-debug run (overlap check)
"""

import argparse
import os
import sys

# Allow running from the repository root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcr import GCRConfig, GCR
from gcr import plotting, tallies
from gcr.analysis.mass_estimate import print_u233_mass_estimate
from gcr.analysis.four_factors import add_four_factor_tallies


def make_config() -> GCRConfig:
    """The reference configuration.  Change parameters HERE (or load a JSON
    snapshot with GCRConfig.from_json) -- never by editing the package."""
    return GCRConfig(
        cross_sections_dir='libraries_xs/jeff40_hdf5',
        n_axial_layers=10,
        h2_density_profile_path='settings/h2_density_profile.npz',
        # Example overrides:
        # L=6.0 * 30.48,
        # th_atom_fraction=0.10,       # thorium sweep, one line
        # seed=1,                      # bit-reproducible run
        batches=250,
        inactive=50,
        particles=500_000,
        #temperature_BeO=1100,
    )


def add_reference_tallies(core: GCR, config: GCRConfig) -> None:
    """The reference tally set.  Order does not matter any more."""
    core.add_power_tally()
    core.add_kinetics_tally(num_groups=6)
    core.add_midplane_flux_tally(slice_thickness=20.0)
    add_four_factor_tallies(core)
    core.add_axial_flux_tally(
        slice_thickness=10.0, nz=700,
        z_min=-config.moderator_top_thickness - 20.0,
        z_max=config.L * 1.2,
    )
    core.add_fission_spectrum_tally()
    core.add_unweighted_lifetime_tally()


def plot_only(config: GCRConfig, output_dir: str = 'settings') -> None:
    """Re-plot from an existing statepoint WITHOUT building any geometry.

    The plot functions only need each tally's mesh and cutoffs, which the
    factories recreate instantly (they build objects; they do not run
    anything).  Cavity-centre markers are skipped since no cavities exist.
    """
    sp_path = os.path.join(output_dir, f'statepoint.{config.batches}.h5')

    power = tallies.power_tally(config)
    midplane = tallies.midplane_flux_tally(config, slice_thickness=20.0)
    axial = tallies.axial_flux_tally(
        config, slice_thickness=10.0, nz=700,
        z_min=-config.moderator_top_thickness - 20.0,
        z_max=config.L * 1.2,
    )

    plotting.plot_midplane_flux(config, midplane.mesh, midplane.meta, sp_path)
    plotting.plot_axial_flux(config, axial.mesh, axial.meta, sp_path)
    plotting.plot_power_distribution(config, power.mesh, sp_path, z_fraction=0.45)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip OpenMC; regenerate plots from the existing statepoint.')
    parser.add_argument('--geo-plot', action='store_true',
                        help='Build geometry and produce geometry plots, then exit.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Geometry-debug run with few particles (overlap check).')
    parser.add_argument('--alpha', type=float, default=None, metavar='A',
                        help='Override fuel_density_alpha (default: config value '
                             '2.0240). Applied inside GCR.build() to every fuel '
                             'material, canonical and per-layer alike.')
    args = parser.parse_args()

    config = make_config()
    if args.alpha is not None:
        config.fuel_density_alpha = args.alpha
        print(f'fuel_density_alpha overridden from CLI: {args.alpha}')

    if args.plot_only:
        plot_only(config)
        return

    core = GCR(config)          # include_tie_rods=False: rods stay OFF
    core.build()                # -> openmc.Model; config JSON snapshot at export

    print_u233_mass_estimate(core.materials, config)

    add_reference_tallies(core, config)

    core.export()               # write XML now so geometry plotting has inputs
    core.plot_geometry()

    if args.geo_plot:
        return

    core.run(dry_run=args.dry_run, map_geometry=True)
    if args.dry_run:
        return

    # --- Post-processing -----------------------------------------------------
    core.plot_midplane_flux()
    core.plot_axial_flux()
    core.plot_power_distribution(z_fraction=0.45)

    import openmc
    sp = openmc.StatePoint(core.statepoint_path)
    kin = sp.get_kinetics_parameters()
    print(f'Lambda_eff = {kin.generation_time}')
    print(f'beta_eff   = {kin.beta_effective}')


if __name__ == '__main__':
    main()

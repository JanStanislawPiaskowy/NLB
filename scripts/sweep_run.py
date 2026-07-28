"""
Code to sweep through the 4 libraries:
    jeff, tendl, jendl, endfb
Does not include all tallies, not enough space for that

Created 28.07.2026, J.S. Piaskowy
"""

import sys
import os

# Allow running from the repository root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcr import GCRConfig, GCR
from gcr.analysis.mass_estimate import print_u233_mass_estimate
from gcr.analysis.four_factors import add_four_factor_tallies

from scripts.sensitivity_analysis import print_material_temperatures

OUTPUT_DIR = 'sweep_runs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_config(library_dir_path: str) -> GCRConfig:
    
    config = GCRConfig(
            cross_sections_dir=library_dir_path,
            n_axial_layers=10,
            h2_density_profile_path='settings/h2_density_profile.npz',
            batches=250,
            inactive=50,
            particles=500_000,
            photon_transport=False,
            )

    return config

def add_reference_tallies(core: GCR, config: GCRConfig, N_GROUPS: int) -> None:
    """
    Reference tallies to be added.

    Parameters:
    N_GROUPS: int
            Number of precursor groups, 8 for jeff, 6 for others.
    """

    add_four_factor_tallies(core)
    core.add_kinetics_tally(num_groups=N_GROUPS)

def main():

    libraries = ('jeff40', 'endfb_viii.1', 'jendl5', 'tendl2025')
    print(libraries[0:3])
    for library in libraries[0:3:-1]:
        print(f'~~~~~~~~~~~~~~~~~~~~~ {library} ~~~~~~~~~~~~~~~~~~~~~~~~~')

        lib_path = f'libraries_xs/{library}_hdf5/'
        precursor_groups = 8 if library == 'jeff40' else 6

        config = make_config(library_dir_path=lib_path)
        
        core = GCR(config)
        OUT_DIR_LIB = f'{OUTPUT_DIR}/{library}/'
        os.makedirs(OUT_DIR_LIB, exist_ok=True)
        core.output_dir = OUT_DIR_LIB
        core.build()

        print_u233_mass_estimate(core.materials, config)
        print_material_temperatures(core)

        add_reference_tallies(core, config, precursor_groups)

        core.run(map_geometry=True)
        
        core.export()

        import openmc
        sp = openmc.StatePoint(core.statepoint_path)
        kin = sp.get_kinetics_parameters()
        print(f'Lambda_eff = {kin.generation_time}')
        print(f'beta_eff   = {kin.beta_effective}')

if __name__ == '__main__':
    main()

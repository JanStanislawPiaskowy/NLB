"""
run_heating.py
==============

Driver for the regional nuclear-heating calculation.  Hand-editable settings
block at the top, no argparse -- same pattern as ``run_reference.py``.

The whole point of this script is that MODE is the only switch.  It selects
the score, the photon transport flag, the output directory and the statepoint
path, and everything downstream follows from the config.  There is no second
place to keep in sync.

Recommended sequence
--------------------
1.  MODE = 'inventory'  Builds geometry only, writes cell_inventory.csv.
                        Check the 'region' column, fix the patterns in
                        heating.default_rules(), repeat until nothing is
                        'unassigned', then commit the region_map.json it
                        writes.  Costs seconds; skipping it means quoting a
                        table with cells silently missing from it.
2.  MODE = 'local'      Photon transport off, score 'heating-local'.  Cheap.
                        Gives the KERMA-at-collision-site answer.
3.  MODE = 'coupled'    Photon transport on, score 'heating' split into
                        neutron and photon.  The physically correct one.
4.  MODE = 'report'     Post-process an existing statepoint without
                        rebuilding geometry.

The (coupled - local) difference per region IS the gamma-transport
redistribution.  Run both; the delta plot is a result in its own right.
"""

from __future__ import annotations

import os
import sys

# Allow running from the repository root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openmc

from gcr import GCRConfig, GCR
from gcr.analysis import heating

# =============================================================================
# SETTINGS
# =============================================================================

MODE = 'inventory'                 # 'inventory' | 'local' | 'coupled' | 'report'

POWER_W = 4.6e9
OUT_DIR = f'heating_runs/{MODE}'

# --- transport ---------------------------------------------------------------
N_PARTICLES = 200_000
N_BATCHES = 220
N_INACTIVE = 20
PHOTON_CUTOFF_EV = 1.0e3         # raise to 1e4 for speed, at some accuracy cost

# --- nuclear data ------------------------------------------------------------
XS_DIR = 'libraries_xs/jeff40_hdf5'
PHOTON_XS_DIR = 'libraries_xs/photon_hdf5'

# --- region classification ---------------------------------------------------
REGION_MAP_JSON = 'settings/region_map.json'
USE_FROZEN_MAP = True            # False forces re-classification from the rules

# --- optional extras ---------------------------------------------------------
ADD_DAMAGE = True
ADD_MESH_MAP = True
MESH_MAP_DIM = (80, 80, 80)      # deposition is diffuse; 600^3 is unnecessary
RUN_VOLUME_CALC = False          # needed only for W/cm^3, Mrad/s and DPA
VOLUME_SAMPLES = 10_000_000

# --- post-processing ---------------------------------------------------------
# For the coupled-vs-local delta plot.  Written by the MODE='local' run.
LOCAL_ROWS_CSV = 'heating_runs/local/heating_report.csv'

# Transparent-wall dose rate: SiO2 mass over all seven cavities, kg.
# Leave None to skip.
TRANSPARENT_WALL_MASS_KG = None


# =============================================================================

def make_config() -> GCRConfig:
    """Reference geometry, with the run parameters this study needs.

    photon_transport is derived from MODE and nowhere else.  Because the
    package acts on that one flag in both places it matters (settings, and
    the photon entries in cross_sections.xml), 'coupled' cannot half-happen.
    """
    return GCRConfig(
        cross_sections_dir=XS_DIR,
        photon_cross_sections_dir=PHOTON_XS_DIR,
        n_axial_layers=10,
        h2_density_profile_path='settings/h2_density_profile.npz',
        #temperature_BeO=1100,
        batches=N_BATCHES,
        inactive=N_INACTIVE,
        particles=N_PARTICLES,
        photon_transport=(MODE == 'coupled'),
        photon_cutoff_ev=PHOTON_CUTOFF_EV,
    )


def statepoint_path(config: GCRConfig) -> str:
    """Where this MODE's statepoint lives.  Same expression GCR uses, so
    'report' can never point at a statepoint from a different batch count."""
    return os.path.join(OUT_DIR, f'statepoint.{config.batches}.h5')


def resolve_region_map(core: GCR) -> dict:
    """The frozen JSON map if it exists, otherwise classify from the rules."""
    if USE_FROZEN_MAP and os.path.exists(REGION_MAP_JSON):
        region_map = heating.load_region_map(REGION_MAP_JSON)
        print(f'[run] frozen region map: {REGION_MAP_JSON} '
              f'({len(region_map)} cells)')
        return region_map

    region_map, unassigned = heating.classify_cells(
        core.geometry, z_active=(0.0, core.config.L))
    if unassigned:
        raise SystemExit(
            f'{len(unassigned)} cells unassigned. Run MODE="inventory", fix '
            'the patterns in heating.default_rules(), and try again.')
    return region_map


def build_mesh(core: GCR) -> openmc.RegularMesh:
    """Coarse Cartesian mesh spanning the model bounding box."""
    mesh = openmc.RegularMesh()
    mesh.dimension = MESH_MAP_DIM
    bb = core.geometry.bounding_box
    mesh.lower_left = bb.lower_left
    mesh.upper_right = bb.upper_right
    return mesh


def report(config: GCRConfig, region_map: dict, mode: str) -> list:
    """Statepoint -> MW-per-region table, CSV, LaTeX and figures."""
    sp = statepoint_path(config)
    if not os.path.exists(sp):
        raise SystemExit(f'no statepoint at {sp}; run MODE="{mode}" first')

    rows = heating.heating_report(
        sp, POWER_W, region_map, mode=mode,
        csv_path=os.path.join(OUT_DIR, 'heating_report.csv'),
        tex_path=os.path.join(OUT_DIR, 'heating_report.tex'),
    )
    heating.plot_heating_comparison(
        rows, os.path.join(OUT_DIR, 'heating_comparison.pdf'))

    # The gamma-redistribution figure needs the local run to exist already.
    if mode == 'coupled' and os.path.exists(LOCAL_ROWS_CSV):
        import csv
        with open(LOCAL_ROWS_CSV) as fh:
            rows_local = []
            for r in csv.DictReader(fh):
                r['total_MW'] = float(r['total_MW'])
                rows_local.append(r)
        heating.plot_local_vs_coupled(
            rows_local, rows, os.path.join(OUT_DIR, 'gamma_redistribution.pdf'))

    if TRANSPARENT_WALL_MASS_KG:
        heating.transparent_wall_dose_rate(
            sp, POWER_W, region_map, TRANSPARENT_WALL_MASS_KG)

    return rows


def main() -> None:
    if MODE not in ('inventory', 'local', 'coupled', 'report'):
        raise SystemExit(f'unknown MODE {MODE!r}')

    os.makedirs(OUT_DIR, exist_ok=True)
    config = make_config()

    # ----------------------------------------------------------------- report
    # No geometry build: the region map and the statepoint are all that is
    # needed, so this is instant.
    if MODE == 'report':
        region_map = heating.load_region_map(REGION_MAP_JSON)
        report(config, region_map, mode='coupled')
        return

    core = GCR(config, output_dir=OUT_DIR)
    core.build()

    # -------------------------------------------------------------- inventory
    if MODE == 'inventory':
        mapping = heating.dump_cell_inventory(
            core.geometry,
            path=os.path.join(OUT_DIR, 'cell_inventory.csv'),
            z_active=(0.0, config.L),
        )
        heating.write_region_map(mapping, REGION_MAP_JSON)
        print('\nOpen the CSV and check the region column.  Correct')
        print('heating.default_rules() until nothing is "unassigned",')
        print('re-run this mode, then move on to MODE="local".')
        return

    # ------------------------------------------------------------------- runs
    region_map = resolve_region_map(core)

    heating.add_heating_tallies(
        core,
        mode=MODE,
        region_map=region_map,
        add_damage=ADD_DAMAGE,
        add_normalisation=True,
        mesh=build_mesh(core) if ADD_MESH_MAP else None,
    )

    if RUN_VOLUME_CALC:
        heating.add_volume_calculation(core, region_map, samples=VOLUME_SAMPLES)

    # One call: writes materials/geometry/settings/tallies/cross_sections XML
    # into OUT_DIR (photon entries included, because config.photon_transport)
    # and executes OpenMC there.
    core.run()

    # Freeze the map that produced THIS statepoint next to it.
    heating.write_region_map(region_map, os.path.join(OUT_DIR, 'region_map.json'))

    report(config, region_map, mode=MODE)


if __name__ == '__main__':
    main()

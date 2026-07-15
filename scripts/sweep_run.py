# run_sweep.py
import os
import csv
import numpy as np
import openmc

from GCR import GCR, GCRConfig, ft_to_cm

N_BATCHES = 150
N_INACTIVE = 25
N_PARTICLES = 300_000

def run_one(config: GCRConfig, output_dir: str) -> dict:
    """Build, run and harvest results for a single GCRConfig."""
    # Disable HDF5 POSIX file locking — prevents summary.h5 write failures
    # on NFS / shared filesystems (errno=11 "Resource temporarily unavailable").
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    core = GCR(config)
    core.output_dir = output_dir          # keep each run isolated
    os.makedirs(output_dir, exist_ok=True)

    core.set_materials()
    if config.n_axial_layers > 1:
        core._create_layered_propellant_materials()
        core._create_layered_fuel_materials()


    FUEL_PREFIXES = ('fuel_inner', 'fuel_outer')

    _seen = set()
    for material in core.materials.values():
        if id(material) in _seen:
            continue
        _seen.add(id(material))
        if material.name == 'fuel' or material.name.startswith(FUEL_PREFIXES):
            material.set_density('g/cm3', material.density * config.fuel_density_alpha)

    # --- geometry ---
    # NOTE: this block must stay in sync with main() in GCR.py if the
    # cavity-construction logic changes there.
    core.build_cavity(tilt=config.tilt)

    sixty_deg = np.pi / 3
    hl  = config.r_inlet * 1.4
    phi = 2 * config.tilt
    for i in range(6):
        y0 = hl * np.sin(sixty_deg) * (1 + np.cos(phi))
        z0 = hl * np.sin(sixty_deg) * np.sin(phi)
        theta = -i * np.pi / 3
        xp = -y0 * np.sin(theta)
        yp =  y0 * np.cos(theta)
        core.build_cavity(x0=xp, y0=yp, z0=z0, tilt=config.tilt,
                          cavity_angle_zz=theta, cavity_angle_xx=phi)

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
    core.export_geometry()

    core.add_power_tally()
    core.add_flux_tally()
    core.add_kinetics_tally(num_groups=6)

    core.run()

    # --- harvest ---
    sp_path      = os.path.join(core.output_dir,
                                f'statepoint.{core.settings.batches}.h5')
    summary_path = os.path.join(core.output_dir, 'summary.h5')

    # If summary.h5 is corrupt (partial write from a locking failure),
    # remove it so StatePoint does not crash with
    # "bad object header version number".
    if os.path.exists(summary_path):
        try:
            import h5py as _h5
            _h5.File(summary_path, 'r').close()
        except OSError:
            print('    [warn] summary.h5 is corrupt — removing it before StatePoint open.')
            os.remove(summary_path)

    sp = openmc.StatePoint(sp_path)
    kin = sp.get_kinetics_parameters()

    result = {
        'k_eff':       float(sp.keff.nominal_value),
        'k_eff_std':   float(sp.keff.std_dev),
        'beta_eff':    float(kin.beta_effective.nominal_value),
        'beta_std':    float(kin.beta_effective.std_dev),
        'Lambda_eff':  float(kin.generation_time.nominal_value),
        'Lambda_std':  float(kin.generation_time.std_dev),
    }
    sp.close()
    return result


def make_config(**overrides) -> GCRConfig:
    """Start from a baseline and apply overrides."""
    base = dict(
        n_axial_layers=10,
        h2_density_profile_path='settings/h2_density_profile.npz',
        fuel_density_alpha=2.0240,
    )
    base.update(overrides)
    return GCRConfig(**base)


if __name__ == '__main__':
    # --- sweep axis 1: cross-section libraries ---
    XS_ROOT = 'libraries_xs'
    xs_libraries = {
        'endfb_viii.1': f'{XS_ROOT}/endfb_viii.1_hdf5',
        'jeff_4.0':  f'{XS_ROOT}/jeff40_hdf5',
        'jendl5':     f'{XS_ROOT}/jendl5_hdf5',
        'tendl2025':      f'{XS_ROOT}/tendl2025_hdf5',
    }
    # NOTE: point each entry at whatever path your GCR class expects
    # (folder vs cross_sections.xml — adjust to match cross_sections_dir usage).

    runs = []
    for label, xs_path in xs_libraries.items():
        runs.append({
            'label': f'xs_{label}',
            'overrides': {'cross_sections_dir': xs_path},
        })

    # --- execute ---
    results = []
    for r in runs:
        print(f'\n========== {r["label"]} ==========')
        cfg = make_config(**r['overrides'])
        out_dir = os.path.join('sweep_runs', r['label'])
        try:
            res = run_one(cfg, out_dir)
            cfg.to_json(os.path.join(out_dir, 'config.json'))
        except Exception as e:
            print(f'  FAILED: {e}')
            res = {'k_eff': None, 'k_eff_std': None,
                   'beta_eff': None, 'beta_std': None,
                   'Lambda_eff': None, 'Lambda_std': None,
                   'error': str(e)}
        res['label'] = r['label']
        res.update(r['overrides'])
        results.append(res)

    # --- save table ---
    os.makedirs('sweep_runs', exist_ok=True)
    fieldnames = ['label', 'cross_sections_dir',
                  'k_eff', 'k_eff_std',
                  'beta_eff', 'beta_std',
                  'Lambda_eff', 'Lambda_std', 'error']
    with open('sweep_runs/results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    print('\nDone. Summary:')
    for r in results:
        print(f'  {r["label"]:25s}  k={r["k_eff"]}  β={r["beta_eff"]}  Λ={r["Lambda_eff"]}')
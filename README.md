# gcr -- OpenMC model of the seven-cavity NLB gas-core reactor


## Package map

```
gcr/
|-- config.py          GCRConfig + unit constants + nuclide lists (imports nothing)
|-- transforms.py      rotation_matrix (replaces the tool_functions dependency)
|-- materials.py       compositions, axial layers, S(a,b), alpha scaling, XS library
|-- geometry/
|   |-- hexmaths.py    PURE NUMPY: hex planes, ridge lines, cavity placements
|   |-- cavity.py      build_cavity(cfg, materials, layered, placement) -> Cavity
|   |-- tie_rods.py    build_tie_rods(...) -> TieRods   [implemented, OFF by default]
|   |-- moderator.py   graphite cone, end caps, nozzle, bounding sphere
|   `-- overlaps.py    shared-midplane trimming between neighbouring cavities
|-- tallies.py         tally FACTORIES (build & return; never touch XML)
|-- plotting.py        colour maps, geometry plots, statepoint plots
|-- analysis/
|   |-- mass_estimate.py   U-233 inventory printout
|   `-- four_factors.py    k_inf = eps*p*f*eta decomposition
`-- model.py           class GCR: build() -> openmc.Model, registry, run()

scripts/run_reference.py   the old main(), as a scenario script
scripts/regression_k.py    fixed-seed bit-reproducibility check
tests/test_hexmaths.py     pure-numpy geometry tests (no OpenMC needed)
GCR.py                     legacy shim: old `from GCR import ...` still works
```

**Dependency rule (the one rule):** imports point downwards only.
`config` imports nothing; `materials`/`geometry`/`tallies` import `config`;
`model` imports all of them; `plotting`/`analysis` read configs and
statepoints.  If a change tempts you to import upwards, the design is
telling you the code is in the wrong module.

## Quick start

```python
from gcr import GCRConfig, GCR

config = GCRConfig(
    cross_sections_dir='libraries_xs/jeff40_hdf5',
    n_axial_layers=10,
    h2_density_profile_path='settings/h2_density_profile.npz',
)

core = GCR(config)
core.build()                      # geometry+materials+settings -> openmc.Model

core.add_power_tally()            
core.add_kinetics_tally(num_groups=6)

core.run()                        # exports XML, then runs OpenMC
core.plot_power_distribution()
```

Or simply: `python scripts/run_reference.py` (flags: `--geo-plot`,
`--plot-only`, `--dry-run`).

## Reproducibility workflow

1. **Config snapshot.** Every `export()`/`run()` writes
   `settings/gcr_config.json` next to the statepoint.  Any old result can
   be rebuilt with `GCRConfig.from_json(...)`.
2. **Fixed seed.** `GCRConfig(seed=1)` makes runs bit-reproducible on the
   same machine/OpenMC build.
3. **Pure-maths tests.** `pytest tests/ -v` verifies the hexagon/ridge/
   placement mathematics in milliseconds, without OpenMC or nuclear data.


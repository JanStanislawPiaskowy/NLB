# gcr -- OpenMC model of the seven-cavity NLB gas-core reactor

Refactored from the monolithic `GCR.py` (2,510 lines) into a package.
**Same physics, new layout**: every numeric expression was ported 1:1;
the deliberate exceptions are listed under *Migration notes* below.

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

core.add_power_tally()            # order of add_* calls no longer matters
core.add_kinetics_tally(num_groups=6)

core.run()                        # exports ALL XML once, then runs OpenMC
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
3. **Regression harness.** `python scripts/regression_k.py` runs a tiny
   fixed-seed case and prints k_eff to 10 digits.  Record it once on the
   old code; after ANY structural change it must match digit for digit.
4. **Pure-maths tests.** `pytest tests/ -v` verifies the hexagon/ridge/
   placement mathematics in milliseconds, without OpenMC or nuclear data.

## Tie rods

Implemented in `gcr/geometry/tie_rods.py` but **excluded by default**
(`GCR(include_tie_rods=False)`).  Relative to the old attempt: duplicate
ridges are merged by LINE (direction + perpendicular offset) instead of
start point, rods sit on the averaged ridge, and the moderator carve is
BOUNDED by the rod cap planes (the old infinite-cylinder carve left
undefined graphite voids -> lost particles).  To enable:

```python
core = GCR(config, include_tie_rods=True)
core.build()
core.run(dry_run=True)     # ALWAYS: overlap/lost-particle check first
```

Rods span the fuel-zone length only; extending them axially requires also
carving the end moderator (see the module docstring).

## Migration notes (deliberate deviations from the original)

* **Be density kept at 1.8392 g/cm3** despite being a suspected copy-paste
  of the graphite value (Be metal ~1.85): changing it would break the
  regression identity.  Flagged with a TODO in `materials.py`; change it
  in a separate, physics-reviewed commit.
* **plots.xml now actually contains the two slice plots.**  The original
  defined them but exported an empty `openmc.Plots([])`.  The huge voxel
  plot is opt-in (`plot_geometry(include_voxel=True)`).
* **tool_functions no longer needed**: `rotation_matrix` is internal
  (`gcr/transforms.py`); convention verified by the ridge-coincidence tests.
* **Dead code removed**: unused `hl_out`, unused `colour_map_tori`, the
  never-called `_resolve_liner_overlaps` (kept as a documented reference
  function in `overlaps.py`), the commented `_report` density printer.
* **Cosmetics**: 'grapihte moderator' typo fixed; a duplicated
  `& -PlaneFuelZoneStart` term removed (identical region).
* **Promoted to GCRConfig** (identical defaults): batches/inactive/
  particles, hexagon side ratio, moderator cone radius, nozzle throat
  radius, fuel temperatures/partial pressures, `th_atom_fraction`,
  temperature tolerance, seed, tie-rod merge tolerances.

## Suggested first verification on the cluster

```bash
# once, on the OLD code: record the baseline
python old/GCR.py   # with batches=15, inactive=5, particles=10000, seed=1
# then, on the NEW code:
python scripts/regression_k.py --record
# the two k_eff values must be identical to the last digit
```

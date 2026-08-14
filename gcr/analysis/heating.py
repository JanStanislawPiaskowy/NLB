"""
heating.py
==========

Regional nuclear-heating tallies and post-processing for the seven-cavity
nuclear light bulb (NLB) gas-core reactor OpenMC model.

Why this exists
---------------
The existing power tally in ``GCR.py`` scores ``fission-q-recoverable``, i.e.

    Q_recov [eV] * Sigma_f [1/cm] * phi [cm]      (eV per source particle)

which attributes the *entire* recoverable energy to the point at which the
fission occurred.  For a gas core this is a fission-rate map wearing an energy
costume: the prompt gammas born in a ~0.04 lb/ft^3 uranium cloud are
essentially unattenuated locally and deposit their energy metres away, in the
BeO, the graphite and the structure.  Any statement about "heat deposition in
region X" therefore cannot be made from that tally.

This module adds true deposition tallies and reduces them to a MW-per-region
table directly comparable with Table VIII of L-910900-16.

Two run modes
-------------
``local``    ``GCRConfig(photon_transport=False)``, score ``heating-local``
             (ENDF MT=901 KERMA: energy of secondary photons is deposited at
             the neutron collision site).  Cheap; run it first.

``coupled``  ``GCRConfig(photon_transport=True)``, score ``heating`` split by
             a ParticleFilter into neutron and photon components (ENDF MT=301
             neutron KERMA + genuinely transported photon deposition).  This
             is the physically correct one.

The mode is set in ONE place -- the config -- and this module refuses to build
tallies whose score contradicts it.  ``add_heating_tallies`` takes the built
``GCR`` core and registers through ``core.register_tally``, so the heating
tallies are written by the same single ``core.export()`` as everything else.

The region-by-region difference ``coupled - local`` *is* the gamma-transport
redistribution.  It is a result in its own right and worth a figure.

Known omissions -- state these explicitly in the report
-------------------------------------------------------
1.  Delayed beta and delayed gamma energy (~19 MeV/fission, ~10 % of the
    recoverable total) is not carried by prompt transport.  Expect

        sum(heating) / sum(fission-q-recoverable)  ~  0.90

    ``heating_report`` prints this ratio as a QA check.  UARL's Table VIII
    lists "Neutron, Gamma, and Beta" heating for the transparent structure,
    so their number includes a component yours structurally cannot.

2.  Fission-fragment kinetic energy (~170 MeV of ~200 MeV) is treated by the
    KERMA numbers as deposited *locally at the fission site*.  At NLB fuel
    densities the fragment range is not negligible, so the fuel-region figure
    is an upper bound and the buffer-gas / transparent-wall figures are lower
    bounds.  OpenMC cannot resolve this; an SRIM-style range calculation is
    required separately.  Flag it as an assumption (A-MAT).

3.  Conduction is not modelled.  UARL's flow-divider and tie-rod entries are
    "Neutron and Gamma *and Conduction*", so expect to undershoot those two.

Dependencies: numpy, openmc.  matplotlib only for the plotting helpers.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import openmc

EV_TO_J = 1.602176634e-19
GY_PER_MRAD = 1.0e4          # 1 Mrad = 10^4 Gy


# =============================================================================
# SETTINGS BLOCK -- edit these, no CLI
# =============================================================================

#: Tally names used throughout.  Post-processing looks tallies up by name, not
#: by id, so you may add/remove other tallies freely without breaking this.
T_HEATING = "heating_by_cell"
T_DAMAGE = "damage_by_cell"
T_NORM = "norm_fission_q"
T_HEATING_MESH = "heating_mesh"

#: Axial extent of the *cavity* region, used to separate the radial BeO
#: moderator from the upper/lower end moderators by cell centroid.
#: Set from GCRConfig: (0.0, cavity_length).
Z_ACTIVE_DEFAULT: Tuple[float, float] = (0.0, 182.8)

#: Reference values, L-910900-16 Table VIII, "Modification II (Present Report)"
#: column.  ``comparable`` marks the rows whose heating mechanism is purely
#: neutron + gamma and which your neutronics can therefore be judged against.
#: The thermal-radiation-dominated rows are carried for completeness but must
#: NOT be used as validation targets.
REFERENCE_MW: Dict[str, Dict[str, object]] = {
    "pressure_shell":     {"mw": 12.5,  "comparable": True,
                            "mech": "Neutron and gamma"},
    "nozzles":             {"mw": 0.2,   "comparable": True,
                            "mech": "Neutron and gamma"},
    "flow_divider":        {"mw": 7.8,   "comparable": True,
                            "mech": "Neutron, gamma and conduction"},
    "tie_rods":            {"mw": 5.7,   "comparable": True,
                            "mech": "Neutron, gamma and conduction"},
    "end_moderators":      {"mw": 75.8,  "comparable": True,
                            "mech": "Neutron and gamma"},
    "beo_moderator":       {"mw": 96.9,  "comparable": True,
                            "mech": "Neutron and gamma"},
    "graphite_moderator":  {"mw": 52.3,  "comparable": True,
                            "mech": "Neutron and gamma"},
    "hydrogen_direct":     {"mw": 56.6,  "comparable": True,
                            "mech": "Direct neutron, gamma and beta"},
    # --- not clean neutronics targets ---------------------------------------
    "transparent_wall":    {"mw": 54.4,  "comparable": False,
                            "mech": "Thermal radiation, convection, n, gamma, beta"},
    "cavity_end_walls":    {"mw": 195.1, "comparable": False,
                            "mech": "Thermal radiation and conduction"},
    "propellant_liners":   {"mw": 173.0, "comparable": False,
                            "mech": "Thermal radiation and conduction"},
    "fuel_circuit":        {"mw": 159.0, "comparable": False,
                            "mech": "Removal of heat from fuel"},
}

#: Sum of the ``comparable`` rows: the number your model should reproduce.
REFERENCE_COMPARABLE_TOTAL_MW = sum(
    v["mw"] for v in REFERENCE_MW.values() if v["comparable"]
)  # = 307.8 MW = 6.7 % of 4600 MW
#: Regions drawn in the comparison bar chart, in order.  This is deliberately
#: NOT the same set as the 'comparable' rows: the transparent wall and the
#: pressure shell are shown because their nuclear load matters for the damage
#: and shielding arguments, even though the transparent-wall Table VIII entry
#: is not a clean neutronics target.
PLOT_REGIONS: List[str] = [
    "end_moderators",
    "beo_moderator",
    "graphite_moderator",
    "hydrogen_direct",
    "transparent_wall",
    "pressure_shell",
]

#: Axis labels for the bar chart.  '\n' is a line break in the tick label.
PLOT_LABELS: Dict[str, str] = {
    "end_moderators":     "Upper and lower\nend moderators",
    "beo_moderator":      "Beryllium oxide\nmoderator",
    "graphite_moderator": "Graphite\nmoderator",
    "hydrogen_direct":    "Direct hydrogen\nheating",
    "transparent_wall":   "Transparent\nwalls",
    "pressure_shell":     "Pressure\nshell",
}

#: Ionising dose-rate target for the transparent wall, L-990929-3: the
#: full-scale engine nominal is 5 Mrad/s, which is the figure the whole
#: fused-silica-versus-BeO radiation damage argument is calibrated against.
TRANSPARENT_WALL_DOSE_TARGET_MRAD_S = 5.0


# =============================================================================
# Region classification
# =============================================================================

@dataclass
class RegionRule:
    """One classification rule.  The first matching rule wins.

    ``cell_name`` and ``material_name`` are regular expressions applied
    case-insensitively.  An empty pattern matches anything.  ``z_range``, if
    given, additionally requires the cell's bounding-box centroid to lie in
    that interval -- this is how the radial BeO reflector is separated from
    the upper and lower end moderators when they share a material.
    """
    region: str
    cell_name: str = ""
    material_name: str = ""
    z_range: Optional[Tuple[float, float]] = None

    def matches(self, cname: str, mname: str, zc: Optional[float]) -> bool:
        if self.cell_name and not re.search(self.cell_name, cname, re.I):
            return False
        if self.material_name and not re.search(self.material_name, mname, re.I):
            return False
        if self.z_range is not None:
            if zc is None or not (self.z_range[0] <= zc <= self.z_range[1]):
                return False
        return True


def default_rules(z_active: Tuple[float, float] = Z_ACTIVE_DEFAULT) -> List[RegionRule]:
    """Starting-point rules for the NLB model.

    These are *guesses* at your naming.  Run :func:`dump_cell_inventory` once,
    look at the ``region`` column, and correct these patterns (or hand-edit the
    generated JSON map) before trusting any number.
    """
    return [
        # specific structures by material 
        #RegionRule("nozzles",            cell_name=r"nozzle"),
        RegionRule("tie_rods",           cell_name=r"tie[_ -]?rod"),
        RegionRule("hydrogen_direct",    cell_name=r"HydrogenCoolant"),
        RegionRule("flow_divider",       cell_name=r"flow[_ -]?divid|divider"),
        RegionRule("cavity_end_walls",   cell_name=r"end[_ -]?wall"),
        RegionRule("propellant_liners",  cell_name=r"Liner\s+beryllium"),
        
        # moderator by name
        RegionRule("beo_moderator",      cell_name=r"BeO_mod"),
        RegionRule("end_moderators",     cell_name=r"end_BeO|end_graphite|top cap|nozzle_BeO"),
        RegionRule("graphite_moderator", cell_name=r"graphite moderator"),

        # everython else by material 
        RegionRule("fuel",               material_name=r"fuel|u-?233|uranium"),
        RegionRule("buffer_gas",         material_name=r"^Ne$|neon"),
        RegionRule("transparent_wall",   material_name=r"sio2|silica|quartz|transparent"),
        RegionRule("hydrogen_direct",    material_name=r"hydrogen|\bh2\b|propellant|tungsten|seed"),
        RegionRule("beo_moderator",      material_name=r"\bBeO\b"),
        RegionRule("graphite_moderator", material_name=r"graphite|carbon"),
        RegionRule("pressure_shell"    , material_name=r'fibreglass'),

    ]


def _cell_centroid_z(cell: openmc.Cell) -> Optional[float]:
    try:
        bb = cell.bounding_box
        lo, hi = float(bb.lower_left[2]), float(bb.upper_right[2])
    except Exception:
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return 0.5 * (lo + hi)


def _material_fills(cell: openmc.Cell) -> List[openmc.Material]:
    """Materials filling a cell (handles distributed-material fills)."""
    fill = cell.fill
    if isinstance(fill, openmc.Material):
        return [fill]
    if isinstance(fill, (list, tuple)):
        return [m for m in fill if isinstance(m, openmc.Material)]
    return []


def classify_cells(
    geometry: openmc.Geometry,
    rules: Optional[Sequence[RegionRule]] = None,
    z_active: Tuple[float, float] = Z_ACTIVE_DEFAULT,
) -> Tuple[Dict[int, str], List[int]]:
    """Map every material-filled cell id to a region name.

    Returns ``(cell_id -> region, unassigned_cell_ids)``.  Cells filled with a
    universe or lattice are skipped: only leaf cells can deposit heat, and
    including the parents would double count.
    """
    rules = list(rules) if rules is not None else default_rules(z_active)
    mapping: Dict[int, str] = {}
    unassigned: List[int] = []

    for cid, cell in geometry.get_all_cells().items():
        mats = _material_fills(cell)
        if not mats:
            continue  # void, or filled with a universe/lattice
        cname = cell.name or ""
        mname = " ".join(m.name or "" for m in mats)
        zc = _cell_centroid_z(cell)
        for rule in rules:
            if rule.matches(cname, mname, zc):
                mapping[cid] = rule.region
                break
        else:
            mapping[cid] = "unassigned"
            unassigned.append(cid)

    return mapping, unassigned


def dump_cell_inventory(
    geometry: openmc.Geometry,
    path: str = "cell_inventory.csv",
    rules: Optional[Sequence[RegionRule]] = None,
    z_active: Tuple[float, float] = Z_ACTIVE_DEFAULT,
) -> Dict[int, str]:
    """Write every material-filled cell with its auto-assigned region.

    Run this once, open the CSV, and check the ``region`` column.  Anything
    marked ``unassigned`` will silently drop out of the comparison table, so it
    must be zero rows before you quote a result.
    """
    mapping, unassigned = classify_cells(geometry, rules, z_active)
    cells = geometry.get_all_cells()

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_id", "cell_name", "material_ids", "material_names",
                    "z_centroid_cm", "region"])
        for cid, region in sorted(mapping.items()):
            cell = cells[cid]
            mats = _material_fills(cell)
            w.writerow([
                cid,
                cell.name or "",
                "|".join(str(m.id) for m in mats),
                "|".join(m.name or "" for m in mats),
                "" if _cell_centroid_z(cell) is None else f"{_cell_centroid_z(cell):.3f}",
                region,
            ])

    print(f"[heating] cell inventory -> {path}  "
          f"({len(mapping)} cells, {len(unassigned)} unassigned)")
    if unassigned:
        print("[heating] WARNING: unassigned cell ids:", unassigned[:40],
              "..." if len(unassigned) > 40 else "")
    return mapping


def write_region_map(mapping: Dict[int, str], path: str = "region_map.json") -> None:
    """Freeze a classification to JSON so it can be hand-corrected once and
    reused for every subsequent run."""
    with open(path, "w") as fh:
        json.dump({str(k): v for k, v in mapping.items()}, fh, indent=2, sort_keys=True)
    print(f"[heating] region map -> {path}")


def load_region_map(path: str = "region_map.json") -> Dict[int, str]:
    with open(path) as fh:
        return {int(k): v for k, v in json.load(fh).items()}


# =============================================================================
# Tally construction
# =============================================================================

# Photon transport is NOT configured here.
#
# It used to be: this module carried an ``enable_photon_transport`` helper that
# poked at ``settings``, and a ``patch_cross_sections_for_photons`` helper that
# rewrote the run's cross_sections.xml after the fact to add ``type="photon"``
# library entries.  Both are gone.  ``GCRConfig`` now owns the switch --
#
#     GCRConfig(photon_transport=True, photon_cutoff_ev=1.0e3,
#               photon_cross_sections_dir='libraries_xs/photon_hdf5')
#
# -- and the package acts on it in the two places that were being patched:
# ``GCR._build_settings`` sets photon_transport / electron_treatment / cutoff,
# and ``build_cross_section_library`` registers one photon HDF5 file per
# element derived from REQUIRED_NUCLIDES.  One flag, one source of truth, and
# no window in which the exported XML disagrees with the model in memory.


def add_heating_tallies(
    core,
    *,
    mode: str = "coupled",
    region_map: Optional[Dict[int, str]] = None,
    add_damage: bool = True,
    add_normalisation: bool = True,
    mesh: Optional[openmc.RegularMesh] = None,
) -> Dict[int, str]:
    """Register the heating tallies on a built ``GCR`` core.

    Parameters
    ----------
    core
        A ``GCR`` instance AFTER ``build()``.  Tallies go through
        ``core.register_tally`` so they land in the same registry as every
        other tally and are written by the single ``core.export()`` call.
    mode
        ``'local'``   -> score ``heating-local``, photon transport OFF.
        ``'coupled'`` -> score ``heating`` split by neutron/photon,
        photon transport ON.  Must agree with ``config.photon_transport``;
        this function checks and refuses if it does not.
    region_map
        ``cell_id -> region``.  If omitted it is generated with the default
        rules over ``(0, config.L)``.  Every material-filled cell in the map
        goes into the CellFilter, so the tally total is conserved and nothing
        can quietly vanish.
    mesh
        Optional mesh for a *spatial* deposition map.  Keep this coarse
        (60^3-100^3): unlike the fission-rate map, the deposition map is
        diffuse and does not need 600^3.

    Returns the region map actually used -- save it, post-processing needs it.
    """
    if mode not in ("local", "coupled"):
        raise ValueError(f"mode must be 'local' or 'coupled', got {mode!r}")

    cfg = core.config
    if (mode == "coupled") != bool(cfg.photon_transport):
        raise ValueError(
            f"mode='{mode}' but config.photon_transport={cfg.photon_transport}. "
            "In 'coupled' mode the 'heating' score needs transported photons; "
            "in 'local' mode 'heating-local' assumes there are none. "
            "Set photon_transport=(MODE == 'coupled') in the config."
        )

    if region_map is None:
        region_map, _ = classify_cells(core.geometry, z_active=(0.0, cfg.L))

    cell_ids = sorted(region_map.keys())

    # Every tally below declares its particles EXPLICITLY, including the mesh.
    # GCR.export() applies a neutron ParticleFilter to any tally that has none,
    # which is right for flux and reaction rates and exactly wrong for heating:
    # a silently neutron-only deposition map is the failure this prevents.
    particles = ["neutron", "photon", "electron", "positron"] if mode == "coupled" else ["neutron"]
    score = "heating" if mode == "coupled" else "heating-local"

    # -- main heating tally ---------------------------------------------------
    # Filter order matters for post-processing: the LAST filter varies fastest
    # in the flat bin array, so _aggregate_by_region reshapes to
    # (n_cells, n_particles).
    t = openmc.Tally(name=T_HEATING)
    t.filters = [openmc.CellFilter(cell_ids), openmc.ParticleFilter(particles)]
    t.scores = [score]
    #t.estimator = "tracklength"

    registered = [t]

    # -- displacement damage --------------------------------------------------
    # Neutron-only score whatever the mode.
    if add_damage:
        td = openmc.Tally(name=T_DAMAGE)
        td.filters = [openmc.CellFilter(cell_ids),
                      openmc.ParticleFilter(["neutron"])]
        td.scores = ["damage-energy"]
        registered.append(td)

    # -- normalisation --------------------------------------------------------
    # Whole-model fission-q-recoverable, one bin.  This is the same quantity
    # the standard power normalisation uses:
    #     S [src/s] = P [W] / (Q_tot [eV/src] * 1.602e-19)
    # Keeping it in the *same* statepoint removes any chance of normalising
    # against a run with different fuel loading.
    if add_normalisation:
        tn = openmc.Tally(name=T_NORM)
        tn.filters = [openmc.ParticleFilter(["neutron"])]
        tn.scores = ["fission-q-recoverable"]
        registered.append(tn)

    # -- optional spatial map -------------------------------------------------
    if mesh is not None:
        tm = openmc.Tally(name=T_HEATING_MESH)
        tm.filters = [openmc.MeshFilter(mesh),
                      openmc.ParticleFilter(particles)]
        tm.scores = [score]
       # tm.estimator = "tracklength"
        registered.append(tm)

    core.register_tally(*registered)

    print(f"[heating] mode='{mode}': {len(registered)} tallies registered, "
          f"heating over {len(cell_ids)} cells, particles={particles}")
    return region_map


def add_volume_calculation(
    core,
    region_map: Dict[int, str],
    samples: int = 10_000_000,
) -> None:
    """Stochastic volumes for the tallied cells.

    Needed only for power *density* (W/cm^3), dose rate (Mrad/s) and DPA.  The
    MW-per-region table does not require it.  Results land in ``volume_1.h5``.
    """
    cells = core.geometry.get_all_cells()
    domains = [cells[cid] for cid in sorted(region_map)]
    bb = core.geometry.bounding_box
    vc = openmc.VolumeCalculation(domains, samples, bb.lower_left, bb.upper_right)
    core.settings.volume_calculations = [vc]
    print(f"[heating] volume calculation queued: {len(domains)} cells, "
          f"{samples:,} samples")


# =============================================================================
# Post-processing
# =============================================================================

def _try_filter(tally: openmc.Tally, ftype):
    """``Tally.find_filter`` raises ValueError when absent; we want None."""
    try:
        return tally.find_filter(ftype)
    except (ValueError, KeyError):
        return None


def _get_tally(sp: openmc.StatePoint, name: str) -> Optional[openmc.Tally]:
    for t in sp.tallies.values():
        if t.name == name:
            return t
    return None


def source_rate(sp: openmc.StatePoint, power_W: float) -> Tuple[float, float]:
    """Return ``(S [source particles/s], Q_tot [eV per source particle])``.

    Identical in form to the normalisation already used in ``GCR.py``:

        P [W] = Q_tot [eV/src] * S [src/s] * 1.602176634e-19 [J/eV]
    """
    t = _get_tally(sp, T_NORM)
    if t is None:
        raise KeyError(f"normalisation tally '{T_NORM}' not present in statepoint")
    q_tot = float(np.sum(t.mean))
    if q_tot <= 0.0:
        raise ValueError("fission-q-recoverable tally is zero -- no fissions scored")
    s = power_W / (q_tot * EV_TO_J)
    return s, q_tot


def _cell_bins(tally: openmc.Tally) -> np.ndarray:
    cf = _try_filter(tally, openmc.CellFilter)
    if cf is None:
        raise KeyError('tally carries no CellFilter')
    return np.asarray(cf.bins).ravel()


def _aggregate_by_region(
    tally: openmc.Tally,
    region_map: Dict[int, str],
    n_particle_bins: int = 1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Sum tally mean and std_dev over the cells belonging to each region.

    Returns ``(mean[region] -> array(n_particle_bins),
               sigma[region] -> array(n_particle_bins))``.

    Uncertainties are combined in quadrature.  Bins from a single simulation
    are correlated, so this is the usual approximation, not a rigorous bound;
    it is nonetheless what everyone quotes and is conservative for the
    dominant (large-N) regions.
    """
    cells = _cell_bins(tally)
    n_cells = len(cells)

    mean = np.asarray(tally.mean).reshape(n_cells, n_particle_bins, -1)[:, :, 0]
    sig = np.asarray(tally.std_dev).reshape(n_cells, n_particle_bins, -1)[:, :, 0]

    out_m: Dict[str, np.ndarray] = {}
    out_s: Dict[str, np.ndarray] = {}
    for i, cid in enumerate(cells):
        region = region_map.get(int(cid), "unassigned")
        if region not in out_m:
            out_m[region] = np.zeros(n_particle_bins)
            out_s[region] = np.zeros(n_particle_bins)
        out_m[region] += mean[i]
        out_s[region] += sig[i] ** 2
    for region in out_s:
        out_s[region] = np.sqrt(out_s[region])
    return out_m, out_s


def heating_report(
    statepoint: str,
    power_W: float,
    region_map: Dict[int, str],
    mode: str = "coupled",
    csv_path: Optional[str] = "heating_report.csv",
    tex_path: Optional[str] = "heating_report.tex",
) -> List[dict]:
    """Reduce a statepoint to the MW-per-region table.

    Returns a list of row dicts with keys: region, neutron_MW, photon_MW,
    total_MW, sigma_MW, reference_MW, ratio, comparable, mechanism.
    """
    with openmc.StatePoint(statepoint) as sp:
        S, q_tot = source_rate(sp, power_W)

        t = _get_tally(sp, T_HEATING)
        if t is None:
            raise KeyError(f"'{T_HEATING}' not present in {statepoint}")

        pfilter = _try_filter(t, openmc.ParticleFilter)
        if pfilter is not None:
            particles = [str(b) for b in pfilter.bins]
        else:
            particles = ["neutron+photon(local)"]

        m, s = _aggregate_by_region(t, region_map, n_particle_bins=len(particles))

        # QA: prompt-transported energy versus recoverable fission energy
        total_heating_ev = sum(float(v.sum()) for v in m.values())
        frac_captured = total_heating_ev / q_tot

    scale = S * EV_TO_J * 1.0e-6  # eV/src -> MW

    rows: List[dict] = []
    ordered = [r for r in REFERENCE_MW if r in m] + \
              [r for r in sorted(m) if r not in REFERENCE_MW]

    for region in ordered:
        vals = m[region] * scale
        errs = s[region] * scale
        ref = REFERENCE_MW.get(region, {})
        neutron = float(vals[particles.index("neutron")]) if "neutron" in particles else float(vals.sum())
        _EM = ("photon", "electron", "positron")
        photon = (sum(float(vals[particles.index(p)]) for p in _EM if p in particles)
          if "photon" in particles else float("nan"))
        total = float(vals.sum())
        sigma = float(np.sqrt(np.sum(errs ** 2)))
        ref_mw = ref.get("mw")
        rows.append({
            "region": region,
            "neutron_MW": neutron,
            "photon_MW": photon,
            "total_MW": total,
            "sigma_MW": sigma,
            "reference_MW": ref_mw,
            "ratio": (total / ref_mw) if ref_mw else None,
            "comparable": bool(ref.get("comparable", False)),
            "mechanism": ref.get("mech", ""),
        })
    if not rows:
        raise ValueError(
                'no regions in the report: the region map is empty or none of its cell IDs appear in the CellFilter of the heating tally')
    comp_total = sum(r["total_MW"] for r in rows if r["comparable"])
    all_total = sum(r["total_MW"] for r in rows)

    # ------------------------------------------------------------------ print
    print()
    print(f"  Regional nuclear heating -- mode '{mode}'")
    print(f"  statepoint      : {statepoint}")
    print(f"  power           : {power_W/1e6:.1f} MW")
    print(f"  Q_recov total   : {q_tot:.4g} eV/source particle")
    print(f"  source rate S   : {S:.6g} particles/s")
    print()
    hdr = f"  {'region':<22}{'neutron':>10}{'photon':>10}{'total':>10}" \
          f"{'sigma':>9}{'L-910900-16':>13}{'C/E':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        ref = "" if r["reference_MW"] is None else f"{r['reference_MW']:.1f}"
        ratio = "" if r["ratio"] is None else f"{r['ratio']:.3f}"
        flag = "" if r["comparable"] or r["reference_MW"] is None else "  (*)"
        ph = "  --  " if np.isnan(r["photon_MW"]) else f"{r['photon_MW']:.2f}"
        print(f"  {r['region']:<22}{r['neutron_MW']:>10.2f}{ph:>10}"
              f"{r['total_MW']:>10.2f}{r['sigma_MW']:>9.2f}{ref:>13}{ratio:>8}{flag}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL (comparable)':<22}{'':>10}{'':>10}{comp_total:>10.2f}"
          f"{'':>9}{REFERENCE_COMPARABLE_TOTAL_MW:>13.1f}"
          f"{comp_total/REFERENCE_COMPARABLE_TOTAL_MW:>8.3f}")
    print(f"  {'TOTAL (all regions)':<22}{'':>10}{'':>10}{all_total:>10.2f}")
    print()
    print("  (*) mechanism is thermal-radiation dominated -- NOT a neutronics")
    print("      validation target; shown for context only.")
    print()
    print(f"  QA: sum(heating)/Q_recov = {frac_captured:.4f}")
    if mode == "coupled":
        print("      Expect ~0.90.  The shortfall is delayed beta + delayed gamma")
        print("      energy, which prompt transport does not carry.  If this is")
        print("      near 1.00 something is double counting; if below ~0.85 check")
        print("      that photon production data exists for the major nuclides.")
        print(f"      Total power unaccounted for: "
          f"{(1 - frac_captured) * power_W / 1e6:.1f} MW")
    else:
        print(" Expecte ~1.0. heating-local deposits secondary photon energy")
        print(" at the collision site, so nothing escapes the tally. Slightly")
        print(" above 1.0 is normal because KERMA included non-fission reactions")
        print(" mainly radiative capture, that fission-q-recoverable omits")
    print()

    # ------------------------------------------------------------------- files
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[heating] table -> {csv_path}")

    if tex_path:
        write_latex_table(rows, tex_path, mode=mode)

    return rows


def write_latex_table(rows: List[dict], path: str, mode: str = "coupled") -> None:
    """booktabs + siunitx table, in the style used elsewhere in the report."""
    pretty = {
        "pressure_shell": "Pressure shell",
        "nozzles": "Nozzles",
        "flow_divider": "Flow divider",
        "tie_rods": "Tie rods",
        "end_moderators": "Upper and lower end moderators",
        "beo_moderator": "Beryllium oxide moderator",
        "graphite_moderator": "Graphite moderator",
        "hydrogen_direct": "Direct hydrogen heating",
        "transparent_wall": "Transparent structure",
        "fuel": "Fuel region",
        "buffer_gas": "Neon buffer",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Steady-state neutron and gamma heat deposition rates at "
        r"\qty{4.6}{\giga\watt}, compared with \cite{L-910900-16} "
        r"Table~VIII. Statistical uncertainties are combined in quadrature "
        r"over cells.}",
        r"  \label{tab:heating_" + mode + "}",
        r"  \begin{tabular}{l S[table-format=3.2] S[table-format=3.2] "
        r"S[table-format=3.2] S[table-format=1.3]}",
        r"    \toprule",
        r"    Region & {Neutron, \unit{\mega\watt}} & {Gamma, \unit{\mega\watt}} "
        r"& {Reference, \unit{\mega\watt}} & {C/E} \\",
        r"    \midrule",
    ]
    for r in rows:
        if not r["comparable"]:
            continue
        name = pretty.get(r["region"], r["region"].replace("_", " ").capitalize())
        ph = "{--}" if np.isnan(r["photon_MW"]) else f"{r['photon_MW']:.2f}"
        ref = "{--}" if r["reference_MW"] is None else f"{r['reference_MW']:.2f}"
        ratio = "{--}" if r["ratio"] is None else f"{r['ratio']:.3f}"
        lines.append(f"    {name} & {r['neutron_MW']:.2f} & {ph} & {ref} & {ratio} \\\\")
    comp_total = sum(r["total_MW"] for r in rows if r["comparable"])
    lines += [
        r"    \midrule",
        f"    Total & \\multicolumn{{2}}{{c}}{{{comp_total:.2f}}} & "
        f"{REFERENCE_COMPARABLE_TOTAL_MW:.1f} & "
        f"{comp_total / REFERENCE_COMPARABLE_TOTAL_MW:.3f} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[heating] LaTeX table -> {path}")


# =============================================================================
# Derived quantities
# =============================================================================

def transparent_wall_dose_rate(
    statepoint: str,
    power_W: float,
    region_map: Dict[int, str],
    wall_mass_kg: float,
    region: str = "transparent_wall",
) -> float:
    """Ionising dose rate in the transparent wall, in Mrad/s.

    Compare against the \\qty{5}{\\mega\\rad\\per\\second} full-scale engine
    figure quoted in L-990929-3, which is the dose rate at which the
    fused-silica and single-crystal BeO irradiation results were interpreted.
    Reproducing it independently is a short calculation with a lot of leverage:
    it either validates or invalidates the entire transparent-wall lifetime
    argument inherited from UARL.

    ``wall_mass_kg`` = SiO2 density x tallied wall volume, summed over the
    seven cavities.  Take the volume from ``volume_1.h5`` if you ran
    :func:`add_volume_calculation`.
    """
    with openmc.StatePoint(statepoint) as sp:
        S, _ = source_rate(sp, power_W)
        t = _get_tally(sp, T_HEATING)
        pfilter = _try_filter(t, openmc.ParticleFilter)
        n_p = len(pfilter.bins) if pfilter is not None else 1
        m, _s = _aggregate_by_region(t, region_map, n_particle_bins=n_p)

    if region not in m:
        raise KeyError(f"region '{region}' absent from the tally")

    # Ionising dose: photon component only if it is resolved, else everything.
    if pfilter is not None and "photon" in [str(b) for b in pfilter.bins]:
        idx = [str(b) for b in pfilter.bins].index("photon")
        ev_per_src = float(m[region][idx])
        label = "photon"
    else:
        ev_per_src = float(m[region].sum())
        label = "neutron+photon (local)"

    watts = ev_per_src * S * EV_TO_J
    gy_per_s = watts / wall_mass_kg
    mrad_per_s = gy_per_s / GY_PER_MRAD

    print(f"[heating] transparent wall ionising dose rate ({label}): "
          f"{mrad_per_s:.3f} Mrad/s  "
          f"(L-990929-3 nominal {TRANSPARENT_WALL_DOSE_TARGET_MRAD_S:.1f} Mrad/s, "
          f"C/E = {mrad_per_s / TRANSPARENT_WALL_DOSE_TARGET_MRAD_S:.2f})")
    return mrad_per_s


def dpa_rate(
    statepoint: str,
    power_W: float,
    region_map: Dict[int, str],
    region: str,
    atoms_in_region: float,
    e_displacement_eV: float = 25.0,
    efficiency: float = 0.8,
) -> float:
    """NRT displacements per atom per second for one region.

        DPA/s = 0.8 * T_dam [eV/s] / (2 * E_d [eV]) / N_atoms

    ``e_displacement_eV``: 25 eV is the usual figure for Si and for Be in BeO;
    O in BeO is nearer 28 eV.  Quote the value you use.

    ``atoms_in_region`` = (atom density [atoms/b-cm] * 1e24) * volume [cm^3].
    """
    with openmc.StatePoint(statepoint) as sp:
        S, _ = source_rate(sp, power_W)
        t = _get_tally(sp, T_DAMAGE)
        if t is None:
            raise KeyError(f"'{T_DAMAGE}' not present -- rerun with add_damage=True")
        m, _s = _aggregate_by_region(t, region_map, n_particle_bins=1)

    t_dam = float(m[region].sum()) * S            # eV/s
    dpa_s = efficiency * t_dam / (2.0 * e_displacement_eV) / atoms_in_region
    print(f"[heating] {region}: {dpa_s:.3e} DPA/s  "
          f"({dpa_s * 1000:.3e} DPA per 1000 s burn, E_d = {e_displacement_eV:g} eV)")
    return dpa_s


# =============================================================================
# Plotting
# =============================================================================
def load_report_csv(path: str) -> List[dict]:
    """Read a heating_report.csv back into the row dicts the plotters expect.

    This is what makes the figures reproducible without a statepoint: the CSV
    is the frozen result, the plot is a view of it.
    """
    rows: List[dict] = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            r = dict(raw)
            for k in ("neutron_MW", "photon_MW", "total_MW", "sigma_MW"):
                r[k] = float(r[k]) if r.get(k) else float("nan")
            r["reference_MW"] = float(r["reference_MW"]) if r.get("reference_MW") else None
            r["ratio"] = float(r["ratio"]) if r.get("ratio") else None
            r["comparable"] = str(r.get("comparable", "")).strip().lower() == "true"
            rows.append(r)
    print(f"[heating] {len(rows)} rows <- {path}")
    return rows

def plot_heating_comparison(
    rows: List[dict],
    path: str = "heating_comparison.pdf",
    regions: Optional[Sequence[str]] = None,
) -> None:
    """Grouped bar chart: computed neutron/gamma split against Table VIII.

    ``regions`` chooses which rows are drawn and in what order; the default is
    ``PLOT_REGIONS``.  A region with no Table VIII entry gets its computed bar
    only, so a region can be shown without inventing a reference for it.
    References whose mechanism is not purely neutron plus gamma are hatched,
    because they are not validation targets.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wanted = list(regions) if regions is not None else PLOT_REGIONS
    by_region = {r["region"]: r for r in rows}
    absent = [k for k in wanted if k not in by_region]
    if absent:
        print(f"[heating] plot: no rows for {absent}, skipped")
    sel = [by_region[k] for k in wanted if k in by_region]
    if not sel:
        raise ValueError("plot_heating_comparison: none of the requested "
                         f"regions are present; rows have {sorted(by_region)}")

    labels = [PLOT_LABELS.get(r["region"], r["region"].replace("_", "\n"))
              for r in sel]
    neutron = np.array([r["neutron_MW"] for r in sel])
    photon = np.array([0.0 if np.isnan(r["photon_MW"]) else r["photon_MW"]
                       for r in sel])
    ref = np.array([np.nan if r["reference_MW"] is None else r["reference_MW"]
                    for r in sel], dtype=float)
    comparable = np.array([bool(r["comparable"]) for r in sel])

    x = np.arange(len(sel))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, neutron, w, label="This work: neutron", color="#3465a4")
    ax.bar(x - w / 2, photon, w, bottom=neutron, label="This work: gamma",
           color="#8ab4e8")

    clean = np.isfinite(ref) & comparable
    mixed = np.isfinite(ref) & ~comparable
    if clean.any():
        ax.bar(x[clean] + w / 2, ref[clean], w, color="#a0a0a0",
               label="L-910900-16 Table VIII")
    if mixed.any():
        ax.bar(x[mixed] + w / 2, ref[mixed], w, color="#d9d9d9",
               edgecolor="#909090", hatch="//",
               label="Table VIII, mixed mechanism")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Heat deposition rate (MW)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[heating] figure -> {path}")

def plot_local_vs_coupled(
    rows_local: List[dict],
    rows_coupled: List[dict],
    path: str = "gamma_redistribution.pdf",
) -> None:
    """The gamma-transport redistribution: coupled minus local, per region.

    Positive bars are regions that *gain* energy once gammas are allowed to
    travel; the fuel and buffer gas should show large negative bars, which is
    the whole argument for why ``fission-q-recoverable`` cannot be read as a
    deposition map in a gas core.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo = {r["region"]: r["total_MW"] for r in rows_local}
    co = {r["region"]: r["total_MW"] for r in rows_coupled}
    regions = [r for r in co if r in lo]
    delta = np.array([co[r] - lo[r] for r in regions])
    order = np.argsort(delta)
    regions = [regions[i] for i in order]
    delta = delta[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#c0392b" if d < 0 else "#27ae60" for d in delta]
    ax.barh([r.replace("_", " ") for r in regions], delta, color=colours)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Coupled $-$ local heating (MW)")
#    ax.set_title("Energy redistributed by gamma transport")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[heating] figure -> {path}")

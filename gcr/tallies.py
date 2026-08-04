"""Tally factories.

THE RULE OF THIS MODULE: every function here BUILDS tally objects and
RETURNS them.  Nothing here writes tallies.xml, keeps a list, or knows
what other tallies exist.  Registration and export are the model's job
(GCR.register_tally / GCR.export) and happen exactly once.

Contrast with the old pattern, where every add_*_tally method re-exported
tallies.xml from its own hand-written `hasattr` inventory of the other
methods' private attributes -- so the LAST call silently decided which
tallies survived, and adding a new tally method meant editing every
existing one.  That whole failure mode is structurally impossible here.

Names ('power_distribution', 'midplane_flux_groups', ...) and mesh
parameters are identical to the original, so plotting and existing
post-processing scripts keep working unchanged.
"""

from dataclasses import dataclass, field

import openmc
import numpy as np

from .config import GCRConfig


@dataclass
class TallyBundle:
    """A tally (or several) plus the context needed to plot it later.

    meta carries whatever the matching plot function needs: the mesh, the
    energy cutoffs, etc.  The model stores bundles in its registry keyed by
    the primary tally's name.
    """
    tallies: list
    mesh: openmc.RegularMesh = None
    meta: dict = field(default_factory=dict)

    @property
    def primary(self) -> openmc.Tally:
        return self.tallies[0]


# ---------------------------------------------------------------------------
# 3-D power
# ---------------------------------------------------------------------------

def power_tally(cfg: GCRConfig, nx: int = 600, ny: int = 600, nz: int = 600,
                z_min: float = None, z_max: float = None) -> TallyBundle:
    """3-D mesh tally of recoverable fission power over the whole reactor."""
    xy_extent = cfg.r_outlet * 4
    if z_min is None:
        z_min = -cfg.moderator_top_thickness - 12.0
    if z_max is None:
        z_max = cfg.L + cfg.L_conv + 10.0

    mesh = openmc.RegularMesh(name='power mesh')
    mesh.dimension = [nx, ny, nz]
    mesh.lower_left = [-xy_extent, -xy_extent, z_min]
    mesh.upper_right = [xy_extent, xy_extent, z_max]

    tally = openmc.Tally(name='power_distribution')
    tally.filters = [openmc.MeshFilter(mesh=mesh)]
    tally.scores = ['fission-q-recoverable']

    print(f'Power tally created: {nx}x{ny}x{nz} mesh over '
          f'[{-xy_extent:.1f}, {xy_extent:.1f}] cm x z=[{z_min:.1f}, {z_max:.1f}] cm')
    return TallyBundle(tallies=[tally], mesh=mesh)

 
def cavity_power_tally(cavities, name: str = 'cavity_power'):
    """Per-cavity, per-axial-layer, per-radial-zone fission power.
 
    Returns a TallyBundle whose meta carries the cell -> (cavity, layer,
    zone) map, so the statepoint can be unpacked without guesswork.
    """
    import openmc
    from .tallies import TallyBundle          # adjust to your import style
 
    cells, meta = [], {}
    for i, cav in enumerate(cavities):
        for cell in _fuel_cells_of(cav):
            fill = getattr(cell, 'fill', None)
            mname = getattr(fill, 'name', '') or ''
            zone = 'inner' if 'inner' in mname else (
                   'outer' if 'outer' in mname else '?')
            layer = -1
            if 'layer_' in mname:
                try:
                    layer = int(mname.rsplit('layer_', 1)[1])
                except ValueError:
                    pass
            cells.append(cell)
            meta[cell.id] = dict(cavity=i, layer=layer, zone=zone)
 
    if not cells:
        raise RuntimeError('no fuel cells found -- see _fuel_cells_of()')
 
    t = openmc.Tally(name=name)
    t.filters = [openmc.CellFilter(cells)]
    t.scores = ['fission-q-recoverable']
 
    print(f'Cavity power tally created: {len(cells)} fuel cells across '
          f'{len(cavities)} cavities')
    return TallyBundle(tallies=[t],
                       meta={'cells': meta,
                             'cell_ids': [c.id for c in cells]})
 
 
def _fuel_cells_of(cavity):
    """Fuel-filled cells of one cavity, whatever the Cavity dataclass calls
    them.  If none of these attribute names hit, expose one on Cavity --
    the alternative is scraping the geometry, which is worse."""
    for attr in ('fuel_cells', 'fuel_layer_cells', 'cells', 'all_cells'):
        got = getattr(cavity, attr, None)
        if not got:
            continue
        sel = [c for c in got
               if (getattr(getattr(c, 'fill', None), 'name', '') or '')
               .startswith('fuel')]
        if sel:
            return sel
    raise AttributeError(
        'Cavity exposes no list of fuel cells; add e.g. `fuel_cells` to the '
        'Cavity dataclass in geometry/cavity.py')
 
 
# And in model.py, next to the other thin wrappers:
#
#     def add_cavity_power_tally(self) -> None:
#         self.register_tally(tally_factories.cavity_power_tally(self.cavities))
 
 
def read_cavity_power(statepoint: str, meta: dict, power_W: float,
                      n_cavities: int = 7, n_layers: int = 10):
    """Unpack the cell tally into (cavity power, per-layer axial matrix).
 
    Returns
    -------
    P_cav      [n_cav]           MW, with
    P_cav_sig  [n_cav]           MW  1-sigma
    P_axial    [n_cav, n_layer]  MW
    F_cav, F_cav_sig             the cavity peaking factor and its uncertainty
    """
    import numpy as np
    import openmc
 
    with openmc.StatePoint(statepoint) as sp:
        t = next(x for x in sp.tallies.values() if x.name == 'cavity_power')
        df = t.get_pandas_dataframe()
 
    total = df['mean'].sum()
    f = power_W / total / 1e6                      # -> MW per tally unit
 
    P = np.zeros(n_cavities)
    V = np.zeros(n_cavities)                       # variance
    A = np.zeros((n_cavities, n_layers))
    for _, row in df.iterrows():
        m = meta['cells'][int(row['cell'])]
        P[m['cavity']] += row['mean'] * f
        V[m['cavity']] += (row['std. dev.'] * f) ** 2
        if 0 <= m['layer'] < n_layers:
            A[m['cavity'], m['layer']] += row['mean'] * f
 
    sig = np.sqrt(V)
    F_cav = P.max() / P.mean()
    i = int(np.argmax(P))
    # d(F)/F = dP_i/P_i (+) the mean's own error; the mean is far better
    # determined than any single cavity, so this is dominated by the first
    # term and is a good approximation.
    F_sig = F_cav * np.hypot(sig[i] / P[i],
                             np.sqrt((sig ** 2).sum()) / P.sum())
    return P, sig, A, F_cav, F_sig


# ---------------------------------------------------------------------------
# 3-D flux
# ---------------------------------------------------------------------------

def flux_tally(cfg: GCRConfig, nx: int = 200, ny: int = 200, nz: int = 200) -> TallyBundle:
    """3-D mesh tally of total neutron flux."""
    xy_extent = cfg.r_outlet * 4 

    mesh = openmc.RegularMesh(name='flux mesh')
    mesh.dimension = [nx, ny, nz]
    mesh.lower_left = [-xy_extent, -xy_extent, -cfg.moderator_top_thickness]
    mesh.upper_right = [xy_extent, xy_extent, cfg.L + cfg.L_conv + 20.0]

    tally = openmc.Tally(name='flux_distribution')
    tally.filters = [openmc.MeshFilter(mesh=mesh)]
    tally.scores = ['flux']

    print(f'Flux tally created: {nx}x{ny}x{nz} mesh')
    return TallyBundle(tallies=[tally], mesh=mesh)


# Fission spectrum tally

def _unique_materials(materials):
    """Yield each material once, de-duplicating by id().

    _create_layered_fuel_materials() aliases 'fuel_inner' / 'fuel_outer' to
    the layer-0 objects, so the same object appears twice in core.materials.
    """
    seen = set()
    for mat in materials.values():
        if id(mat) in seen:
            continue
        seen.add(id(mat))
        yield mat

def _is_fuel(mat):
    name = mat.name or ''
    return (name == 'fuel'
            or name.startswith('fuel_inner')
            or name.startswith('fuel_outer'))

def fission_spectrum_tally(materials) -> TallyBundle:
    """Energy-binned tally of flux and fission, restricted to fuel materials."""
    import numpy as np
    import os
    N_E_BINS = 500
    E_MIN = 1.0e-5  # eV
    E_MAX = 2.0e7  # eV  (20 MeV)
    energy_bins = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_E_BINS + 1)

    fuel_mats = [m for m in _unique_materials(materials) if _is_fuel(m)]
    if not fuel_mats:
        raise RuntimeError('No fuel materials found -- nothing to tally.')

    energy_filter   = openmc.EnergyFilter(energy_bins)
    material_filter = openmc.MaterialFilter(fuel_mats)

    tally = openmc.Tally(name='fuel_spectrum')
    tally.filters = [material_filter, energy_filter]
    tally.scores  = ['flux', 'fission']


    print(f'Spectrum tally added: {N_E_BINS} log bins from '
          f'{E_MIN:.1e} to {E_MAX:.1e} eV across {len(fuel_mats)} fuel materials.')
    return TallyBundle(tallies=[tally])


# ---------------------------------------------------------------------------
# IFP kinetics (beta_eff, Lambda_eff)
# ---------------------------------------------------------------------------

def kinetics_tallies(num_groups: int = 6) -> TallyBundle:
    """IFP tallies for adjoint-weighted kinetic parameters.

    Produces beta_eff (total and group-wise) and Lambda_eff via the
    Iterated Fission Probability method.  Requires
    settings.ifp_n_generation > 0 (set by the model's settings builder).

    Parameters
    ----------
    num_groups :
        Delayed-neutron precursor groups for the group-wise beta_eff.
        ENDF/B-VIII uses 6.  Pass None to tally only the total.
    """
    total_tally = openmc.Tally(name='ifp_kinetics_scores')
    total_tally.scores = [
        'ifp-time-numerator',
        'ifp-beta-numerator',
        'ifp-denominator',
    ]
    tallies = [total_tally]

    if num_groups is not None:
        group_beta = openmc.Tally(name='ifp_kinetics_beta_group')
        group_beta.scores = ['ifp-beta-numerator']
        group_beta.filters = [openmc.DelayedGroupFilter(list(range(1, num_groups + 1)))]
        tallies.append(group_beta)

    print('Kinetics (IFP) tallies created: total beta_eff, Lambda_eff'
          + (f' + {num_groups}-group beta_eff' if num_groups else ''))
    return TallyBundle(tallies=tallies)


# ---------------------------------------------------------------------------
# Midplane 3-group flux map
# ---------------------------------------------------------------------------
DEFAULT_ENERGY_BOUNDS = (0.0, 0.625, 1.125, 1.86, 1.0e5, 20e6)

def _energy_bounds(energy_bounds, thermal_cutoff, epithermal_cutoff):
    if energy_bounds is None:
        energy_bounds = (0.0, thermal_cutoff, epithermal_cutoff, 20e6)
    e = np.asarray(sorted(float(x) for x in energy_bounds), dtype=float)
    if len(e) < 2 or e[0] != 0.0:
        raise ValueError('energy bounds must start with 0.0 and hold at least '
                         f'two edges, got {e}')

    if np.any(np.diff(e) <= 0):
        raise ValueError(f'energy bounds must be strictlt increasing, got {e}')

    return e

def midplane_flux_tally(cfg: GCRConfig, nx: int = 600, ny: int = 600,
                        slice_thickness: float = 1.0,
                        thermal_cutoff: float = 0.625,
                        epithermal_cutoff: float = 1.0e5,
                        energy_bounds=None) -> TallyBundle:
    """2-D mesh tally at the fuel midplane (z = L/2), thermal/epithermal/fast.

    Energy groups:
        thermal    : E < thermal_cutoff            (default < 0.625 eV)
        epithermal : thermal_cutoff < E < epithermal_cutoff
        fast       : E > epithermal_cutoff         (default > 100 keV)
    """
    xy_extent = cfg.r_outlet * 5
    z_mid = cfg.L / 2
    half_dz = slice_thickness / 2

    mesh = openmc.RegularMesh(name='midplane flux mesh')
    mesh.dimension = [nx, ny, 1]
    mesh.lower_left = [-xy_extent, -xy_extent, z_mid - half_dz]
    mesh.upper_right = [xy_extent, xy_extent, z_mid + half_dz]

    edges = _energy_bounds(energy_bounds, thermal_cutoff, epithermal_cutoff)
    energy_filter = openmc.EnergyFilter(edges)

    tally = openmc.Tally(name='midplane_flux_groups')
    tally.filters = [openmc.MeshFilter(mesh=mesh), energy_filter]
    tally.scores = ['flux']

    print(f'Midplane three-group flux tally created: {nx}x{ny} mesh, '
          f'slab z=[{z_mid - half_dz:.2f}, {z_mid + half_dz:.2f}] cm, '
          f'cutoffs = {thermal_cutoff:g} eV / {epithermal_cutoff:g} eV')
    return TallyBundle(tallies=[tally], mesh=mesh,
                       meta={'thermal_cutoff': thermal_cutoff,
                             'epithermal_cutoff': epithermal_cutoff,
                             'energy_bounds': edges})


# ---------------------------------------------------------------------------
# Axial 3-group flux map
# ---------------------------------------------------------------------------

def axial_flux_tally(cfg: GCRConfig, ny: int = 600, nz: int = 600,
                     slice_thickness: float = 10.0,
                     z_min: float = None, z_max: float = None,
                     thermal_cutoff: float = 0.625,
                     epithermal_cutoff: float = 1.0e5,
                     energy_bounds=None) -> TallyBundle:
    """Thin x-slab mesh tally of the 3-group flux over the axial extent."""
    xy_extent = cfg.r_outlet * 5.0
    half_dx = slice_thickness / 2
    if z_min is None:
        z_min = 0.0
    if z_max is None:
        z_max = cfg.L

    mesh = openmc.RegularMesh(name='axial flux mesh')
    mesh.dimension = [1, ny, nz]
    mesh.lower_left = [-half_dx, -xy_extent, z_min]
    mesh.upper_right = [half_dx, xy_extent, z_max]

    edges = _energy_bounds(energy_bounds, thermal_cutoff, epithermal_cutoff)
    energy_filter = openmc.EnergyFilter(edges)

    tally = openmc.Tally(name='axial_flux_groups')
    tally.filters = [openmc.MeshFilter(mesh=mesh), energy_filter]
    tally.scores = ['flux']

    print(f'Axial three-group flux tally created: {ny}x{nz} mesh (yz frame), '
          f'slab x=[{-half_dx:.2f}, {half_dx:.2f}] cm, '
          f'z=[{z_min:.1f}, {z_max:.1f}] cm, '
          f'cutoffs = {thermal_cutoff:g} eV / {epithermal_cutoff:g} eV')
    return TallyBundle(tallies=[tally], mesh=mesh,
                       meta={'thermal_cutoff': thermal_cutoff,
                             'epithermal_cutoff': epithermal_cutoff,
                             'energy_bounds': edges})

def unweighted_lifetime_tally() -> TallyBundle:
    """Global tallies for the unweighted generation time / removal lifetime.

    In an eigenvalue run all tallies are normalised per source neutron, so
    with  I = <phi/v>,  F = <nu-fission>  (= k_eff),  A = <absorption>:

        Lambda_unweighted = I / F          (generation time)
        ell_unweighted    = I / (F/k - A)  (removal lifetime, 1/(v*Sigma_a + leak))

    and the two agree to within the leakage/absorption balance, i.e.
    ell = k * Lambda exactly when the balance closes.

    No filters -- these are whole-geometry integrals, which is what the
    balance requires.
    """
    tally = openmc.Tally(name='unweighted_lifetime')
    tally.scores = ['inverse-velocity', 'nu-fission', 'absorption']

    print('Unweighted lifetime tally created: inverse-velocity, '
          'nu-fission, absorption (global, unfiltered)')
    return TallyBundle(tallies=[tally])

def apply_particle_filter(tallies: openmc.Tallies) -> None:
    """
    Restrict the tallies to neutron-only, otherwise openmc throws a warning/error,
    and the tally collected does not reflect what you are looking for.

    inverse velocity was a hard error
    """

    neutron_filter = openmc.ParticleFilter(['neutron'])
    for tally in tallies:
        if not any(isinstance(f, openmc.ParticleFilter) for f in tally.filters):
            tally.filters = tally.filters + [neutron_filter]
            # particle filter is added to the tally only if I do not explicitly
            # define another particle filter beforehand

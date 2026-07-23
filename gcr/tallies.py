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
        z_min = -cfg.moderator_top_thickness - 10.0
    if z_max is None:
        z_max = cfg.L + cfg.L_conv + 30.0

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


# ---------------------------------------------------------------------------
# 3-D flux
# ---------------------------------------------------------------------------

def flux_tally(cfg: GCRConfig, nx: int = 200, ny: int = 200, nz: int = 200) -> TallyBundle:
    """3-D mesh tally of total neutron flux."""
    xy_extent = cfg.r_outlet * 2.5

    mesh = openmc.RegularMesh(name='flux mesh')
    mesh.dimension = [nx, ny, nz]
    mesh.lower_left = [-xy_extent, -xy_extent, -cfg.moderator_top_thickness]
    mesh.upper_right = [xy_extent, xy_extent, cfg.L + cfg.L_conv]

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

def midplane_flux_tally(cfg: GCRConfig, nx: int = 600, ny: int = 600,
                        slice_thickness: float = 1.0,
                        thermal_cutoff: float = 0.625,
                        epithermal_cutoff: float = 1.0e5) -> TallyBundle:
    """2-D mesh tally at the fuel midplane (z = L/2), thermal/epithermal/fast.

    Energy groups:
        thermal    : E < thermal_cutoff            (default < 0.625 eV)
        epithermal : thermal_cutoff < E < epithermal_cutoff
        fast       : E > epithermal_cutoff         (default > 100 keV)
    """
    xy_extent = cfg.r_outlet * 4
    z_mid = cfg.L / 2
    half_dz = slice_thickness / 2

    mesh = openmc.RegularMesh(name='midplane flux mesh')
    mesh.dimension = [nx, ny, 1]
    mesh.lower_left = [-xy_extent, -xy_extent, z_mid - half_dz]
    mesh.upper_right = [xy_extent, xy_extent, z_mid + half_dz]

    energy_filter = openmc.EnergyFilter([0.0, thermal_cutoff, epithermal_cutoff, 20.0e6])

    tally = openmc.Tally(name='midplane_flux_groups')
    tally.filters = [openmc.MeshFilter(mesh=mesh), energy_filter]
    tally.scores = ['flux']

    print(f'Midplane three-group flux tally created: {nx}x{ny} mesh, '
          f'slab z=[{z_mid - half_dz:.2f}, {z_mid + half_dz:.2f}] cm, '
          f'cutoffs = {thermal_cutoff:g} eV / {epithermal_cutoff:g} eV')
    return TallyBundle(tallies=[tally], mesh=mesh,
                       meta={'thermal_cutoff': thermal_cutoff,
                             'epithermal_cutoff': epithermal_cutoff})


# ---------------------------------------------------------------------------
# Axial 3-group flux map
# ---------------------------------------------------------------------------

def axial_flux_tally(cfg: GCRConfig, ny: int = 600, nz: int = 600,
                     slice_thickness: float = 10.0,
                     z_min: float = None, z_max: float = None,
                     thermal_cutoff: float = 0.625,
                     epithermal_cutoff: float = 1.0e5) -> TallyBundle:
    """Thin x-slab mesh tally of the 3-group flux over the axial extent."""
    xy_extent = cfg.r_outlet * 4.0
    half_dx = slice_thickness / 2
    if z_min is None:
        z_min = 0.0
    if z_max is None:
        z_max = cfg.L

    mesh = openmc.RegularMesh(name='axial flux mesh')
    mesh.dimension = [1, ny, nz]
    mesh.lower_left = [-half_dx, -xy_extent, z_min]
    mesh.upper_right = [half_dx, xy_extent, z_max]

    energy_filter = openmc.EnergyFilter([0.0, thermal_cutoff, epithermal_cutoff, 20.0e6])

    tally = openmc.Tally(name='axial_flux_groups')
    tally.filters = [openmc.MeshFilter(mesh=mesh), energy_filter]
    tally.scores = ['flux']

    print(f'Axial three-group flux tally created: {ny}x{nz} mesh (yz frame), '
          f'slab x=[{-half_dx:.2f}, {half_dx:.2f}] cm, '
          f'z=[{z_min:.1f}, {z_max:.1f}] cm, '
          f'cutoffs = {thermal_cutoff:g} eV / {epithermal_cutoff:g} eV')
    return TallyBundle(tallies=[tally], mesh=mesh,
                       meta={'thermal_cutoff': thermal_cutoff,
                             'epithermal_cutoff': epithermal_cutoff})

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

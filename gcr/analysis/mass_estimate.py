"""Analytic estimate of the initial U-233 inventory.

Pure post-material analysis: needs the materials dict and the config,
nothing from the geometry.  Ported from print_u233_mass_estimate.
"""

import numpy as np
import openmc
import openmc.data

from ..config import GCRConfig


def print_u233_mass_estimate(materials: dict, cfg: GCRConfig,
                             n_cavities: int = 7) -> None:
    """Print total fuel and U-233 mass for the current material densities.

    Assumes every fuel cell is the canonical inner-cylinder (r <= R0) +
    outer-annulus (R0 <= r <= R2) shape, replicated over n_cavities.
    Ignores top-cap injector/extraction pipes and inter-cavity hex-overlap
    trim (each a <1% correction).  Uses whatever densities the materials
    currently hold, so pre-scaling by fuel_density_alpha is picked up
    automatically -- call this AFTER GCR.build().
    """
    n_layers = cfg.n_axial_layers
    L_layer = cfg.L / n_layers   # cm

    V_inner = np.pi * cfg.R0 ** 2 * L_layer
    V_outer = np.pi * (cfg.R2 ** 2 - cfg.R0 ** 2) * L_layer

    def rho_u233(mat):
        nuc = dict(mat.get_nuclide_atom_densities())
        if 'U233' not in nuc:
            return 0.0
        A = {n: openmc.data.atomic_mass(n) for n in nuc}
        mass_per_atomcm = sum(ad * A[n] for n, ad in nuc.items())
        frac_U233 = nuc['U233'] * A['U233'] / mass_per_atomcm
        return mat.density * frac_U233   # g/cm3

    m_u233 = 0.0
    m_fuel = 0.0
    for k in range(n_layers):
        if n_layers > 1:
            inner = materials[f'fuel_inner_layer_{k}']
            outer = materials[f'fuel_outer_layer_{k}']
        else:
            inner = materials['fuel_inner']
            outer = materials['fuel_outer']

        m_u233 += n_cavities * (rho_u233(inner) * V_inner + rho_u233(outer) * V_outer)
        m_fuel += n_cavities * (inner.density * V_inner + outer.density * V_outer)

    bar = '=' * 60
    print('\n' + bar)
    print('  Initial fuel inventory (cylinder approximation)')
    print(bar)
    print(f'  cavities         : {n_cavities}')
    print(f'  axial layers     : {n_layers}')
    print(f'  total fuel mass  : {m_fuel:10.3f} g')
    print(f'  total U-233 mass : {m_u233:10.3f} g')
    print(bar + '\n')

"""All material definitions for the GCR model.

"""

import os
from dataclasses import dataclass, field

import numpy as np
import openmc
import openmc.data
from scipy.interpolate import interp1d

import gcnr

from .config import GCRConfig, REQUIRED_NUCLIDES, OPTIONAL_SAB_NUCLIDES, PRESSURE_VESSEL_NUCLIDES

# Physical constants used in the gas-density recipes
R_GAS = 8.31446261815324      # [J/(mol K)]
ATM_TO_PA = 101_325.0

# Molar masses [kg/mol]
M_U = 233.0e-3
M_F = 18.998e-3
M_Ne = 20.180e-3
M_Si = 28.086e-3

# oxide split for s-glass:
# SiO2 65, Al2O3 25, MgO 10 wt%
S901_GLASS = {'Si': 0.30383, 'O': 0.50355, 'Al': 0.13232, 'Mg': 0.06030}
# generic EPON 828 (C21H24O4)n
EPOXY = {'C': 0.7205, 'H': 0.0754, 'O': 0.1725, 'N': 0.0316}

def _fuel_gas_recipe(name: str, T_fuel: float, P_U_atm: float,
                     P_Ne_atm: float, P_Si_atm: float,
                     th_atom_fraction: float) -> openmc.Material:
    """One fuel material: real-gas uranium + ideal-gas Ne/Si buffer.

    The uranium mass density comes from the Ievlev real-gas EOS (gcnr);
    the buffer gases are dilute enough that the ideal gas law is fine.
    The heavy metal is split (1 - th) U-233 / th natural thorium by atoms.
    """
    U = gcnr.eos.UraniumEOS(method='ievlev')
    rho_U = float(U.rho(p=P_U_atm, T=T_fuel, p_unit='atm')) / 1000  # [g/cm3]

    # Number densities [mol/cm3]
    n_U = rho_U / (M_U * 1000)                                # M_U kg/mol -> g/mol
    n_Ne = (P_Ne_atm * ATM_TO_PA) / (R_GAS * T_fuel) * 1e-6   # m3 -> cm3
    n_Si = (P_Si_atm * ATM_TO_PA) / (R_GAS * T_fuel) * 1e-6

    # Total mass density [g/cm3]  (fluorine intentionally excluded, as before)
    rho_total = (n_U * M_U * 1000 +
                 n_Ne * M_Ne * 1000 +
                 n_Si * M_Si * 1000)

    mat = openmc.Material(name=name, temperature=T_fuel)
    mat.add_nuclide('U233', (1.0 - th_atom_fraction) * n_U, 'ao')
    mat.add_element('Th', th_atom_fraction * n_U, 'ao')
    mat.add_element('Ne', n_Ne, 'ao')
    mat.add_element('Si', n_Si, 'ao')
    mat.set_density('g/cm3', rho_total)
    return mat

def _seeded_hydrogen(name: str, T: float, rho_h_gcc: float,
                     f_seed: float) -> openmc.Material:
    """
    In the duct, the propellant (hydrogen) is seeded with tungsten to increase
    the opacity to the radiation of the normally transparent hydrogen.

    The f_seed is defined relative to the hydrogen, i.e. m_W/m_H = f_seed
    and thus m_prop = (1 + f_seed) * m_H

    As far as I remember (check config) the value provided is given for the inlet
    condition. However, here it is assumed it stays constant along the way for
    simplification.
    """
    material = openmc.Material(name=name, temperature=T)

    if f_seed > 0.0: # allow for runs where no tungsten
        material.add_element('H', 1.0 / (1.0 + f_seed), 'wo')
        material.add_element('W', f_seed / (1.0 + f_seed), 'wo')
        material.set_density('g/cm3', rho_h_gcc * (1.0 + f_seed))
    else:
        material.add_element('H', 1.0)
        material.set_density('g/cm3', rho_h_gcc)
    return material

def _fibreglass(cfg: GCRConfig) -> openmc.Material:
    """
    Filament-wound fibre-glass (S901) / epoxy composite from F-910093-37

    Density/resin fractions from config should be the values taken from this report.
    """
    w_resin = cfg.resin_mass_fraction
    comp = {}

    for element, w_frac in S901_GLASS.items():
        comp[element] = comp.get(element, 0.0) + (1.0 - w_resin) * w_frac
    for element, w_frac in EPOXY.items():
        comp[element] = comp.get(element, 0.0) + w_resin * w_frac
    
    material = openmc.Material(name='fibreglass', temperature=cfg.temperature_pv)
    for element, w_frac in comp.items():
        material.add_element(element, w_frac, 'wo')
    material.set_density('g/cm3', cfg.density_fibreglass)
    return material

def _pv_hydrogen(cfg: GCRConfig) -> openmc.Material:
    """
    H2 in the inter-shell annulus F-910093-37
    """
    material = openmc.Material(name='hydrogen_pv', temperature=cfg.pv_h2_temperature)
    material.add_element('H', 1.0)
    material.set_density('g/cm3', cfg.density_h2_pv)
    return material

def _photon_elements(nuclides) -> list:
    """
    Thus function just outputs unique elements symbol given nuclide names>
    It is done for photon data where particular isotopes don't matter.
    It is done like so, thus there is a sngle source of truth (the list)
    And I dont have to declare each time.

    Also god bless Claude for finding such function as ATOMIC_SYMBOL. What an amazing 
    feature otherwise hidden in the source file. It is Sunday, i am cooked.
    """

    sorted_tuple = sorted({
        openmc.data.ATOMIC_SYMBOL[openmc.data.zam(nuclide)[0]]
        for nuclide in nuclides
        })
    return sorted_tuple

def build_materials(cfg: GCRConfig) -> dict:
    """Define all base materials.  Purely composition + density: no file I/O.

    Returns a dict keyed by the same names the model has always used.
    """
    th = cfg.th_atom_fraction

    # --- Fuel: inner zone and outer buffer zone -----------------------------
    fuel_inner = _fuel_gas_recipe(
        'fuel_inner', cfg.fuel_inner_temperature,
        cfg.fuel_inner_P_U_atm, cfg.fuel_inner_P_Ne_atm, cfg.fuel_inner_P_Si_atm,
        th,
    )
    # In the buffer, Ne fills whatever the total pressure leaves over.
    P_Ne_outer = cfg.fuel_outer_P_total_atm - cfg.fuel_outer_P_U_atm - cfg.fuel_outer_P_Si_atm
    fuel_outer = _fuel_gas_recipe(
        'fuel_outer', cfg.fuel_outer_temperature,
        cfg.fuel_outer_P_U_atm, P_Ne_outer, cfg.fuel_outer_P_Si_atm,
        th,
    )

    # --- Moderators -----------------------------------------------------------
    BeO = openmc.Material(name='BeO', temperature=cfg.temperature_BeO)
    BeO.add_element('Be', 1.0)
    BeO.add_element('O', 1.0)
    BeO.set_density('g/cm3', cfg.density_BeO)

    graphite = openmc.Material(name='graphite', temperature=cfg.temperature_graphite)
    graphite.add_element('C', 1.0)
    graphite.set_density('g/cm3',cfg.density_graphite)

    # --- Propellant --------------------------------------------------------------

    rho_h2_avg = cfg.rho_h2_avg

    hydrogen = openmc.Material(name='hydrogen', temperature=cfg.temperature_h2_general)
    hydrogen.add_element('H', 1.0)
    hydrogen.set_density('g/cm3', rho_h2_avg)

    hydrogen_liner = openmc.Material(name='hydrogen_liner', temperature=cfg.temperature_h2_general)
    hydrogen_liner.add_element('H', 1.0)
    hydrogen_liner.set_density('g/cm3', rho_h2_avg)

    hydrogen_tori = openmc.Material(name='hydrogen_tori', temperature=cfg.temperature_h2_tori)
    hydrogen_tori.add_element('H', 1.0)
    hydrogen_tori.set_density('g/cm3', rho_h2_avg)

    # --- Buffer gas ------------------------------------------------------------------

    neon = openmc.Material(name='Ne', temperature=cfg.temperature_Ne)
    neon.add_element('Ne', 1.0)
    neon.set_density('g/cm3', cfg.rho_ne_avg)

    # --- Transparent wall: fused silica -------------------------------------------------
    SiO2 = openmc.Material(name='SiO2', temperature=cfg.temperature_Si02)
    SiO2.add_element('Si', 1.0)
    SiO2.add_element('O', 2.0)
    SiO2.set_density('g/cm3', cfg.density_SiO2)

    # --- Tie rods: beryllium ---------------------------------------------------------------
    beryllium = openmc.Material(name='Be', temperature=cfg.temperature_Be)
    beryllium.add_element('Be', 1.0)
    beryllium.set_density('g/cm3', cfg.density_beryllium)

    out = {
        'fuel_inner': fuel_inner,
        'fuel_outer': fuel_outer,
        'graphite': graphite,
        'hydrogen': hydrogen,
        'hydrogen_liner': hydrogen_liner,
        'hydrogen_tori': hydrogen_tori,
        'neon': neon,
        'SiO2': SiO2,
        'BeO': BeO,
        'Be': beryllium,
    }

    if cfg.include_pressure_vessel:
        out['fibreglass'] = _fibreglass(cfg)
        out['hydrogen_pv'] = _pv_hydrogen(cfg)

    return out

def fuel_temperature_from_h2(cfg: GCRConfig, T_H: float) -> float:
    """Radiation-equilibrium fuel temperature given a propellant temperature.

        T_fuel = 3 * ( Q_total * 0.8375 / (2 pi R L eps sigma) + T_H**4 ) ** (1/4)

    The average fuel temperature is 3x the effective radiating temperature.

    Same formula I used for creating XS (obviously), so maybe it would be smarter to make one function
    used by both scripts - one source of truth
    """
    from scipy.constants import pi, sigma
    R_m = cfg.R2_radiation / 100.0   # cm -> m
    L_m = cfg.L / 100.0
    flux_term = cfg.Q_total * 0.8375 / (2 * pi * R_m * L_m * cfg.epsilon_fuel * sigma)
    return 3 * (flux_term + T_H ** 4) ** 0.25


@dataclass
class LayeredMaterials:
    """The per-axial-layer materials, bundled so geometry code receives ONE
    explicit object instead of sniffing four private attributes.

    For n_axial_layers <= 1 every list simply contains the canonical
    material, so downstream code never needs a special case.
    """
    h2_layers: list = field(default_factory=list)
    h2_header: openmc.Material = None
    fuel_inner_layers: list = field(default_factory=list)
    fuel_outer_layers: list = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.h2_layers)


def build_layered_materials(cfg: GCRConfig, materials: dict) -> LayeredMaterials:
    """Create per-layer propellant and fuel materials from the H2 profile.

    Loads the .npz written by the propellant-channel script, interpolates
    rho(x) and T(x), and builds one material per axial layer evaluated at
    the layer midpoint.  Fuel layer temperatures follow from the
    radiation-equilibrium relation, the real-gas uranium density is
    re-queried from gcnr for each layer so density tracks temperature.

    """
    n = cfg.n_axial_layers

    if n <= 1:
        return LayeredMaterials(
            h2_layers=[materials['hydrogen']],
            h2_header=materials['hydrogen'],
            fuel_inner_layers=[materials['fuel_inner']],
            fuel_outer_layers=[materials['fuel_outer']],
        )

    # ~~~~ Loading the profile of the propellant flow ~~~~~~~~~
    profile = np.load(cfg.h2_density_profile_path)
    x_m = profile['x_m']
    rho_function = interp1d(x_m, profile['rho_kgm3'], kind='linear', bounds_error=True)
    T_function = interp1d(x_m, profile['T_K'], kind='linear', bounds_error=True)
    #print('exit temperature', T_function(cfg.L / 100.0))

    L_m = cfg.L / 100.0
    dz_m = L_m / n

    layered = LayeredMaterials()

    # ~~~~ Create layered materials (seeded) ~~~~~~~
    for k in range(n):
        z_mid_m = (k + 0.5) * dz_m
        rho_gcc = float(rho_function(z_mid_m)) / 1000   # given in SI unit kg/m3 in the .npz file
        T_val = float(T_function(z_mid_m))

        prop_mat = _seeded_hydrogen(name=f'hydrogen_layer{k}', T=T_val, rho_h_gcc=rho_gcc,
                                    f_seed = cfg.seed_mass_fraction)

        layered.h2_layers.append(prop_mat)
        materials[f'hydrogen_layer_{k}'] = prop_mat

    # Header region above the fuel zone: inlet conditions
    h2_header = _seeded_hydrogen(name='hydrogen_header', T=float(T_function(0.0)), rho_h_gcc=float(rho_function(0.0)) / 1_000,
                                 f_seed=cfg.seed_mass_fraction)
    layered.h2_header = h2_header
    materials['hydrogen_header'] = h2_header

    # --- Fuel layers (inner + outer at each layer's radiation temperature) -------
    P_Ne_outer = cfg.fuel_outer_P_total_atm - cfg.fuel_outer_P_U_atm - cfg.fuel_outer_P_Si_atm

    for k in range(n):
        z_mid_m = (k + 0.5) * dz_m
        T_H = float(T_function(z_mid_m))
        T_fuel = float(fuel_temperature_from_h2(cfg, T_H))

        inner = _fuel_gas_recipe(
            f'fuel_inner_layer_{k}', T_fuel,
            cfg.fuel_inner_P_U_atm, cfg.fuel_inner_P_Ne_atm, cfg.fuel_inner_P_Si_atm,
            cfg.th_atom_fraction,
        )
        outer = _fuel_gas_recipe(
            f'fuel_outer_layer_{k}', T_fuel,
            cfg.fuel_outer_P_U_atm, P_Ne_outer, cfg.fuel_outer_P_Si_atm,
            cfg.th_atom_fraction,
        )
        layered.fuel_inner_layers.append(inner)
        layered.fuel_outer_layers.append(outer)
        materials[f'fuel_inner_layer_{k}'] = inner
        materials[f'fuel_outer_layer_{k}'] = outer

    # Update the canonical 'fuel_inner'/'fuel_outer' entries IN PLACE.
    # Do NOT replace the original Material objects -- geometry cells (the
    # injector/extraction pipes) hold references to them by Python identity,
    # so swapping the dict entry would leave geometry.xml pointing at IDs
    # that materials.xml no longer exports ("Could not find material N").
    # Copying layer-0's composition into the original objects keeps both
    # XML files consistent.
    _copy_material(layered.fuel_inner_layers[0], materials['fuel_inner'])
    _copy_material(layered.fuel_outer_layers[0], materials['fuel_outer'])

    return layered


def _copy_material(src: openmc.Material, dst: openmc.Material) -> None:
    """Overwrite dst's nuclides, density and temperature from src (in place)."""
    dst._nuclides.clear()
    for nuc, density, typ in src._nuclides:
        dst.add_nuclide(nuc, density, typ)
    dst.set_density('g/cm3', src.density)
    dst.temperature = src.temperature


def apply_beo_sab(cfg: GCRConfig, materials: dict, t_max: float = 1200.0,
                  sab_tables: tuple = ('c_Be_in_BeO', 'c_O_in_BeO')) -> None:
    """Conditionally attach BeO S(a,b) thermal-scattering kernels.
    """
    if 'BeO' not in materials:
        return

    available = tuple(
        t for t in sab_tables
        if os.path.isfile(os.path.join(cfg.cross_sections_dir, f'{t}.h5'))
    )

    mat = materials['BeO']
    if available and mat.temperature <= t_max:
        attached = {name for name, _ in (mat._sab or [])}
        for table in available:
            if table not in attached:
                mat.add_s_alpha_beta(table)
    else:
        mat._sab = []

def apply_graphite_sab(cfg: GCRConfig, materials: dict, t_max: float = 3000.0,
                       sab_tables: tuple = ('c_Graphite', 'else')) -> None:
    """Conditionally attach graphite S(a,b) TSL.

    The maximum temperature of 3000K found in the JEFF-4.0 evaluation.

    this function is a copy of the apply_beo_sab function just above
    """
    
    print('started apply graphite')
    #print(materials)
    if 'graphite' not in materials:
        return
    print(os.path.join(cfg.cross_sections_dir, f'{sab_tables[0]}.h5'))
    available = tuple(
            t for t in sab_tables
            if os.path.isfile(os.path.join(cfg.cross_sections_dir, f'{t}.h5'))
            )
    print('inside apply graphite, t_max:', t_max, 't:', materials['graphite'].temperature)
    print(bool(available))
    mat = materials['graphite']
    if available and mat.temperature <= t_max:
        attached = {name for name, _ in (mat._sab or [])}
        for table in available:
            if table not in attached:
                mat.add_s_alpha_beta(table)
    elif available and mat.temperature <= t_max:
        print("Warning: S(a,b) for graphite not attached. File exists"
              f"but the graphite temperature {mat.temperature}K exceeds the maximum temperature {t_max}K")
    else:
        mat._sab = []

def apply_fuel_density_alpha(materials: dict, alpha: float) -> None:
    """Scale every fuel material's density by `alpha` (once).

    It is called by GCR.build(), AFTER layered fuel materials exist, so the scaling covers
    canonical and per-layer fuels alike.  Deduplication by object identity
    guards against the same Material appearing under several dict keys.
    """
    fuel_prefixes = ('fuel_inner', 'fuel_outer')
    seen = set()
    for mat in materials.values():
        if id(mat) in seen:
            continue
        seen.add(id(mat))
        #print(mat)
        if mat.name == 'fuel' or mat.name.startswith(fuel_prefixes):
            mat.set_density('g/cm3', mat.density * alpha)


def build_cross_section_library(cfg: GCRConfig, output_dir: str) -> str:
    """Register all required nuclide HDF5 files and export cross_sections.xml.

    Written into `output_dir` so concurrent runs in different
    directories don't overwrite each other's copy.

    Returns
    -------
    str
        Absolute path of the exported cross_sections.xml.

    Raises
    ------
    FileNotFoundError
        If any required nuclide file is missing from cross_sections_dir.
    """
    library = openmc.data.DataLibrary()

    for nuclide in REQUIRED_NUCLIDES:
        path = os.path.join(cfg.cross_sections_dir, f'{nuclide}.h5')
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'Cross-section file not found: {path}\n'
                f"Check that 'cross_sections_dir' points to the correct directory."
            )
        library.register_file(path)

    for nuclide in OPTIONAL_SAB_NUCLIDES:
        path = os.path.join(cfg.cross_sections_dir, f'{nuclide}.h5')
        if os.path.isfile(path):
            print('Registered S(a,b)', nuclide)
            library.register_file(path)
        else:
            print(f"  S(a,b) table '{nuclide}' absent in "
                  f'{cfg.cross_sections_dir}; BeO runs free-gas.')

    if cfg.photon_transport:
        photon_nuclides = list(REQUIRED_NUCLIDES)
        if cfg.include_pressure_vessel:
            photon_nuclides += PRESSURE_VESSEL_NUCLIDES

        photon_dir = cfg.photon_cross_sections_dir or cfg.cross_sections_dir

        for element in _photon_elements(photon_nuclides):
            photon_h5_file = os.path.join(photon_dir, f'{element}_photon.h5')
            if not os.path.isfile(photon_h5_file):
                raise FileNotFoundError(
                    f'Photon cross-section file not found: {photon_h5_file}\n'
                    'Check the file or turn photon_transport off by setting = False'
                )
            library.register_file(photon_h5_file)       

    if cfg.include_pressure_vessel:
        for nuclide in PRESSURE_VESSEL_NUCLIDES:
            path = os.path.join(cfg.cross_sections_dir, f'{nuclide}.h5')
            if os.path.isfile(path):
                library.register_file(path)
            else:
                raise FileNotFoundError(
                        f'The cross-section file for pressure shell nuclide not found: {nuclide}.h5'
                        f'\ncheck the cross section dir: {cfg.cross_sections_dir}'
                        )
    xs_xml_path = os.path.abspath(os.path.join(output_dir, 'cross_sections.xml'))
    library.export_to_xml(xs_xml_path)
    return xs_xml_path

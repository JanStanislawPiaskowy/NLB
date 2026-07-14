"""Configuration for the GCR model.

This module is the SINGLE home for:
  * unit-conversion constants,
  * the list of nuclides the model needs,
  * GCRConfig — every physical, geometric and run parameter.

Nothing in here imports OpenMC or builds anything.  That is deliberate:
you can open, inspect, serialise or sweep a configuration on a laptop
that has no nuclear-data libraries installed at all.

Design rule for the whole package
---------------------------------
Every other module imports *from* config; config imports from nobody
(except the standard library and numpy).  If you ever feel the urge to
import geometry or materials in here, something is upside down.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
ft_to_cm = 30.48   # [cm/ft]
in_to_cm = 2.54    # [cm/in]

# ---------------------------------------------------------------------------
# Nuclide files required by the model (checked at export time)
# ---------------------------------------------------------------------------
REQUIRED_NUCLIDES = [
    'Be9', 'C12', 'C13', 'F19',
    'H1', 'H2',
    'Ne20', 'Ne21', 'Ne22',
    'O16', 'O17', 'O18',
    'Si28', 'Si29', 'Si30',
    'U233', 'Th230', 'Th232',
]

OPTIONAL_SAB_NUCLIDES = [
    'c_Be_in_BeO',
    'c_O_in_BeO',
]


@dataclass
class GCRConfig:
    """All physical and geometric parameters that define a simulation.

    All lengths are in centimetres unless stated otherwise.
    References are to original UARL drawing/report numbers.
    """

    # --- Paths -------------------------------------------------------------
    cross_sections_dir: str = '../CrossSections/endfb_viii.1-hdf5'

    # --- Fuel region geometry (L-910905-16) ---------------------------------
    R0: float = 11.0     # [cm] radius within which fuel partial pressure is constant
    R1: float = 19.35    # [cm] inner radial boundary of uranium–neon
    R2: float = 20.75    # [cm] edge-of-fuel location, L-910905-16
    R2_radiation: float = 20.75  # [cm], I had to add this one because when doing sensitivity
    # study on fuel radius I wanted to keep the fuel mass constant but even if you scale
    # the density appropriately the change of temp. due to change in radiation changes
    # the density yet again
    R3: float = 24.45    # [cm] inner radial boundary of transparent wall
    L:  float = 6.0 * ft_to_cm   # [cm] length of the fuel region

    # ~~~~~~~~~~~ Fuel Composition Variables ~~~~~~
    th_atom_fraction: float = 0.05        # [-] Th share of heavy-metal atoms
    fuel_inner_temperature: float = 25_000.0   # [K]
    fuel_inner_P_U_atm:  float = 195.0    # [atm] uranium partial pressure
    fuel_inner_P_Ne_atm: float = 295.0    # [atm]
    fuel_inner_P_Si_atm: float = 10.0     # [atm]
    ## inner/outer partial pressures need to add up to 500atm
    fuel_outer_temperature: float = 20_000.0   # [K]
    fuel_outer_P_U_atm:  float = 195.0 / 2    # [atm]
    fuel_outer_P_Si_atm: float = 10.0     # [atm]
    fuel_outer_P_total_atm: float = 500.0 # [atm]  Ne fills the remainder

    # ~~~~~~~~~ Temperatures of Structures ~~~~~~~~~~~~
    temperature_BeO: float = 1698.333  # [K], F-910093-37 p.55
    temperature_graphite: float = 2392.0  # [K], F-910093-37
    temperature_h2_general: float = 4000.0  # [K]
    temperature_Ne: float = 4722.222  # [K], avg 2000/15000 R, p.72
    temperature_Si02: float = 1698.333 # [K].
    temperature_Be: float = 555.56 # [K]

    # ~~~~~~~~~~ Densities of Structures ~~~~~~~~~~~~
    density_BeO: float = 3.019480346 # [g/cm^3], F-910093-37 p.55
    density_graphite: float = 1.8392399646  # [g/cm^3] p.40 G-910375
    density_SiO2: float = 2.5197042887  # [g/cm3]
    density_beryllium: float = 1.8392399646  # [g/cm3]

    rho_h2_in: float = 0.0053822037    # [g/cm3] F-910093-37, p.72
    rho_h2_out: float = 0.0012109958   # [g/cm3] F-910093-37, p.72

    rho_ne_fuel: float = 0.0148010602        # [g/cm3]
    rho_ne_periphery: float = 0.1110079512   # [g/cm3]

    # --- Moderator -----------------------------------------------------------
    moderator_top_thickness: float = 27.0  # [cm] visual estimation, F-910093-37
    moderator_cone_inner_radius: float = 120.0  # [cm] R_in of the graphite cone

    # --- Tie rods (F-910093-37, tab. VII) -------------------------------------
    diameter_tierod_inner: float = 1.0 * in_to_cm    # [cm] coolant channel
    diameter_tierod_outer: float = 1.358 * in_to_cm  # [cm] outer beryllium diameter
    thickness_tierod_graphite: float = 0.3 * in_to_cm  # [cm] graphite coating
    # Ridge-merging tolerances for the tie-rod builder.  Adjacent *peripheral*
    # hexagons cannot tile exactly (the residual is ~ tilt² · hl ≈ 3 mm), so
    # near-coincident ridge lines within these tolerances are treated as one rod.
    tierod_merge_distance: float = 0.5    # [cm]  perpendicular line-to-line distance
    tierod_merge_angle_deg: float = 0.5   # [deg] angle between ridge directions

    # --- Hydrogen channel (F-910093-37 p.54 tab. V) ----------------------------
    r_inlet:  float = 0.911 * ft_to_cm   # [cm] liner inside radius at inlet
    r_outlet: float = 1.320 * ft_to_cm   # [cm] liner inside radius at outlet

    # --- Hexagonal unit cell ----------------------------------------------------
    # Side length = hex_side_to_radius * r_inlet ("measured by hand, F-910093-37").
    hex_side_to_radius: float = 1.4

    # --- Transparent wall tubes (H-910375 p.55 tab. VIII) -----------------------
    r_tw:   float = 0.005 * in_to_cm      # [cm] tube wall thickness
    r_t_id: float = 0.05 * in_to_cm / 2   # [cm] tube inner radius

    # --- Nozzle -------------------------------------------------------------------
    L_conv: float = 48.0                  # [cm] convergent nozzle length
    nozzle_throat_radius: float = 3.67    # [cm]

    n_liner_tubes: int = 72

    # --- Axial layering of the propellant channel -----------------------------------
    n_axial_layers: int = 10              # 1 = legacy single-material propellant
    h2_density_profile_path: str = 'settings/h2_density_profile.npz'

    # --- Fuel radiation-equilibrium model (used when n_axial_layers > 1) -------------
    # Per-layer fuel temperature is derived from the hydrogen temperature via
    #     T_fuel = ( Q_total / (2π R L ε σ) + T_H**4 ) ** (1/4)
    # (n cancels between Q_ax and A_ax, so the formula is per-cavity).
    Q_total: float = 4.6e9 / 7            # [W] cavity power in the radiation balance
    epsilon_fuel: float = 0.85            # [-] emissivity at the fuel–wall interface

    # --- Density scaling ---------------------------------------------------------------
    fuel_density_alpha: float = 2.0540    # [-] multiplier applied to all fuel densities

    # --- Monte Carlo run parameters ------------------------------------------------------
    batches:   int = 250
    inactive:  int = 25
    particles: int = 600_000
    temperature_tolerance: float = 300.0  # [K] settings.temperature interpolation window
    seed: int | None = None               # None = OpenMC default; set for bit-reproducible runs

    # --- Derived values ---------------------
    d_tw_outer: float = field(init=False)
    tilt:       float = field(init=False)
    rho_h2_avg: float = field(init=False)
    rho_ne_avg: float = field(init=False)

    def __post_init__(self):
        """Compute derived geometric values after the dataclass is initialised."""
        self.d_tw_outer = 2 * self.r_tw + 2 * self.r_t_id
        self.tilt = float(-np.arctan((self.r_outlet - self.r_inlet) / self.L))
        self.rho_h2_avg = (self.rho_h2_in + self.rho_h2_out ) / 2
        self.rho_ne_avg = (self.rho_ne_fuel + self.rho_ne_periphery) / 2

    # Extra

    @property
    def hex_side_length(self) -> float:
        """Hexagon side length [cm]."""
        return self.hex_side_to_radius * self.r_inlet

    # ~~~~~~ json reader and maker

    def to_json(self, path: str) -> None:
        """Save this configuration to a JSON file.

        For reproducibility/traceability

        Example
        -------
        >>> config = GCRConfig()
        >>> config.to_json('settings/run_001_config.json')
        """
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> 'GCRConfig':
        """Load a configuration from a previously saved JSON file.

        Derived fields (d_tw_outer, tilt) are recomputed on load.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        data.pop('d_tw_outer', None)
        data.pop('tilt', None)
        return cls(**data)

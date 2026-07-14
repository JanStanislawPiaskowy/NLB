"""The GCR orchestrator.

This is the "master file" of the package.  It owns three things and ONLY
three things:

  1.  THE BUILD ORDER.  ``GCR.build()`` is the one place in the whole code
      base where the sequence materials -> cavities -> sphere -> (tie rods)
      -> moderator -> end moderator -> nozzle -> overlap resolution ->
      settings exists.  The old code documented this order in docstrings
      and hoped; here it is enforced, because there is nowhere else the
      order could even be written down.

  2.  THE STATE.  Every attribute the model accumulates (materials,
      cavities, cells, tally registry, ...) is declared in ``__init__``.
      No ``hasattr`` archaeology: if you want to know what a GCR object
      can hold, read one method.

  3.  THE TALLY REGISTRY.  ``register_tally`` appends; ``export`` writes
      tallies.xml exactly once, from the full registry, immediately
      before the run.  Order of add_* calls no longer matters and nothing
      is silently dropped.

Everything with actual physics or geometry in it lives in the other
modules and is called from here as a plain function.

Typical use
-----------
    from gcr import GCRConfig, GCR

    config = GCRConfig(cross_sections_dir='libraries_xs/jeff40_hdf5')
    core = GCR(config)
    core.build()                      # returns an openmc.Model

    core.add_power_tally()
    core.add_kinetics_tally(num_groups=6)

    core.run()                        # exports XML once, runs OpenMC
"""

import os
import warnings

import numpy as np
import openmc
import openmc.stats

from .config import GCRConfig
from .materials import (LayeredMaterials, apply_beo_sab, apply_fuel_density_alpha,
                        build_cross_section_library, build_layered_materials,
                        build_materials)
from .geometry.hexmaths import cavity_placements
from .geometry.cavity import build_cavity
from .geometry.tie_rods import build_tie_rods
from .geometry.moderator import (build_end_moderator, build_moderator,
                                 build_nozzle_end, create_bounding_sphere)
from .geometry.overlaps import resolve_cavity_overlaps
from . import tallies as tally_factories
from .tallies import TallyBundle
from . import plotting


class GCR:
    """Seven-cavity NLB gas-core reactor model.

    Parameters
    ----------
    config :
        A GCRConfig with all physical/geometric/run parameters.
        Defaults to GCRConfig() if omitted.
    include_tie_rods :
        Build tie rods along the hexagon ridges and carve them from the
        slots and the moderator.  OFF by default -- enable deliberately
        and validate with ``run(dry_run=True)`` before production use.
    output_dir :
        Where all OpenMC XML inputs and the statepoint land.
    """

    def __init__(self, config: GCRConfig = None,
                 include_tie_rods: bool = False,
                 output_dir: str = 'settings'):
        self.config = config or GCRConfig()
        self.include_tie_rods = include_tie_rods
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # ---- All state, declared up front (nothing appears later) --------
        self.materials: dict = {}                 # name -> openmc.Material
        self.layered: LayeredMaterials = None
        self.cavities: list = []                  # [Cavity, ...]
        self.cells: list = []                     # root-universe cells
        self.tie_rods = None                      # TieRods | None
        self.bounding_sphere = None               # openmc.Sphere
        self.settings: openmc.Settings = None
        self.geometry: openmc.Geometry = None
        self.model: openmc.Model = None           # assembled by build()
        self._registry: dict = {}                 # tally name -> TallyBundle

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self) -> openmc.Model:
        """Build materials, geometry and settings; assemble the openmc.Model.

        This method IS the build order.  Read it top to bottom and you have
        read the whole recipe of the reactor.
        """
        cfg = self.config

        # 1) Materials -- base set, axial layers, S(a,b), density scaling.
        #    apply_fuel_density_alpha runs AFTER the layered fuels exist so
        #    the scaling covers every fuel material exactly once.
        self.materials = build_materials(cfg)
        self.layered = build_layered_materials(cfg, self.materials)
        apply_beo_sab(cfg, self.materials)
        apply_fuel_density_alpha(self.materials, cfg.fuel_density_alpha)

        # 2) The seven cavities, placed by the ONE placement function.
        for placement in cavity_placements(cfg):
            cavity = build_cavity(cfg, self.materials, self.layered, placement)
            self.cavities.append(cavity)
            self.cells.append(cavity.slot_cell)

        # 3) Vacuum bounding sphere (needed by the moderator builder).
        self.bounding_sphere = create_bounding_sphere(cfg)

        # 4) Tie rods -- optional, OFF by default.  Built BEFORE the
        #    moderator so the moderator can carve around them (bounded).
        if self.include_tie_rods:
            self.tie_rods = build_tie_rods(cfg, self.materials, self.cavities)
            self.cells.extend(self.tie_rods.cells)

        # 5) Everything outside the slots.
        self.cells += build_moderator(cfg, self.materials, self.cavities,
                                      self.bounding_sphere, tie_rods=self.tie_rods)
        self.cells += build_end_moderator(cfg, self.materials, self.cavities)
        self.cells += build_nozzle_end(cfg, self.materials, self.cavities,
                                       layered=self.layered,
                                       tie_rods=self.tie_rods)

        # 6) The tapered hexagons cannot tile exactly (~3 mm residual);
        #    give every neighbouring pair a shared midplane.
        resolve_cavity_overlaps(cfg, self.cavities)

        # 7) Settings and final assembly.
        self.settings = self._build_settings()
        root = openmc.Universe(cells=self.cells)
        self.geometry = openmc.Geometry(root)

        # Deduplicate materials by object identity (order-preserving); the
        # same Material may legitimately sit under several dict keys.
        seen, unique_mats = set(), []
        for mat in self.materials.values():
            if id(mat) not in seen:
                seen.add(id(mat))
                unique_mats.append(mat)

        self.model = openmc.Model(
            geometry=self.geometry,
            materials=openmc.Materials(unique_mats),
            settings=self.settings,
        )
        return self.model

    def _build_settings(self) -> openmc.Settings:
        """Run settings + one Watt fission source box per cavity."""
        cfg = self.config

        settings = openmc.Settings()
        settings.batches = cfg.batches
        settings.inactive = cfg.inactive
        settings.particles = cfg.particles

        # IFP for kinetic parameters (beta_eff, Lambda_eff)
        settings.ifp_n_generation = min(5, cfg.inactive)

        # Windowed temperature interpolation between library temperatures
        settings.temperature = {
            'method': 'interpolation',
            'tolerance': cfg.temperature_tolerance,
            'multipole': False,
        }

        if cfg.seed is not None:
            # A fixed seed makes runs bit-reproducible -- essential for the
            # regression workflow in scripts/regression_k.py.
            settings.seed = cfg.seed

        sources = []
        for cavity in self.cavities:
            x0, y0, z0 = cavity.translation
            spatial = openmc.stats.Box(
                lower_left=(x0 - cfg.R1, y0 - cfg.R1, z0),
                upper_right=(x0 + cfg.R1, y0 + cfg.R1, z0 + cfg.L),
            )
            energy = openmc.stats.Watt(a=0.988e6, b=2.249e-6)  # openmc default
            sources.append(openmc.IndependentSource(
                space=spatial, energy=energy, constraints={'fissionable': True}))
        settings.source = sources
        return settings

    # ------------------------------------------------------------------
    # Tally registry
    # ------------------------------------------------------------------

    def register_tally(self, *items) -> None:
        """Register tallies (TallyBundle or bare openmc.Tally) for export.

        Idempotent by name: registering the same name again REPLACES the
        old entry (with a warning) rather than duplicating it, so calling
        an add_* method twice cannot double a tally.
        """
        for item in items:
            bundle = item if isinstance(item, TallyBundle) else TallyBundle(tallies=[item])
            name = bundle.primary.name
            if name in self._registry:
                warnings.warn(f'Tally {name!r} replaced in the registry.')
            self._registry[name] = bundle

    @property
    def all_tallies(self) -> list:
        """Every registered openmc.Tally, flattened, in registration order."""
        out = []
        for bundle in self._registry.values():
            out.extend(bundle.tallies)
        return out

    def _bundle(self, name: str) -> TallyBundle:
        """Fetch a registered bundle or fail with a helpful message."""
        if name not in self._registry:
            raise RuntimeError(
                f'No tally {name!r} registered. Call the matching add_*_tally() '
                f'method BEFORE run() (and before plotting from a statepoint, '
                f'call it again so the mesh context exists).')
        return self._registry[name]

    # Thin convenience wrappers so scripts read like the old API.  Each one
    # is a single call to a factory plus registration -- no bookkeeping.

    def add_power_tally(self, **kwargs) -> None:
        self.register_tally(tally_factories.power_tally(self.config, **kwargs))

    def add_flux_tally(self, **kwargs) -> None:
        self.register_tally(tally_factories.flux_tally(self.config, **kwargs))

    def add_kinetics_tally(self, num_groups: int = 6) -> None:
        self.register_tally(tally_factories.kinetics_tallies(num_groups))

    def add_midplane_flux_tally(self, **kwargs) -> None:
        self.register_tally(tally_factories.midplane_flux_tally(self.config, **kwargs))

    def add_axial_flux_tally(self, **kwargs) -> None:
        self.register_tally(tally_factories.axial_flux_tally(self.config, **kwargs))

    def add_fission_spectrum_tally(self) -> None:
        self.register_tally(tally_factories.fission_spectrum_tally(self.materials))

    # ------------------------------------------------------------------
    # Export / run
    # ------------------------------------------------------------------

    @property
    def statepoint_path(self) -> str:
        return os.path.join(self.output_dir, f'statepoint.{self.config.batches}.h5')

    def export(self) -> None:
        """Write cross_sections.xml and ALL model XML files, exactly once.

        Safe to call repeatedly (e.g. before geometry plotting and again
        before the run) -- it always writes the complete current state.
        """
        if self.model is None:
            raise RuntimeError('Call build() before export().')

        cfg = self.config
        xs_path = build_cross_section_library(cfg, self.output_dir)
        openmc.config['cross_sections'] = xs_path

        self.model.tallies = openmc.Tallies(self.all_tallies)
        self.model.export_to_xml(self.output_dir)

        # Reproducibility anchor: a JSON snapshot of the EXACT configuration
        # lands next to every statepoint.  Months later, any result in this
        # directory can be traced to -- and rebuilt from -- its parameters:
        #     GCRConfig.from_json('settings/gcr_config.json')
        cfg.to_json(os.path.join(self.output_dir, 'gcr_config.json'))

    def run(self, dry_run: bool = False, dry_run_particles: int = 500,
            map_geometry: bool = False) -> str:
        """Export all XML inputs and execute OpenMC.

        Parameters
        ----------
        dry_run :
            Run OpenMC in geometry-debug mode with few particles to check
            for overlaps and lost particles, without committing to a full
            simulation.  ALWAYS do this after any geometry change.
        map_geometry :
            Write a surface/cell ID map, invaluable when chasing lost
            particles.

        Returns
        -------
        str
            Path to the statepoint file the run will have produced.
        """
        self.export()

        if map_geometry:
            self.print_surface_map(
                self.geometry,
                filepath=os.path.join(self.output_dir, 'surface_map.txt'))

        if dry_run:
            print(f'\n--- DRY RUN: geometry debug mode, {dry_run_particles} particles ---')
            openmc.run(geometry_debug=True, particles=dry_run_particles,
                       output=True, cwd=self.output_dir)
            print('--- Dry run complete. Check output for overlaps or lost particles. ---\n')
        else:
            openmc.run(cwd=self.output_dir)

        return self.statepoint_path

    @staticmethod
    def print_surface_map(geometry, filepath: str = 'surface_map.txt'):
        """Write a map of surface/cell IDs to a file, for debugging."""
        surfaces = geometry.get_all_surfaces()
        cells = geometry.get_all_cells()

        with open(filepath, 'w') as f:
            f.write('=== SURFACES ===\n')
            for sid in sorted(surfaces.keys()):
                surf = surfaces[sid]
                label = surf.name if surf.name else '(unnamed)'
                f.write(f'  Surface {sid:>6d} : {surf.type:<20s} {label}\n')

            f.write('\n=== CELLS ===\n')
            for cid in sorted(cells.keys()):
                cell = cells[cid]
                label = cell.name if cell.name else '(unnamed)'
                fill = cell.fill.name if hasattr(cell.fill, 'name') else str(cell.fill)
                f.write(f'  Cell {cid:>6d} : fill={fill:<25s} {label}\n')

        print(f"Surface/cell map written to '{filepath}'")

    # ------------------------------------------------------------------
    # Plotting delegates (thin -- all logic lives in gcr.plotting)
    # ------------------------------------------------------------------

    def plot_geometry(self, figures_dir: str = 'figures',
                      include_voxel: bool = False) -> None:
        plotting.export_geometry_plots(self, figures_dir=figures_dir,
                                       include_voxel=include_voxel)

    def plot_power_distribution(self, statepoint_path: str = None,
                                z_fraction: float = 0.5, save: bool = True,
                                figures_dir: str = 'figures'):
        bundle = self._bundle('power_distribution')
        return plotting.plot_power_distribution(
            self.config, bundle.mesh,
            statepoint_path or self.statepoint_path,
            z_fraction=z_fraction, save=save, figures_dir=figures_dir)

    def plot_flux_distribution(self, statepoint_path: str = None,
                               z_fraction: float = 0.5, save: bool = True,
                               figures_dir: str = 'figures'):
        bundle = self._bundle('flux_distribution')
        return plotting.plot_flux_distribution(
            self.config, bundle.mesh,
            statepoint_path or self.statepoint_path,
            cavities=self.cavities,
            z_fraction=z_fraction, save=save, figures_dir=figures_dir)

    def plot_midplane_flux(self, statepoint_path: str = None, save: bool = True,
                           power_W: float = 4.6e9, figures_dir: str = 'figures'):
        bundle = self._bundle('midplane_flux_groups')
        return plotting.plot_midplane_flux(
            self.config, bundle.mesh, bundle.meta,
            statepoint_path or self.statepoint_path,
            cavities=self.cavities, power_W=power_W,
            save=save, figures_dir=figures_dir)

    def plot_axial_flux(self, statepoint_path: str = None, save: bool = True,
                        power_W: float = 4.6e9, figures_dir: str = 'figures'):
        bundle = self._bundle('axial_flux_groups')
        return plotting.plot_axial_flux(
            self.config, bundle.mesh, bundle.meta,
            statepoint_path or self.statepoint_path,
            power_W=power_W, save=save, figures_dir=figures_dir)

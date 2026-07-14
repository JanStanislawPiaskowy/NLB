"""All plotting: geometry slice plots and statepoint post-processing.

Every function here takes EXPLICIT arguments (config, mesh, statepoint
path, ...) instead of reaching into a model object's private attributes.
Two practical consequences:

  * You can plot from an old statepoint on your laptop without building
    any geometry -- just recreate the matching TallyBundle (cheap, no
    OpenMC run) and hand its mesh to the plot function.  This is what
    scripts/run_reference.py --plot-only does.
  * The infamous axis-ordering trap of ``get_reshaped_data`` (OpenMC
    returns mesh axes as (nz, ny, nx), REVERSED with respect to the mesh
    definition) is handled in exactly the places it was handled before,
    with the explanatory comments kept.

Ported 1:1 from the original plot_* methods.  One intentional deviation
(see README - Migration notes): the original ``plot()`` defined three
geometry plots but then exported an EMPTY ``openmc.Plots([])`` list, so
``openmc.plot_geometry`` produced nothing.  Here the two slice plots are
actually exported; the very heavy voxel plot stays opt-in.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import openmc

from .config import GCRConfig


def fmt_E(E_eV: float) -> str:
    """Render an energy in the most compact unit."""
    if E_eV >= 1e6:
        return f'{E_eV / 1e6:g} MeV'
    if E_eV >= 1e3:
        return f'{E_eV / 1e3:g} keV'
    return f'{E_eV:g} eV'


# ===========================================================================
# Colour maps
# ===========================================================================

def propellant_colour_map(cfg: GCRConfig, materials: dict) -> dict:
    """Propellant coloured as a blue gradient, everything else in greys."""
    all_materials = list(materials.values())
    n = len(all_materials)

    grey_values = [int(60 + (210 - 60) * i / max(n - 1, 1)) for i in range(n)]
    colour_map = {material: (g, g, g) for material, g in zip(all_materials, grey_values)}

    # Collect propellant-duct materials in axial order
    n_layers = cfg.n_axial_layers
    if n_layers > 1:
        layers_materials = []
        if 'hydrogen_header' in materials:                 # header before fuel
            layers_materials.append(materials['hydrogen_header'])
        for k in range(n_layers):                          # fuel zone
            if f'hydrogen_layer_{k}' in materials:
                layers_materials.append(materials[f'hydrogen_layer_{k}'])
    else:
        layers_materials = [materials['hydrogen']]

    # Blue gradient along the duct
    n_layers = len(layers_materials)
    for k, mat in enumerate(layers_materials):
        t = k / max(n_layers - 1, 1)
        colour_map[mat] = (int(173 * (1 - t)),
                           int(216 * (1 - t)),
                           int(230 * (1 - t) + 139 * t))
    return colour_map


def material_colour_map(cfg: GCRConfig, materials: dict) -> dict:
    """Fuel in reds (inlet salmon -> outlet brick red), propellant in blues,
    moderator/structure fixed.  Unassigned materials fall back to grey."""
    n_layers = cfg.n_axial_layers

    # Grey fallback for everything
    all_mats = list(materials.values())
    n = len(all_mats)
    grey = [int(60 + (210 - 60) * i / max(n - 1, 1)) for i in range(n)]
    cmap = {m: (g, g, g) for m, g in zip(all_mats, grey)}

    # --- propellant: blue gradient (inlet light -> outlet dark) ---
    h2 = [materials['hydrogen_header']] if 'hydrogen_header' in materials else []
    if n_layers > 1:
        h2 += [materials[f'hydrogen_layer_{k}'] for k in range(n_layers)
               if f'hydrogen_layer_{k}' in materials]
    else:
        h2 += [materials['hydrogen']]
    for k, m in enumerate(h2):
        t = k / max(len(h2) - 1, 1)
        cmap[m] = (int(173 * (1 - t)), int(216 * (1 - t)),
                   int(230 * (1 - t) + 139 * t))
    for key in ('hydrogen', 'hydrogen_liner', 'hydrogen_tori'):  # keep all H2 blue
        if key in materials:
            cmap[materials[key]] = (90, 150, 210)

    # --- fuel: red gradient (salmon -> brick red) ---
    def red(t):
        return (int(round(255 + (150 - 255) * t)),
                int(round(160 + (40 - 160) * t)),
                int(round(140 + (30 - 140) * t)))

    if n_layers > 1:
        for k in range(n_layers):
            t = k / max(n_layers - 1, 1)
            for pre in ('fuel_inner_layer_', 'fuel_outer_layer_'):
                if f'{pre}{k}' in materials:
                    cmap[materials[f'{pre}{k}']] = red(t)
    for key in ('fuel_inner', 'fuel_outer'):  # canonical (injector/extraction pipes)
        if key in materials:
            cmap[materials[key]] = red(1.0)

    # --- fixed structure / moderator colours ---
    for key, rgb in {
        'graphite': (15, 15, 15),   # near-black grey
        'BeO': (60, 160, 150),      # blue-green / teal
        'SiO2': (160, 160, 160),    # grey
        'neon': (225, 215, 140),    # hazy yellow
    }.items():
        if key in materials:
            cmap[materials[key]] = rgb

    return cmap


# ===========================================================================
# Geometry plots (run through openmc.plot_geometry)
# ===========================================================================

def export_geometry_plots(gcr, figures_dir: str = 'figures',
                          include_voxel: bool = False) -> None:
    """Export plots.xml and run OpenMC in plotting mode.

    Produces the two propellant-gradient slice plots (XY at the fuel
    midplane, YZ through the axis).  The half-reactor VOXEL plot is
    opt-in: at 290x640x580 it is enormous and slow.

    IFP tallying is not allowed in plot mode, so it is temporarily
    disabled in settings and restored afterwards -- exactly the dance the
    original performed.
    """
    cfg = gcr.config
    figures_dir = os.path.abspath(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)

    colour_map_prop = propellant_colour_map(cfg, gcr.materials)
    colour_map_mats = material_colour_map(cfg, gcr.materials)

    plot_xy = openmc.Plot()
    plot_xy.basis = 'xy'
    plot_xy.origin = (0.0, 0.0, cfg.L / 2)
    plot_xy.width = (290, 290)
    plot_xy.pixels = (2000, 2000)
    plot_xy.color_by = 'material'
    plot_xy.colors = colour_map_prop
    plot_xy.filename = os.path.join(figures_dir, 'plot_propellant_gradient')

    plot_yz = openmc.Plot()
    plot_yz.basis = 'yz'
    plot_yz.origin = (0.0, 0.0, cfg.L / 2 + cfg.L_conv / 2 - 10.0)
    plot_yz.width = (350, 280)
    plot_yz.pixels = (3500, 2800)
    plot_yz.color_by = 'material'
    plot_yz.colors = colour_map_prop
    plot_yz.filename = os.path.join(figures_dir, 'plot_propellant_gradient_axial')

    plots = [plot_xy, plot_yz]

    if include_voxel:
        plot_half = openmc.Plot()
        plot_half.type = 'voxel'
        plot_half.origin = (72.5, 0.0, cfg.L / 2 + cfg.L_conv / 2)
        plot_half.width = (145, 320, 290)   # half-width in x -> cut at x = 0
        plot_half.pixels = (290, 640, 580)
        plot_half.color_by = 'material'
        plot_half.colors = colour_map_mats
        plot_half.filename = os.path.join(figures_dir, 'half_reactor')
        plots.append(plot_half)

    openmc.Plots(plots).export_to_xml(os.path.join(gcr.output_dir, 'plots.xml'))

    # IFP is not allowed in plot mode -- temporarily disable it
    saved_ifp = getattr(gcr.settings, 'ifp_n_generation', None)
    gcr.settings.ifp_n_generation = None
    gcr.settings.export_to_xml(os.path.join(gcr.output_dir, 'settings.xml'))
    try:
        openmc.plot_geometry(cwd=gcr.output_dir)
    finally:
        gcr.settings.ifp_n_generation = saved_ifp
        gcr.settings.export_to_xml(os.path.join(gcr.output_dir, 'settings.xml'))


# ===========================================================================
# Statepoint plots
# ===========================================================================

def plot_power_distribution(cfg: GCRConfig, mesh: openmc.RegularMesh,
                            statepoint_path: str, z_fraction: float = 0.5,
                            save: bool = True, figures_dir: str = 'figures'):
    """XY and XZ power-distribution maps from the 'power_distribution' tally.

    z_fraction: fractional axial position (0-1) of the XY slice; 0.5 is the
    fuel-region midplane, typically where peak power occurs.
    """
    sp = openmc.StatePoint(statepoint_path)
    tally = sp.get_tally(name='power_distribution')

    nx, ny, nz = mesh.dimension
    ll, ur = mesh.lower_left, mesh.upper_right

    # OpenMC stores mesh bins as (nz, ny, nx) -- REVERSED with respect to
    # the mesh definition.  Reshape accordingly, then transpose to (nx,ny,nz).
    power = tally.get_reshaped_data(value='mean').reshape(
        nz, ny, nx, 1, 1).squeeze()
    power = np.transpose(power, (2, 1, 0))

    z_idx = int(z_fraction * nz)
    z_cm = ll[2] + (z_idx + 0.5) * (ur[2] - ll[2]) / nz

    xy_slice = power[:, :, z_idx]
    x_idx = nx // 2
    yz_slice = power[x_idx, :, :].T   # shape (nz, ny), z on the vertical axis

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    im0 = axes[0].imshow(xy_slice, origin='lower',
                         extent=[ll[0], ur[0], ll[1], ur[1]],
                         cmap='hot', aspect='equal')
    plt.colorbar(im0, ax=axes[0], label='Fission-Q recoverable (eV/source particle)')
    axes[0].set_title(f'XY power distribution  |  z = {z_cm:.1f} cm (axial midplane)')
    axes[0].set_xlabel('X (cm)')
    axes[0].set_ylabel('Y (cm)')

    im1 = axes[1].imshow(yz_slice, origin='lower',
                         extent=[ll[1], ur[1], ll[2], ur[2]],
                         cmap='hot', aspect='equal')
    plt.colorbar(im1, ax=axes[1], label='Fission-Q recoverable (eV/source particle)')
    axes[1].set_title('XZ power distribution  |  y = 0 (central cavity slice)')
    axes[1].set_xlabel('X (cm)')
    axes[1].set_ylabel('Z (cm)')

    plt.suptitle('GCR Power Distribution', fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out = os.path.join(figures_dir, 'power_distribution.pdf')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')

    plt.show()
    return fig


def plot_flux_distribution(cfg: GCRConfig, mesh: openmc.RegularMesh,
                           statepoint_path: str, cavities=(),
                           z_fraction: float = 0.5, save: bool = True,
                           figures_dir: str = 'figures'):
    """XY and XZ total-flux maps from the 'flux_distribution' tally."""
    sp = openmc.StatePoint(statepoint_path)
    tally = sp.get_tally(name='flux_distribution')

    nx, ny, nz = mesh.dimension
    ll, ur = mesh.lower_left, mesh.upper_right

    flux = tally.get_reshaped_data(value='mean').squeeze()
    if flux.ndim != 3:
        flux = flux.reshape(nx, ny, nz)

    z_idx = int(z_fraction * nz)
    z_cm = ll[2] + (z_idx + 0.5) * (ur[2] - ll[2]) / nz

    xy_slice = flux[:, :, z_idx].T   # (ny, nx)
    y_idx = ny // 2
    xz_slice = flux[:, y_idx, :].T   # (nz, nx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(xy_slice, origin='lower',
                         extent=[ll[0], ur[0], ll[1], ur[1]],
                         cmap='viridis', aspect='equal')
    plt.colorbar(im0, ax=axes[0], label='Flux (n/source particle*cm2)')
    axes[0].set_title(f'XY flux distribution  |  z = {z_cm:.1f} cm')
    axes[0].set_xlabel('X (cm)')
    axes[0].set_ylabel('Y (cm)')

    for cavity in cavities:                     # mark cavity centres
        cx, cy, _ = cavity.translation
        axes[0].plot(cx, cy, 'w+', markersize=8, markeredgewidth=1.5)

    im1 = axes[1].imshow(xz_slice, origin='lower',
                         extent=[ll[0], ur[0], ll[2], ur[2]],
                         cmap='viridis', aspect='auto')
    plt.colorbar(im1, ax=axes[1], label='Flux (n/source particle*cm2)')
    axes[1].set_title('XZ flux distribution  |  y = 0')
    axes[1].set_xlabel('X (cm)')
    axes[1].set_ylabel('Z (cm)')

    plt.suptitle('GCR Neutron Flux Distribution', fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out = os.path.join(figures_dir, 'flux_distribution.pdf')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')

    plt.show()
    return fig


def _power_normalisation(statepoint_path: str, power_W: float) -> float:
    """Source rate S [src/s] such that the tallied power equals power_W."""
    eV_to_J = 1.602176634e-19
    sp = openmc.StatePoint(statepoint_path)
    power_tally = sp.get_tally(name='power_distribution')
    Q_per_src = power_tally.get_reshaped_data(value='mean').sum()  # eV/src, all voxels
    return power_W / (Q_per_src * eV_to_J)


def plot_midplane_flux(cfg: GCRConfig, mesh: openmc.RegularMesh, meta: dict,
                       statepoint_path: str, cavities=(), save: bool = True,
                       power_W: float = 4.6e9, figures_dir: str = 'figures'):
    """3-group midplane flux maps + line-outs, normalised to reactor power.

    power_W: pass None to leave the tally output in n/cm2/src.  The
    normalisation needs the 'power_distribution' tally in the same
    statepoint (add_power_tally must have been active during the run).
    """
    sp = openmc.StatePoint(statepoint_path)
    tally = sp.get_tally(name='midplane_flux_groups')

    nx, ny, _ = mesh.dimension
    ll, ur = mesh.lower_left, mesh.upper_right
    cE_th = meta['thermal_cutoff']
    cE_ep = meta['epithermal_cutoff']

    flux = tally.get_reshaped_data(value='mean').squeeze()
    if flux.shape != (nx, ny, 3):
        flux = tally.get_reshaped_data(value='mean').reshape(
            nx, ny, 1, 3, 1, 1).squeeze()

    if power_W is not None:
        S = _power_normalisation(statepoint_path, power_W)
        flux = flux * S
        flux_unit = 'n*cm/s'
        norm_label = f'P = {power_W * 1e-9:g} GW'
    else:
        flux_unit = 'n/cm2/src'
        norm_label = 'per source neutron'

    extent = [ll[0], ur[0], ll[1], ur[1]]
    z_label = f'z = {0.5 * (ll[2] + ur[2]):.1f} cm, dz = {ur[2] - ll[2]:.1f} cm'

    labels = [
        f'Thermal (E < {fmt_E(cE_th)})',
        f'Epithermal ({fmt_E(cE_th)} < E < {fmt_E(cE_ep)})',
        f'Fast (E > {fmt_E(cE_ep)})',
    ]
    cmaps = ['viridis', 'cividis', 'inferno']

    # -- Figure 1: 1x3 heatmaps ------------------------------------------------
    fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
    z_slice = 0.5 * (ll[2] + ur[2])

    for g, (ax, label, cmap) in enumerate(zip(axes, labels, cmaps)):
        im = ax.imshow(flux[:, :, g].T, origin='lower', extent=extent,
                       cmap=cmap, aspect='equal', interpolation='nearest')
        plt.colorbar(im, ax=ax, label=f'Flux ({flux_unit})')
        ax.set_title(label)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')

        # Cavity centres at the slice plane: solve R[2,2]*z_local + tz = z_slice
        for cavity in cavities:
            R = np.asarray(cavity.rotation)
            tx, ty, tz = cavity.translation
            if abs(R[2, 2]) < 1e-12:
                continue
            z_local = (z_slice - tz) / R[2, 2]
            cx = R[0, 2] * z_local + tx
            cy = R[1, 2] * z_local + ty
            ax.plot(cx, cy, 'w+', markersize=10, markeredgewidth=1.5)

    plt.suptitle(f'Midplane flux  ({z_label}, {norm_label})', y=1.02)
    plt.tight_layout()
    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out = os.path.join(figures_dir, 'midplane_flux_2D.pdf')
        fig1.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')

    # -- Figure 2: line-outs -----------------------------------------------------
    x_c = 0.5 * (np.linspace(ll[0], ur[0], nx + 1)[:-1]
                 + np.linspace(ll[0], ur[0], nx + 1)[1:])
    y_c = 0.5 * (np.linspace(ll[1], ur[1], ny + 1)[:-1]
                 + np.linspace(ll[1], ur[1], ny + 1)[1:])
    i_x0, j_y0 = nx // 2, ny // 2
    colours = ['C0', 'C2', 'C3']

    fig2, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for g, (label, c) in enumerate(zip(labels, colours)):
        axes[0].plot(x_c, flux[:, j_y0, g], color=c, label=label)
        axes[1].plot(y_c, flux[i_x0, :, g], color=c, label=label)
    axes[0].set_xlabel('x (cm)')
    axes[0].set_title('Line-out along y = 0')
    axes[1].set_xlabel('y (cm)')
    axes[1].set_title('Line-out along x = 0')
    for ax in axes:
        ax.set_ylabel(f'Flux ({flux_unit})')
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
    plt.tight_layout()
    if save:
        out = os.path.join(figures_dir, 'midplane_flux_lineout.pdf')
        fig2.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.show()
    return fig1, fig2


def plot_axial_flux(cfg: GCRConfig, mesh: openmc.RegularMesh, meta: dict,
                    statepoint_path: str, save: bool = True,
                    power_W: float = 4.6e9, figures_dir: str = 'figures'):
    """3-group axial flux on the y = 0 slab + line-outs, power-normalised."""
    sp = openmc.StatePoint(statepoint_path)
    tally = sp.get_tally(name='axial_flux_groups')

    nx, ny, nz = mesh.dimension
    ll, ur = mesh.lower_left, mesh.upper_right
    cE_th = meta['thermal_cutoff']
    cE_ep = meta['epithermal_cutoff']

    # Mesh is [1, ny, nz] (thin x-slab).  OpenMC stores bins as (nz, ny, nx),
    # so get_reshaped_data returns (nz, ny, 1, ...).  Squeeze the x dimension
    # and transpose to (ny, nz, 3) so flux is indexed as [i_y, i_z, g].
    flux = tally.get_reshaped_data(value='mean').reshape(
        nz, ny, 1, 3, 1, 1).squeeze()      # -> (nz, ny, 3)
    flux = np.transpose(flux, (1, 0, 2))   # -> (ny, nz, 3)

    if power_W is not None:
        S = _power_normalisation(statepoint_path, power_W)
        flux = flux * S
        flux_unit = 'n*cm/s'
        norm_label = f'P = {power_W * 1e-9:g} GW'
    else:
        flux_unit = 'n/cm2/src'
        norm_label = 'per source'

    L = cfg.L
    x_label = f'y = 0, dy = {ur[0] - ll[0]:.1f} cm'
    labels = [
        f'Thermal (E < {fmt_E(cE_th)})',
        f'Epithermal ({fmt_E(cE_th)} < E < {fmt_E(cE_ep)})',
        f'Fast (E > {fmt_E(cE_ep)})',
    ]
    cmaps = ['viridis', 'cividis', 'inferno']

    # -- Figure 1: 3 heatmaps, z vertical -------------------------------------
    extent = [ll[1], ur[1], ur[2], ll[2]]   # (y_min, y_max, z_max, z_min)
    y_span = ur[1] - ll[1]
    z_span = ur[2] - ll[2]
    panel_h = 6.5
    panel_w = max(6.0, panel_h * y_span / z_span)
    fig1, axes = plt.subplots(1, 3, figsize=(3 * panel_w + 3, panel_h + 1.5),
                              constrained_layout=True)

    for g, (ax, label, cmap) in enumerate(zip(axes, labels, cmaps)):
        im = ax.imshow(flux[:, :, g].T, origin='upper', extent=extent,
                       cmap=cmap, aspect='equal', interpolation='nearest')
        plt.colorbar(im, ax=ax, label=f'Flux ({flux_unit})', shrink=0.85)
        ax.set_title(label)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('z (cm)')
        if ll[2] <= 0.0 <= ur[2]:
            ax.axhline(0.0, color='w', lw=0.6, ls='--', alpha=0.6)
        if ll[2] <= L <= ur[2]:
            ax.axhline(L, color='w', lw=0.6, ls='--', alpha=0.6)

    fig1.suptitle(f'Axial flux ({x_label}, {norm_label})')
    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out = os.path.join(figures_dir, 'axial_flux_2D.pdf')
        fig1.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')

    # -- Figure 2: line-outs ------------------------------------------------------
    y_edges = np.linspace(ll[1], ur[1], ny + 1)
    z_edges = np.linspace(ll[2], ur[2], nz + 1)
    y_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_c = 0.5 * (z_edges[:-1] + z_edges[1:])
    j_y0 = int(np.argmin(np.abs(y_c)))
    k_zL2 = int(np.argmin(np.abs(z_c - L / 2)))
    colours = ['C0', 'C2', 'C3']

    fig2, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for g, (label, c) in enumerate(zip(labels, colours)):
        axes[0].plot(z_c, flux[j_y0, :, g], color=c, label=label)
        axes[1].plot(y_c, flux[:, k_zL2, g], color=c, label=label)
    axes[0].set_xlabel('z (cm)')
    axes[0].set_title('Axial profile at x = 0')
    if ll[2] <= 0.0 <= ur[2]:
        axes[0].axvline(0.0, color='k', lw=0.5, ls='--')
    if ll[2] <= L <= ur[2]:
        axes[0].axvline(L, color='k', lw=0.5, ls='--')
    axes[1].set_xlabel('x (cm)')
    axes[1].set_title(f'Radial profile at z = L/2 = {L / 2:.1f} cm')
    for ax in axes:
        ax.set_ylabel(f'Flux ({flux_unit})')
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
    plt.tight_layout()
    if save:
        out = os.path.join(figures_dir, 'axial_flux_lineout.pdf')
        fig2.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.show()
    return fig1, fig2

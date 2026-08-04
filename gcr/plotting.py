"""All plotting: geometry slice plots and statepoint post-processing.

GEOMETRY FIGURES: one component per figure.  Each greys the whole reactor and
colours a single part.  The default set is

    pressure_shell    core    radial + axial
    graphite          core    radial + axial
    beo               core    radial + axial
    propellant        core    radial + axial   (duct H2 only, temperature ramp)
    fuel              core    radial + axial   (inner and outer zone separated)
    neon              core    radial + axial
    liner             cavity  radial + axial   (Be wall + coolant, one colour)
    transparent_wall  cavity  radial + axial   (SiO2 + tori coolant, one colour)

Material rules match on the DICT KEY of gcr.materials, never on
Material.name, because the names do not track the keys: key
"""

import colorsys
import os
import re
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import openmc

from .config import GCRConfig

RGB = Tuple[int, int, int]


def fmt_E(E_eV: float) -> str:
    """Render an energy in the most compact unit."""
    if E_eV >= 1e6:
        return f'{E_eV / 1e6:g} MeV'
    if E_eV >= 1e3:
        return f'{E_eV / 1e3:g} keV'
    return f'{E_eV:g} eV'


# ===========================================================================
# 1.  Colour policy
# ===========================================================================

#: Flat highlight colours, 0-255 RGB.
ROLE_COLOURS: Dict[str, RGB] = {
    'pressure_shell':   (184, 134,  11),   # dark goldenrod
    'graphite':         ( 84,  39, 143),   # dark violet
    'beo':              (  0, 102,  44),   # dark green
    'liner':            (124, 214, 124),   # light green
    'transparent_wall': (236, 112, 179),   # pink
    'neon':             ( 70, 118, 160),   # steel / grey-blue
    'fuel':             (200,  60,  40),   # only if the zone split ever fails
}

#: Two-point gradients: (colour at MINIMUM T, colour at MAXIMUM T).
GRADIENTS: Dict[str, Tuple[RGB, RGB]] = {
    'propellant': ((173, 216, 230), (  0,   0, 139)),   # light blue -> navy
    'fuel_inner': ((255, 209,  41), (139,   0,   0)),   # saturated gold -> dark red
    'fuel_outer': ((255, 236, 168), (206, 122, 100)),   # pale gold -> dusty brick
}

#: Grey each role takes while it is NOT the highlighted component.
#:
#: Laid out as a ladder with >= 15 levels between any two components that
#: touch each other in the radial build, so the greyscale background still
#: reads as a reactor.  The propellant sits at mid grey (158): at the old 215
#: it was nearly white and the duct disappeared into the page.
ROLE_GREYS: Dict[str, int] = {
    'fuel_inner':        50,
    'fuel_outer':        68,
    'pressure_shell':    78,
    'graphite':          92,
    'transparent_wall': 110,
    'liner':            128,
    'propellant':       158,   # <- the duct: mid grey, not near-white
    'h2_other':         174,   # generic H2 that is not duct, liner or tori
    'neon':             196,
    'beo':              214,
}

#: Grey ramp for materials no rule matched.
GREY_RANGE: Tuple[int, int] = (70, 205)


@dataclass(frozen=True)
class MaterialRule:
    """First-match-wins rule mapping a material onto a colour role.

    ``key``  regex against the KEY of ``gcr.materials`` (e.g. ``hydrogen_layer_3``);
    ``name`` regex against ``openmc.Material.name``.
    Both given -> both must match.  Case-insensitive.
    """
    role: str
    key: Optional[str] = None
    name: Optional[str] = None


def default_material_rules() -> List[MaterialRule]:
    """Ordered rules.  ORDER IS LOAD-BEARING throughout.

    The hydrogen block is the delicate part: five different hydrogens exist
    and they belong to four different components, so every specific hydrogen
    must be claimed BEFORE the generic ``^hydrogen`` catch-all.
    """
    return [
        # -- fuel: the two coaxial zones stay separate so the radial build
        #    is visible.  ^fuel_inner also catches fuel_inner_layer_k and the
        #    canonical fuel_inner used by the injector / extraction pipes.
        MaterialRule('fuel_inner',       key=r'^fuel_inner'),
        MaterialRule('fuel_outer',       key=r'^fuel_outer'),
        MaterialRule('fuel',             key=r'^fuel'),
        MaterialRule('fuel',             name=r'fuel|u-?233|uranium'),

        # -- hydrogen, most specific first ---------------------------------
        # inter-shell annulus gas: part of the pressure vessel, not propellant
        MaterialRule('pressure_shell',   key=r'^hydrogen_pv$'),
        # coolant inside the liner tubes -> coloured WITH the liner walls
        MaterialRule('liner',            key=r'^hydrogen_liner$'),
        # coolant inside the transparent-wall tori -> coloured WITH the wall
        MaterialRule('transparent_wall', key=r'^hydrogen_tori$'),
        # THE PROPELLANT ANNULUS, and nothing else: the ten duct layers plus
        # the header.  For an n_axial_layers <= 1 run the duct is filled with
        # plain 'hydrogen' instead, so add r'|^hydrogen$' here for those runs
        # (the group is skipped with a message otherwise).
        MaterialRule('propellant',       key=r'^hydrogen_layer_\d+$|^hydrogen_header$'),
        # everything else made of H2 (end walls, coolant passages, ...)
        MaterialRule('h2_other',         key=r'^hydrogen'),
        MaterialRule('h2_other',         name=r'hydrogen|hyrdogen|\bh2\b'),

        # -- rest of the cavity interior ------------------------------------
        MaterialRule('neon',             key=r'^neon$|buffer'),
        MaterialRule('neon',             name=r'^ne$|\bneon\b|buffer'),
        MaterialRule('transparent_wall', key=r'^sio2$|silica|quartz|transparent'),
        MaterialRule('transparent_wall', name=r'sio2|silica|quartz|transparent'),

        # -- moderator BEFORE the liner rule: 'BeO' must never be swallowed
        #    by a loose beryllium pattern.
        MaterialRule('beo',              key=r'^beo$'),
        MaterialRule('beo',              name=r'\bbeo\b|beryllium\s*oxide'),
        MaterialRule('graphite',         key=r'^graphite$|carbon'),
        MaterialRule('graphite',         name=r'graphite|carbon'),

        # -- structure --------------------------------------------------------
        MaterialRule('pressure_shell',   key=r'fibre|fiber|filament|vessel|shell|steel|inconel'),
        MaterialRule('pressure_shell',   name=r'fibre|fiber|filament|vessel|shell|steel|inconel'),
        # NOTE: 'Be' fills BOTH the liner tube walls and the tie rods, so a
        # material-level liner highlight lights up the tie rods too.  If that
        # matters, pass color_by='cell' -- the cell-name override splits them.
        MaterialRule('liner',            key=r'^be$|^beryllium$|liner'),
        MaterialRule('liner',            name=r'^be$|^beryllium$'),
    ]


def classify_materials(materials: Dict[str, openmc.Material],
                       rules: Optional[Sequence[MaterialRule]] = None,
                       verbose: bool = False):
    """materials dict -> ({role: [(key, Material), ...]}, [unmatched keys])."""
    rules = list(rules if rules is not None else default_material_rules())
    roles: Dict[str, List[Tuple[str, openmc.Material]]] = {}
    unassigned: List[str] = []

    for key, mat in materials.items():
        mname = getattr(mat, 'name', '') or ''
        for rule in rules:
            if rule.key is None and rule.name is None:
                continue
            if rule.key is not None and not re.search(rule.key, key, re.I):
                continue
            if rule.name is not None and not re.search(rule.name, mname, re.I):
                continue
            roles.setdefault(rule.role, []).append((key, mat))
            break
        else:
            unassigned.append(key)

    if verbose:
        print('Material -> role classification')
        for role in sorted(roles):
            keys = [k for k, _ in roles[role]]
            head = ', '.join(keys[:5]) + (' ...' if len(keys) > 5 else '')
            print(f'  {role:<18s} {len(keys):>4d}  [{head}]')
        if unassigned:
            print(f'  {"(grey ramp)":<18s} {len(unassigned):>4d}  '
                  f'[{", ".join(unassigned)}]')
    return roles, unassigned


# ---------------------------------------------------------------------------

def _lerp(c0: RGB, c1: RGB, t: float) -> RGB:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))


def _shade_family(base: RGB, n: int, spread: float = 0.42) -> List[RGB]:
    """n shades of one hue, light -> dark, at constant hue and saturation.

    Used where a role holds several materials that should stay separable --
    the two pressure shells and the gas between them, for instance.  Switch
    it off (PlotGroup.shade = False) where the materials should read as ONE
    object: the liner tube wall and its coolant must merge into a single
    green tube, not a green ring around a different green core.
    """
    if n <= 1:
        return [tuple(base)]
    r, g, b = (c / 255.0 for c in base)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    out = []
    for i in range(n):
        t = i / (n - 1)
        L = float(np.clip(l * (1.0 + spread) * (1 - t) + l * (1.0 - spread) * t,
                          0.10, 0.90))
        rr, gg, bb = colorsys.hls_to_rgb(h, L, s)
        out.append((int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255))))
    return out


_NUM_SPLIT = re.compile(r'(\d+)')


def _natural_key(s: str):
    return [int(p) if p.isdigit() else p.lower() for p in _NUM_SPLIT.split(s)]


def _temperatures(mats: Sequence[openmc.Material]) -> np.ndarray:
    """Material temperatures in K; NaN where OpenMC has none set."""
    out = []
    for m in mats:
        T = getattr(m, 'temperature', None)
        out.append(float(T) if T is not None else np.nan)
    return np.asarray(out, dtype=float)


@dataclass
class ColourScheme:
    """Everything needed to paint one figure AND to draw its legend."""
    colours: Dict[openmc.Material, RGB] = field(default_factory=dict)
    roles: Dict[str, List[Tuple[str, openmc.Material]]] = field(default_factory=dict)
    highlighted: Tuple[str, ...] = ()
    gradients: Dict[str, dict] = field(default_factory=dict)
    flats: Dict[str, List[RGB]] = field(default_factory=dict)
    unassigned: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(self.roles.get(r) for r in self.highlighted)


def build_colour_scheme(cfg: GCRConfig,
                        materials: Dict[str, openmc.Material],
                        highlight: Sequence[str],
                        rules: Optional[Sequence[MaterialRule]] = None,
                        role_colours: Optional[Dict[str, RGB]] = None,
                        gradients: Optional[Dict[str, Tuple[RGB, RGB]]] = None,
                        pool_roles: Sequence[Sequence[str]] = (),
                        shade_flat_roles: bool = True,
                        verbose: bool = False) -> ColourScheme:
    """Grey everywhere, colour only the roles in ``highlight``.

    Gradient roles ramp by MATERIAL TEMPERATURE, not layer index.  For the
    fuel that is the point: fuel_inner_layer_k and fuel_outer_layer_k share
    an index but sit at different temperatures, so an index ramp painted them
    identically and the radial build vanished.

    ``pool_roles`` lists role tuples normalised over a COMMON temperature
    range.  Pooling the two fuel zones makes their colours directly
    comparable (inner uniformly dark, outer uniformly pale, axial variation
    squeezed); not pooling gives each zone its own full ramp so the axial
    gradient stays legible inside each zone.  Default: not pooled.
    """
    role_colours = {**ROLE_COLOURS, **(role_colours or {})}
    gradients = {**GRADIENTS, **(gradients or {})}
    highlight = tuple(highlight)

    roles, unassigned = classify_materials(materials, rules, verbose=verbose)
    scheme = ColourScheme(roles=roles, highlighted=highlight, unassigned=unassigned)

    # -- 1. fixed grey per role -------------------------------------------
    for role, entries in roles.items():
        g = ROLE_GREYS.get(role, 160)
        for _, mat in entries:
            scheme.colours[mat] = (g, g, g)

    # -- 2. stable grey ramp for whatever no rule matched -------------------
    if unassigned:
        lo, hi = GREY_RANGE
        for i, key in enumerate(sorted(unassigned, key=_natural_key)):
            g = int(round(lo + (hi - lo) * i / max(len(unassigned) - 1, 1)))
            scheme.colours[materials[key]] = (g, g, g)

    # -- 3. highlighted flat roles -> shade family --------------------------
    for role in highlight:
        if role in gradients or not roles.get(role):
            continue
        base = role_colours.get(role)
        if base is None:
            continue
        entries = sorted(roles[role], key=lambda kv: _natural_key(kv[0]))
        shades = (_shade_family(base, len(entries)) if shade_flat_roles
                  else [tuple(base)] * len(entries))
        for (_, mat), rgb in zip(entries, shades):
            scheme.colours[mat] = rgb
        scheme.flats[role] = shades

    # -- 4. highlighted gradient roles -> temperature ramp ------------------
    def _pool(role):
        for p in pool_roles:
            if role in p:
                return tuple(p)
        return (role,)

    for role in highlight:
        if role not in gradients or not roles.get(role):
            continue
        c_lo, c_hi = gradients[role]
        entries = sorted(roles[role], key=lambda kv: _natural_key(kv[0]))
        mats = [m for _, m in entries]
        T = _temperatures(mats)
        pool_mats = [m for r in _pool(role) for _, m in roles.get(r, [])]
        Tp = _temperatures(pool_mats)

        if (np.all(np.isfinite(T)) and np.all(np.isfinite(Tp))
                and (Tp.max() - Tp.min()) > 1.0):
            t = (T - Tp.min()) / (Tp.max() - Tp.min())
            basis, rng = 'temperature', (float(Tp.min()), float(Tp.max()))
        else:
            t = np.linspace(0.0, 1.0, len(mats)) if len(mats) > 1 else np.zeros(1)
            basis, rng = 'index', None
        stops = [_lerp(c_lo, c_hi, x) for x in t]
        for mat, rgb in zip(mats, stops):
            scheme.colours[mat] = rgb
        scheme.gradients[role] = {
            'basis': basis, 'temperatures': T, 'stops': stops, 'range': rng,
            'keys': [k for k, _ in entries], 'ends': (c_lo, c_hi),
        }
        if verbose:
            rng_s = f'{rng[0]:.0f}-{rng[1]:.0f} K' if rng else 'no T'
            print(f'  gradient {role:<12s} ramped by {basis:<11s} '
                  f'({rng_s}, {len(mats)} materials)')

    return scheme


# -- backwards-compatible thin wrappers -------------------------------------

def propellant_colour_map(cfg: GCRConfig, materials: dict) -> dict:
    """Duct propellant as a blue gradient, everything else grey."""
    return build_colour_scheme(cfg, materials, highlight=('propellant',)).colours


def material_colour_map(cfg: GCRConfig, materials: dict) -> dict:
    """Everything highlighted at once -- kept only for the voxel plot."""
    return build_colour_scheme(
        cfg, materials,
        highlight=('pressure_shell', 'graphite', 'beo', 'propellant', 'fuel',
                   'fuel_inner', 'fuel_outer', 'neon', 'liner',
                   'transparent_wall')).colours


# ---------------------------------------------------------------------------
# Cell-level colouring (only when a component shares its material)
# ---------------------------------------------------------------------------

#: cell-name regex -> role, applied on top of the material colours.  The
#: liner entry is the one that matters: 'Be' fills the liner tube walls AND
#: the tie rods, so only a cell-name test can separate them.
DEFAULT_CELL_OVERRIDES: Tuple[Tuple[str, str], ...] = (
    (r'tie[_ -]?rod', 'tie_rod'),
    (r'liner', 'liner'),
    (r'transparent|torus|tori', 'transparent_wall'),
    (r'pressure[_ -]?vessel|vessel|shell|fibre|fiber', 'pressure_shell'),
)

#: colours used only by the cell-name overrides
OVERRIDE_COLOURS: Dict[str, RGB] = {'tie_rod': (150, 150, 150)}


def _material_fills(cell: openmc.Cell) -> List[openmc.Material]:
    fill = cell.fill
    if isinstance(fill, openmc.Material):
        return [fill]
    if isinstance(fill, (list, tuple)):
        return [m for m in fill if isinstance(m, openmc.Material)]
    return []


def cell_colour_map(gcr, scheme: ColourScheme,
                    overrides: Sequence[Tuple[str, str]] = DEFAULT_CELL_OVERRIDES,
                    role_colours: Optional[Dict[str, RGB]] = None,
                    verbose: bool = False) -> Dict[openmc.Cell, RGB]:
    """Per-CELL colours derived from the per-MATERIAL scheme.

    Needed when two components that must be told apart share a material.  In
    this model that is exactly one case: beryllium in the liner tubes and in
    the tie rods.  Cost: ~2e4 leaf cells, so plots.xml grows about 1 MB per
    plot -- use it for the liner figures only if the tie rods are in frame.
    """
    role_colours = {**ROLE_COLOURS, **OVERRIDE_COLOURS, **(role_colours or {})}
    compiled = [(re.compile(pat, re.I), role) for pat, role in overrides
                if role in scheme.highlighted or role in OVERRIDE_COLOURS]
    out: Dict[openmc.Cell, RGB] = {}

    for _, cell in gcr.geometry.get_all_cells().items():
        fills = _material_fills(cell)
        if not fills:                      # universe / lattice fill: never painted
            continue
        rgb = scheme.colours.get(fills[0], (160, 160, 160))
        cname = cell.name or ''
        for rx, role in compiled:
            if rx.search(cname):
                rgb = role_colours.get(role, rgb)
                break
        out[cell] = rgb

    if verbose:
        print(f'cell_colour_map: {len(out)} leaf cells')
    return out


# ===========================================================================
# 2.  Plot windows
# ===========================================================================
#
# openmc.Plot.origin is the CENTRE of the window and width is the FULL extent,
# so "show one half" is not a crop: halve the width AND push the origin by a
# quarter of the original width.  That arithmetic lives in half_window().

_BASIS_AXES = {'xy': ('x', 'y'), 'xz': ('x', 'z'), 'yz': ('y', 'z')}
_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


@dataclass(frozen=True)
class Window:
    """A slice-plot window, resolution-independent."""
    basis: str                                   # 'xy' | 'xz' | 'yz'
    origin: Tuple[float, float, float]           # centre, cm
    width: Tuple[float, float]                   # (horizontal, vertical), cm
    px_per_cm: float = 7.0
    label: str = ''

    @property
    def pixels(self) -> Tuple[int, int]:
        return (max(1, int(round(self.width[0] * self.px_per_cm))),
                max(1, int(round(self.width[1] * self.px_per_cm))))

    @property
    def cm_per_px(self) -> float:
        return 1.0 / self.px_per_cm

    def bounds(self):
        h, v = _BASIS_AXES[self.basis]
        oh, ov = self.origin[_AXIS_INDEX[h]], self.origin[_AXIS_INDEX[v]]
        return ((oh - self.width[0] / 2, oh + self.width[0] / 2),
                (ov - self.width[1] / 2, ov + self.width[1] / 2))

    def describe(self) -> str:
        (h0, h1), (v0, v1) = self.bounds()
        h, v = _BASIS_AXES[self.basis]
        px = self.pixels
        return (f'{self.basis}  {h}:[{h0:8.2f},{h1:8.2f}]  '
                f'{v}:[{v0:8.2f},{v1:8.2f}]  '
                f'{px[0]}x{px[1]} px  {self.cm_per_px * 10:.2f} mm/px')


def grow_window(win: Window, dh: float = 0.0, dv: float = 0.0) -> Window:
    """Push the window out by dh / dv centimetres ON EACH SIDE.

    Margins are the distance the picture gains outwards, which is what you
    estimate when geometry runs off the edge; total width grows by 2*dh.
    Grow BEFORE halving so the kept half gains the full dh.
    """
    return replace(win, width=(win.width[0] + 2 * dh, win.width[1] + 2 * dv))


def half_window(win: Window, side: Optional[str]) -> Window:
    """Keep one half of ``win``.  ``side`` is '+x'|'-x'|'+y'|'-y'|'+z'|'-z'.

    The cut plane passes through the window CENTRE, so on a cavity-centred
    zoom it cuts through that cavity's axis.  Apply twice on two axes for a
    quadrant.  Pixel PITCH is preserved, so half and full plots of the same
    object print at the same scale.
    """
    if not side:
        return win
    sign, letter = side[0], side[1] if len(side) > 1 else None
    if sign not in '+-' or letter not in _AXIS_INDEX:
        raise ValueError(f"side must look like '+y', got {side!r}")
    h, v = _BASIS_AXES[win.basis]
    if letter == h:
        wi = 0
    elif letter == v:
        wi = 1
    else:
        raise ValueError(f"axis {letter!r} is normal to basis {win.basis!r}: "
                         f"it is the slice normal, not an in-plane direction")
    w, o = list(win.width), list(win.origin)
    shift = w[wi] / 4.0
    w[wi] = w[wi] / 2.0
    o[_AXIS_INDEX[letter]] += shift if sign == '+' else -shift
    lab = (win.label + ', ' if win.label else '') + f'{letter} {sign}ve half'
    return replace(win, origin=tuple(o), width=tuple(w), label=lab)


#: Whole-core windows BEFORE margins.  Deliberately hard-coded rather than
#: taken from geometry.bounding_box: a case with a different reflector
#: thickness would otherwise render at a different scale.
BASE_RADIAL_WIDTH = (290.0, 290.0)
BASE_AXIAL_WIDTH = (350.0, 280.0)


def default_windows(cfg: GCRConfig, px_per_cm: float = 7.0,
                    radial_margin: float = 40.0,
                    axial_margin_r: float = 20.0,
                    axial_margin_z: float = 10.0) -> Dict[str, Window]:
    """Whole-core windows with the outward margins applied.

    radial_margin  = 40 -> x, y reach 145 + 40 = 185 cm
    axial_margin_r = 20 -> transverse reaches 175 + 20 = 195 cm
    axial_margin_z = 10 -> z gains 10 cm at BOTH ends
    (halve these if you meant "total width + 40 cm" instead.)
    """
    z_mid = cfg.L / 2
    z_ax = cfg.L / 2 + cfg.L_conv / 2 - 10.0
    rad = Window('xy', (0.0, 0.0, z_mid), BASE_RADIAL_WIDTH, px_per_cm,
                 f'z = {z_mid:.1f} cm')
    ax = Window('yz', (0.0, 0.0, z_ax), BASE_AXIAL_WIDTH, px_per_cm, 'x = 0')
    return {'radial': grow_window(rad, radial_margin, radial_margin),
            'axial': grow_window(ax, axial_margin_r, axial_margin_z)}


# ---------------------------------------------------------------------------
# Cavity bookkeeping
# ---------------------------------------------------------------------------

def _rotation_matrix(cav) -> np.ndarray:
    """3x3 rotation for a cavity placement, whatever form it is stored in."""
    R = getattr(cav, 'rotation_matrix', None)
    if R is None:
        R = getattr(cav, 'rotation', None)
    if R is None:
        return np.eye(3)
    R = np.asarray(R, dtype=float)
    if R.shape == (3, 3):
        return R
    if R.shape == (3,):        # OpenMC x-y-z Euler angles, degrees
        phi, theta, psi = np.radians(R)
        cx, sx = np.cos(phi), np.sin(phi)
        cy, sy = np.cos(theta), np.sin(theta)
        cz, sz = np.cos(psi), np.sin(psi)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx
    return np.eye(3)


def cavity_centres(cavities: Iterable, z: float) -> List[Tuple[float, float]]:
    """(x, y) where each cavity AXIS crosses the plane z = const.

    A tilted cavity is not at its translation vector once you leave the
    reference plane; this solves R[2,2]*z_local + tz = z and evaluates the
    axis there -- the algebra already used in plot_midplane_flux.
    """
    out = []
    for cav in cavities:
        R = _rotation_matrix(cav)
        tx, ty, tz = (float(c) for c in cav.translation)
        if abs(R[2, 2]) < 1e-12:
            out.append((tx, ty))
            continue
        z_local = (z - tz) / R[2, 2]
        out.append((R[0, 2] * z_local + tx, R[1, 2] * z_local + ty))
    return out


def describe_cavities(cavities: Iterable, cfg: GCRConfig) -> None:
    """Print where every cavity axis is at both ends of the active length.

    Run once per geometry: it tells you which index sits on which azimuth (so
    you can pick the zoom target) and whether the plane you are about to cut
    on is really a mirror plane.
    """
    a = cavity_centres(cavities, 0.0)
    b = cavity_centres(cavities, cfg.L)
    print(f'{"i":>3s} {"x(z=0)":>9s} {"y(z=0)":>9s} {"x(z=L)":>9s} '
          f'{"y(z=L)":>9s} {"r":>8s} {"azimuth":>9s}')
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(a, b)):
        r = float(np.hypot(x1, y1))
        az = float(np.degrees(np.arctan2(y1, x1))) % 360.0
        print(f'{i:>3d} {x0:9.3f} {y0:9.3f} {x1:9.3f} {y1:9.3f} '
              f'{r:8.3f} {az:9.2f}')


def _pick_axial_basis(cavities: Iterable, cfg: GCRConfig,
                      tol: float = 1.0) -> str:
    """Choose 'xz' or 'yz' so the slice plane CONTAINS a ring-cavity axis.

    OpenMC slice plots are axis-aligned only -- there is no oblique basis.  If
    the ring sits at azimuths 0, 60, ... the plane y = 0 ('xz') contains two
    cavity axes while x = 0 cuts through solid moderator; at 30, 90, ... it is
    the other way round.
    """
    ring = [(x, y) for x, y in cavity_centres(cavities, cfg.L / 2)
            if np.hypot(x, y) > tol]
    if not ring:
        return 'yz'
    if any(abs(y) < tol for _, y in ring):
        return 'xz'
    if any(abs(x) < tol for x, _ in ring):
        return 'yz'
    print('WARNING: no ring-cavity axis lies in x=0 or y=0; the axial slice '
          'will cut cavities obliquely.')
    return 'yz'


# ===========================================================================
# 3.  Plot groups -- ONE COMPONENT PER FIGURE
# ===========================================================================

@dataclass(frozen=True)
class PlotGroup:
    """One highlighted component -> one radial and one axial figure.

    roles       role(s) coloured in this group's figures.  Only the fuel has
                more than one, because inner and outer zone must appear
                together for the radial build to read.
    scales      'core' (whole reactor) and/or 'cavity' (single-cavity zoom).
    zoom_width  cavity-window width in cm, overriding the global default.
                Smaller = tighter crop at the SAME pixel pitch; raise
                zoom_px_per_cm if you want finer detail rather than a
                tighter crop.
    zoom_offset (transverse, axial) shift of the cavity window off the cavity
                axis, cm.  Use it for a true detail inset, e.g.
                zoom_width=26, zoom_offset=(31, 0) puts a 26 cm box on the
                wall/duct/liner stack instead of the whole cavity.
    shade       False renders every material of the role in the IDENTICAL
                colour, so a tube wall and the coolant inside it read as one
                object.  True spreads them over a lightness family.
    """
    name: str
    roles: Tuple[str, ...]
    scales: Tuple[str, ...] = ('core',)
    zoom_width: Optional[float] = None
    zoom_offset: Tuple[float, float] = (0.0, 0.0)
    shade: bool = True


DEFAULT_GROUPS: Tuple[PlotGroup, ...] = (
    PlotGroup('pressure_shell',   ('pressure_shell',),                 ('core',)),
    PlotGroup('graphite',         ('graphite',),                       ('core',)),
    PlotGroup('beo',              ('beo',),                            ('core',)),
    PlotGroup('propellant',       ('propellant',),                     ('core',)),
    PlotGroup('fuel',             ('fuel_inner', 'fuel_outer', 'fuel'), ('core',)),
    # Neon buffer: one material, so flat colour and no ramp, but the same
    # whole-core treatment as the fuel -- the annulus is ~3.6 cm thick, which
    # is ~25 px at the core pitch, so it does not need the cavity zoom.
    PlotGroup('neon',             ('neon',),                           ('core',)),
    # Liner and transparent wall are millimetre-scale: cavity zoom only, and
    # tighter than the full hexagonal cell.  shade=False so tube wall and the
    # coolant inside it come out the same colour.
    PlotGroup('liner',            ('liner',),            ('cavity',),
              zoom_width=74.0, shade=False),
    PlotGroup('transparent_wall', ('transparent_wall',), ('cavity',),
              zoom_width=74.0, shade=False),
)

#: Same components, every one at both scales.  Pass groups=ALL_SCALE_GROUPS.
ALL_SCALE_GROUPS: Tuple[PlotGroup, ...] = tuple(
    replace(g, scales=('core', 'cavity')) for g in DEFAULT_GROUPS)


def _plot_from_window(win: Window, colours: dict, filename: str,
                      color_by: str = 'material',
                      background: RGB = (255, 255, 255)) -> openmc.Plot:
    p = openmc.Plot()
    p.basis = win.basis
    p.origin = win.origin
    p.width = win.width
    p.pixels = win.pixels
    p.color_by = color_by
    p.colors = colours
    p.background = background      # OpenMC's default is black; white prints better
    p.filename = filename
    return p


def _warn_resolution(win: Window, name: str, thinnest_cm: float = 0.06) -> None:
    """The liner tube wall is ~0.6 mm (1.24 cm OD, 0.9 ID/OD ratio).  OpenMC
    samples one ray per pixel, so below ~3 px across a feature it flickers or
    vanishes and you conclude the geometry is broken."""
    if win.cm_per_px * 3.0 > thinnest_cm:
        print(f'  NOTE [{name}]: {win.cm_per_px * 10:.2f} mm/px resolves a '
              f'{thinnest_cm * 10:.1f} mm feature with '
              f'{thinnest_cm / win.cm_per_px:.1f} px; use px_per_cm >= '
              f'{3.0 / thinnest_cm:.0f} to see the liner walls.')


def build_geometry_plots(gcr, figures_dir: str, tag: str = '',
                         groups: Sequence[PlotGroup] = DEFAULT_GROUPS,
                         # -- whole-core windows -------------------------------
                         px_per_cm: float = 7.0,
                         radial_margin: float = 40.0,
                         axial_margin_r: float = 20.0,
                         axial_margin_z: float = 10.0,
                         radial_half: Optional[str] = '+y',
                         axial_half: Optional[str] = '+',
                         axial_basis: str = 'auto',
                         # -- cavity zoom --------------------------------------
                         cavities: Sequence = (),
                         zoom_cavity: int = 1,
                         zoom_centre: Optional[Tuple[float, float]] = None,
                         zoom_width: float = 92.0,
                         zoom_px_per_cm: float = 50.0,
                         zoom_axial_px_per_cm: Optional[float] = 20.0,
                         zoom_axial_z_range: Optional[Tuple[float, float]] = None,
                         zoom_radial_half: Optional[str] = None,
                         zoom_axial_half: Optional[str] = None,
                         # -- colour -------------------------------------------
                         color_by: str = 'material',
                         rules: Optional[Sequence[MaterialRule]] = None,
                         pool_fuel_temperature: bool = False,
                         verbose: bool = True):
    """Build the per-component figure set for ONE case.

    Returns (plots, schemes_by_group, windows).  Filenames are
    ``{tag}_{group}_{view}`` with view in
    {radial, axial, cavity_radial, cavity_axial}.
    """
    cfg = gcr.config
    prefix = f'{tag}_' if tag else ''
    figures_dir = os.path.abspath(figures_dir)
    os.makedirs(figures_dir, exist_ok=True)
    if not cavities:
        cavities = getattr(gcr, 'cavities', ()) or ()

    # ---- whole-core windows, shared by every group ------------------------
    base = default_windows(cfg, px_per_cm, radial_margin,
                           axial_margin_r, axial_margin_z)
    windows: Dict[str, Window] = {'radial': half_window(base['radial'], radial_half)}

    w_ax = base['axial']
    if axial_basis == 'auto' and cavities:
        b = _pick_axial_basis(cavities, cfg)
        if b != w_ax.basis:
            w_ax = replace(w_ax, basis=b,
                           label=f'{"y" if b == "xz" else "x"} = 0')
    elif axial_basis in _BASIS_AXES:
        w_ax = replace(w_ax, basis=axial_basis)
    # Only the SIGN of axial_half is used: the transverse axis follows from
    # the basis, and z is never halved (you always want the full length).
    if axial_half:
        w_ax = half_window(w_ax, axial_half[0] + _BASIS_AXES[w_ax.basis][0])
    windows['axial'] = w_ax

    # ---- cavity windows, one pair per distinct zoom width/offset ----------
    cav_windows: Dict[tuple, Tuple[Window, Window]] = {}
    if any('cavity' in g.scales for g in groups):
        z_mid = cfg.L / 2
        if zoom_centre is not None:
            xc, yc = zoom_centre
        else:
            centres = cavity_centres(cavities, z_mid)
            if not centres:
                raise ValueError('cavity-scale groups requested but no '
                                 'cavities given: pass cavities=... or '
                                 'zoom_centre=(x, y)')
            xc, yc = centres[int(zoom_cavity)]
        # The axial slice plane must contain THIS cavity's axis: cut on
        # whichever of x, y is ~0 for it.
        if abs(yc) <= abs(xc):
            zb, ax_off = 'xz', (xc, 0.0)
        else:
            zb, ax_off = 'yz', (0.0, yc)
        # 'transverse' = the in-plane axis of the axial window; for a ring
        # cavity that is the machine radial direction, so a positive
        # zoom_offset[0] walks outwards through wall -> duct -> liner -> BeO.
        t_idx = _AXIS_INDEX[_BASIS_AXES[zb][0]]

        if zoom_axial_z_range is not None:
            z_lo, z_hi = zoom_axial_z_range
            z_c, z_h = 0.5 * (z_lo + z_hi), (z_hi - z_lo)
        else:
            z_c, z_h = w_ax.origin[2], base['axial'].width[1]

        for grp in groups:
            if 'cavity' not in grp.scales:
                continue
            gw = grp.zoom_width if grp.zoom_width is not None else zoom_width
            key = (gw, grp.zoom_offset)
            if key in cav_windows:
                continue
            dt, dz = grp.zoom_offset

            r_org = [xc, yc, z_mid]
            r_org[t_idx] += dt
            w_zr = Window('xy', tuple(r_org), (gw, gw), zoom_px_per_cm,
                          f'cavity {zoom_cavity}, z = {z_mid:.1f} cm')
            w_zr = half_window(w_zr, zoom_radial_half)

            a_org = [ax_off[0], ax_off[1], z_c + dz]
            a_org[t_idx] += dt
            w_za = Window(zb, tuple(a_org), (gw, z_h),
                          zoom_axial_px_per_cm or zoom_px_per_cm,
                          f'cavity {zoom_cavity}, '
                          f'{"y" if zb == "xz" else "x"} = 0')
            if zoom_axial_half:
                w_za = half_window(w_za, zoom_axial_half[0] + _BASIS_AXES[zb][0])
            cav_windows[key] = (w_zr, w_za)

    if verbose:
        print('Plot windows:')
        for name, w in windows.items():
            print(f'  {name:<16s} {w.describe()}')
            _warn_resolution(w, name)
        for (gw, off), (a, b_) in cav_windows.items():
            print(f'  cavity w={gw:.0f} off={off}')
            print(f'    radial       {a.describe()}')
            print(f'    axial        {b_.describe()}')
            _warn_resolution(a, f'cavity_radial w={gw:.0f}')
            _warn_resolution(b_, f'cavity_axial w={gw:.0f}')

    # ---- one scheme, two plots, per group ---------------------------------
    pool = [('fuel_inner', 'fuel_outer')] if pool_fuel_temperature else []
    plots: List[openmc.Plot] = []
    schemes: Dict[str, ColourScheme] = {}
    total_px = 0

    for grp in groups:
        scheme = build_colour_scheme(cfg, gcr.materials, highlight=grp.roles,
                                     rules=rules, pool_roles=pool,
                                     shade_flat_roles=grp.shade, verbose=False)
        if scheme.is_empty():
            if verbose:
                print(f'  SKIP {grp.name}: no material carries role(s) '
                      f'{grp.roles} -- add a MaterialRule or drop the group')
            continue
        schemes[grp.name] = scheme
        colours = (cell_colour_map(gcr, scheme) if color_by == 'cell'
                   else scheme.colours)

        views = []
        if 'core' in grp.scales:
            views += [('radial', windows['radial']), ('axial', windows['axial'])]
        if 'cavity' in grp.scales:
            gw = grp.zoom_width if grp.zoom_width is not None else zoom_width
            wr, wa = cav_windows[(gw, grp.zoom_offset)]
            views += [('cavity_radial', wr), ('cavity_axial', wa)]

        for view, w in views:
            plots.append(_plot_from_window(
                w, colours,
                os.path.join(figures_dir, f'{prefix}{grp.name}_{view}'),
                color_by))
            total_px += w.pixels[0] * w.pixels[1]

    if verbose:
        n_grad = sum(len(s.gradients) for s in schemes.values())
        print(f'{len(plots)} plots over {len(schemes)} components '
              f'({n_grad} temperature ramps), {total_px / 1e6:.0f} Mpx total. '
              f'Every plot is ray-traced separately: lower px_per_cm / '
              f'zoom_px_per_cm if this is slow.')

    return plots, schemes, windows


def export_geometry_plots(gcr, figures_dir: str = 'figures',
                          include_voxel: bool = False,
                          tag: str = '', legend: bool = True, **kwargs):
    """Export plots.xml and run OpenMC in plotting mode.

    Every figure goes into ONE plots.xml and ONE openmc.plot_geometry call, so
    the geometry is parsed once per case rather than once per figure.

    IFP tallying is not allowed in plot mode, so it is temporarily disabled in
    settings and restored afterwards -- the same dance as before.
    """
    plots, schemes, windows = build_geometry_plots(gcr, figures_dir, tag=tag,
                                                   **kwargs)

    if include_voxel:
        cfg = gcr.config
        pv = openmc.Plot()
        pv.type = 'voxel'
        pv.origin = (72.5, 0.0, cfg.L / 2 + cfg.L_conv / 2)
        pv.width = (145, 320, 290)
        pv.pixels = (290, 640, 580)
        pv.color_by = 'material'
        pv.colors = material_colour_map(cfg, gcr.materials)
        pv.filename = os.path.join(os.path.abspath(figures_dir),
                                   f'{tag + "_" if tag else ""}half_reactor')
        plots.append(pv)

    openmc.Plots(plots).export_to_xml(os.path.join(gcr.output_dir, 'plots.xml'))

    saved_ifp = getattr(gcr.settings, 'ifp_n_generation', None)
    gcr.settings.ifp_n_generation = None
    gcr.settings.export_to_xml(os.path.join(gcr.output_dir, 'settings.xml'))
    try:
        openmc.plot_geometry(cwd=gcr.output_dir)
    finally:
        gcr.settings.ifp_n_generation = saved_ifp
        gcr.settings.export_to_xml(os.path.join(gcr.output_dir, 'settings.xml'))

    if legend:
        plot_colour_legend(schemes, figures_dir=figures_dir, tag=tag)

    return plots, schemes, windows


# ===========================================================================
# 4.  Legend -- OpenMC PNGs carry no key, so draw one
# ===========================================================================

_ROLE_LABELS = {
    'pressure_shell':   'Pressure shells (+ inter-shell H$_2$)',
    'graphite':         'Graphite moderator',
    'beo':              'BeO moderator / reflector',
    'liner':            'Liner tubes (Be + coolant)',
    'transparent_wall': 'Transparent wall (SiO$_2$ + tori coolant)',
    'neon':             'Neon buffer gas',
    'propellant':       'H$_2$ propellant, duct only',
    'fuel':             'Fuel',
    'fuel_inner':       'Fuel, inner zone',
    'fuel_outer':       'Fuel, outer zone',
}


def plot_colour_legend(schemes: Dict[str, ColourScheme],
                       figures_dir: str = 'figures', tag: str = '',
                       save: bool = True):
    """One legend covering the whole figure set.

    Flat components appear as swatches; gradient components get a colourbar
    annotated with the ACTUAL material temperatures, so a reader can read a
    fuel-zone or duct temperature off the figure.
    """
    flats, grads = [], []
    for gname, s in schemes.items():
        for role in s.highlighted:
            if role in s.gradients:
                grads.append((role, s.gradients[role]))
            elif role in s.flats:
                flats.append((role, s.flats[role]))

    fig, axes = plt.subplots(
        1 + len(grads), 1,
        figsize=(6.8, 0.36 * max(len(flats), 1) + 1.05 * len(grads) + 0.7),
        gridspec_kw={'height_ratios': [max(len(flats), 1)] + [1.0] * len(grads)})
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.axis('off')
    handles = []
    for role, shades in flats:
        mid = np.array(shades[len(shades) // 2]) / 255.0
        lab = _ROLE_LABELS.get(role, role)
        if len(set(shades)) > 1:
            lab += f' ({len(shades)} shades)'
        handles.append(mpatches.Patch(facecolor=mid, edgecolor='0.3', label=lab))
    handles.append(mpatches.Patch(facecolor='0.6', edgecolor='0.3',
                                  label='Everything else (greyscale)'))
    ax.legend(handles=handles, loc='center left', frameon=False)

    for ax, (role, info) in zip(axes[1:], grads):
        c0 = np.array(info['ends'][0]) / 255.0
        c1 = np.array(info['ends'][1]) / 255.0
        cmap = LinearSegmentedColormap.from_list(role, [c0, c1])
        rng = info['range']
        if info['basis'] == 'temperature' and rng:
            norm = Normalize(vmin=rng[0], vmax=rng[1])
            label = f'{_ROLE_LABELS.get(role, role)} temperature (K)'
        else:
            norm = Normalize(vmin=0, vmax=max(len(info['keys']) - 1, 1))
            label = f'{_ROLE_LABELS.get(role, role)} (layer index)'
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax,
                          orientation='horizontal')
        cb.set_label(label)
        if info['basis'] == 'temperature' and rng:
            T = info['temperatures']
            cb.ax.plot(T, np.full_like(T, 0.5), 'k|', ms=9, mew=1.0)

    fig.tight_layout()
    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out = os.path.join(os.path.abspath(figures_dir),
                           f'{tag + "_" if tag else ""}colour_legend.pdf')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'Saved -> {out}')
    return fig

# ===========================================================================
# 5.  Statepoint post-processing
# ===========================================================================
#
# THE AXIS-ORDERING TRAP, handled once and for all.
#
# OpenMC orders mesh bins with x varying fastest, then y, then z, so a flat
# mesh-filter axis must be reshaped as (nz, ny, nx) -- REVERSED with respect
# to mesh.dimension.  plot_power_distribution and plot_axial_flux always did
# this; plot_midplane_flux did not (it reshaped straight to (nx, ny, ...)).
# Because the midplane mesh is square, nx == ny, the wrong reshape raised no
# error: it silently transposed the map, i.e. reflected it about the line
# y = x, which maps azimuth phi -> 90 deg - phi.  With cavities on a hex ring
# that turns azimuths {30, 90, 150, 210, 270, 330} into {60, 0, 300, 240,
# 180, 120} -- still a perfectly plausible hexagon, 30 deg out of phase with
# the '+' markers, which were computed from the placements and were right all
# along.  It also swapped the two line-out panels.
#
# Everything below goes through mesh_data(), so there is now one place to get
# this wrong instead of four.

E_JOULE = 1.602176634e-19          # J per eV, exact by SI definition


def _filter_axis(tally, filter_type):
    for i, f in enumerate(tally.filters):
        if isinstance(f, filter_type):
            return i, f
    return None, None


def mesh_data(tally: openmc.Tally, mesh: openmc.RegularMesh,
              value: str = 'mean') -> np.ndarray:
    """Tally data as (nx, ny, nz, *remaining filters, nuclides, scores).

    Undoes OpenMC's (nz, ny, nx) bin ordering.  Use this instead of calling
    get_reshaped_data().reshape(...) at the call site.
    """
    nx, ny, nz = mesh.dimension
    axis, _ = _filter_axis(tally, openmc.MeshFilter)
    if axis is None:
        raise ValueError(f'tally {tally.name!r} has no MeshFilter')
    data = np.moveaxis(tally.get_reshaped_data(value=value), axis, 0)
    data = data.reshape(nz, ny, nx, *data.shape[1:])
    return np.transpose(data, (2, 1, 0) + tuple(range(3, data.ndim)))


def energy_edges(tally: openmc.Tally, meta: dict = None) -> np.ndarray:
    """Bin edges [eV] of the tally's EnergyFilter, or reconstructed from meta.

    Reading them from the statepoint rather than assuming three groups means
    that adding a boundary to the EnergyFilter (see plot_midplane_flux's
    cumulative_cut_eV) makes the extra group appear in every figure with no
    further code change.
    """
    _, ef = _filter_axis(tally, openmc.EnergyFilter)
    if ef is not None:
        return np.asarray(ef.values, dtype=float)
    if meta is None:
        raise ValueError(f'tally {tally.name!r} has no EnergyFilter and no '
                         f'meta was given')
    return np.array([0.0, meta['thermal_cutoff'], meta['epithermal_cutoff'],
                     2.0e7], dtype=float)


def group_labels(edges: np.ndarray) -> list:
    """Legend labels for an arbitrary group structure.

    The three-group case reproduces the original wording exactly, so existing
    figures are unchanged.
    """
    n = len(edges) - 1
    if n == 3:
        return [f'Thermal (E < {fmt_E(edges[1])})',
                f'Epithermal ({fmt_E(edges[1])} < E < {fmt_E(edges[2])})',
                f'Fast (E > {fmt_E(edges[2])})']
    out = [f'Thermal (E < {fmt_E(edges[1])})']
    for g in range(1, n - 1):
        out.append(f'{fmt_E(edges[g])} < E < {fmt_E(edges[g + 1])}')
    out.append(f'Fast (E > {fmt_E(edges[-2])})')
    return out

def collapse_to(flux: np.ndarray, edges: np.ndarray, keep) -> tuple:
    edges = np.asarray(edges, dtype=float)
    want = np.unique(np.concatenate(
        [[edges[0]], np.atleast_1d(np.asarray(keep, dtype=float)), [edges[-1]]]))

    idx = []
    for e in want:
        j = int(np.argmin(np.abs(edges - e)))
        if abs(edges[j] - e) > 1e-6 * max(e, 1.0):
            raise ValueError(
                f'{e:g} eV is not a bin edge of this tally. Edges present: '
                f'{np.array2string(edges, precision=4)}. Add it to '
                f'energy_bounds and re-run.')
        idx.append(j)
    idx = sorted(set(idx))

    out = np.stack([flux[..., a:b].sum(axis=-1)
                    for a, b in zip(idx[:-1], idx[1:])], axis=-1)
    return out, edges[idx]

_GROUP_CMAPS = ['viridis', 'cividis', 'inferno', 'magma', 'plasma', 'YlGnBu']
_GROUP_COLOURS = ['C0', 'C2', 'C3', 'C4', 'C5', 'C6']


def positive_half(coord, *series, include_zero: bool = True):
    """Trim a line-out to coord >= 0 (or > 0).

    Valid only where the cut is a true mirror plane; for the hex ring both
    x = 0 and y = 0 qualify.  Note the two line-outs are NOT equivalent cuts:
    with cavities at azimuth 30, 90, ..., the x = 0 line passes through two
    cavity axes while the y = 0 line passes between them, which is why one
    shows sharp peaks near |r| = 80 cm and the other does not.
    """
    coord = np.asarray(coord)
    m = coord >= 0 if include_zero else coord > 0
    return (coord[m],) + tuple(np.asarray(s)[m] for s in series)


def lineout_normalisation(curve_sets, mode):
    """Denominators for a set of line-out curves that SHARE one y axis.

    curve_sets: one entry per curve, each a list of the arrays that curve
                contributes to the figure (i.e. both panels).
    mode:       None     absolute values, logarithmic axis (default)
                'global' every curve divided by the single largest value
                         anywhere in the figure.  Relative group magnitudes
                         are preserved, so the spectral ordering stays
                         readable and only one curve reaches 1.0.
                'group'  each curve divided by its own maximum.  Every curve
                         peaks at 1.0, so SHAPES are comparable and relative
                         magnitudes are gone.

    Maxima are taken across both panels, never per panel: the panels share a
    y axis, so per-panel denominators would put two different scales on one
    axis and the comparison between them would be meaningless.

    Returns (denominators, common) where common is the single denominator for
    'global' (used in the axis label) and None otherwise.
    """
    if not mode:
        return [1.0] * len(curve_sets), None
    peaks = []
    for arrs in curve_sets:
        vals = [float(np.nanmax(a)) for a in arrs if np.size(a)]
        m = max(vals) if vals else 1.0
        peaks.append(m if m > 0 else 1.0)
    if mode == 'group':
        return peaks, None
    if mode == 'global':
        m = max(peaks)
        return [m] * len(curve_sets), m
    raise ValueError(f"lineout_norm must be None, 'global' or 'group', "
                     f"got {mode!r}")

def lineout_denominators(cuts, mode, scope='panel'):
    """Denominators for a two-panel line-out figure.

    cuts:  one entry per curve, each ((x, f_left), (y, f_right)).
    scope: 'figure' -> one denominator per curve, shared by both panels.
                       The panels stay directly comparable and may share a
                       y axis.  This was the only behaviour before.
           'panel'  -> each panel normalised on its own maxima, so both
                       fill their axes.  The panels then carry DIFFERENT
                       scales and must NOT share a y axis; the denominators
                       go in the axis labels so the ratio is recoverable.

    Returns (denom_left, denom_right, common_left, common_right).
    """
    if scope == 'figure':
        d, c = lineout_normalisation([[a[1], b[1]] for a, b in cuts], mode)
        return d, d, c, c
    if scope != 'panel':
        raise ValueError(f"scope must be 'figure' or 'panel', got {scope!r}")
    da, ca = lineout_normalisation([[a[1]] for a, _ in cuts], mode)
    db, cb = lineout_normalisation([[b[1]] for _, b in cuts], mode)
    return da, db, ca, cb

def _style_lineout(axes, mode, unit, common=None):
    commons = (list(common) if isinstance(common, (list, tuple))
               else [common] * len(axes))
    for ax, com in zip(axes, commons):
        if mode:
            ax.set_yscale('linear')
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, which='major', alpha=0.3)
            ax.set_ylabel(f'Flux / {com:.3e} {unit}' if com is not None
                          else 'Flux / curve maximum (-)')
        else:
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.3)
            ax.set_ylabel(f'Flux ({unit})')
        ax.legend()

# ---------------------------------------------------------------------------
# Power normalisation
# ---------------------------------------------------------------------------

def _power_normalisation(statepoint_path: str, power_W: float,
                         total_eV: float = None) -> float:
    """Source rate S [src/s] such that the tallied power equals power_W."""
    if total_eV is None:
        sp = openmc.StatePoint(statepoint_path)
        t = sp.get_tally(name='power_distribution')
        total_eV = float(t.get_reshaped_data(value='mean').sum())
    return power_W / (total_eV * E_JOULE)


def fission_q_to_power(values_eV, total_power_W: float, total_eV: float = None,
                       volumes_cm3=None, std_dev_eV=None, verbose: bool = True):
    """Convert a fission-q-recoverable tally from eV/source to watts.

    OpenMC scores fission-q-recoverable as eV released per source particle:

        P_i = t_i [eV/src] * S [src/s] * e [J/eV],  S = P_tot / (E_tot * e)
    =>  P_i = t_i * P_tot / E_tot

    so the elementary charge cancels and the conversion is a pure
    normalisation.

    ``total_eV`` MUST be the model-wide total, not the sum over this mesh.
    Defaulting to the mesh sum makes the map integrate to exactly P_tot by
    construction, silently redistributing whatever the mesh fails to capture
    -- with the ~4.95 % of cavity power lost to axial clipping at
    z = 182.8 cm every surviving cell comes out ~5 % high and the shortfall
    disappears from view.

    Returns (power_W, std_W), plus power density in W/cm3 if volumes given.
    """
    v = np.asarray(values_eV, dtype=float)
    if total_eV is None:
        total_eV = float(v.sum())
        if verbose:
            print('WARNING: total_eV defaulted to the mesh sum; energy '
                  'released outside the mesh is being folded into the cells '
                  'inside it.')
    k = float(total_power_W) / float(total_eV)          # W per (eV/src)
    power = v * k
    std = None if std_dev_eV is None else np.asarray(std_dev_eV, float) * k

    if verbose:
        S = float(total_power_W) / (float(total_eV) * E_JOULE)
        print(f'  source rate   {S:.4e} n/s     scale {k:.4e} W per eV/src')
        print(f'  mapped power  {power.sum() / 1e6:.4f} MW of '
              f'{total_power_W / 1e6:.4f} MW '
              f'({100 * power.sum() / total_power_W:.2f} %)')
        print('  NOTE: fission ENERGY RELEASE, not deposited power -- prompt '
              'gammas escape the fissioning cell.')

    if volumes_cm3 is None:
        return power, std
    vol = np.asarray(volumes_cm3, dtype=float)
    dens = np.divide(power, vol, out=np.zeros_like(power), where=vol > 0)
    return power, std, dens


def regular_mesh_volumes(mesh: openmc.RegularMesh) -> np.ndarray:
    """Per-cell volume [cm3] of a RegularMesh, shaped (nx, ny, nz)."""
    lo = np.asarray(mesh.lower_left, float)
    hi = np.asarray(mesh.upper_right, float)
    n = np.asarray(mesh.dimension, int)
    return np.full(tuple(n), float(np.prod((hi - lo) / n)))


def marker_phase_check(cavities, z, peaks) -> None:
    """Compare marker azimuths against measured flux-peak azimuths.

    The quickest way to catch a phase or handedness slip before a figure goes
    into the thesis: a mean offset of 30 deg is a half-pitch slip in the
    marker ring, 90 deg minus the azimuth is a transposed map.
    """
    def polar(pts):
        return sorted((float(np.degrees(np.arctan2(y, x))) % 360.0,
                       float(np.hypot(x, y))) for x, y in pts
                      if np.hypot(x, y) > 1.0)

    m, p = polar(cavity_centres(cavities, z)), polar(peaks)
    print(f'{"marker az":>11s} {"marker r":>9s} {"peak az":>9s} {"peak r":>8s}')
    for (ma, mr), (pa, pr) in zip(m, p):
        print(f'{ma:11.2f} {mr:9.2f} {pa:9.2f} {pr:8.2f}')
    if m and p:
        d = np.array([a for a, _ in m]) - np.array([a for a, _ in p])
        d = (d + 180.0) % 360.0 - 180.0
        print(f'mean azimuthal offset {d.mean():+.2f} deg')


# ---------------------------------------------------------------------------
# Power distribution
# ---------------------------------------------------------------------------

def plot_power_distribution(cfg: GCRConfig, mesh: openmc.RegularMesh,
                            statepoint_path: str, z_fraction: float = 0.5,
                            save: bool = True, figures_dir: str = 'figures',
                            power_W: float = 4.6e9, total_eV: float = None,
                            density: bool = True):
    """XY and YZ maps from the 'power_distribution' tally, in power units.

    power_W: total reactor power for the normalisation; None leaves the map
             in eV per source particle.
    density: True plots W/cm3 (the physically meaningful field, independent
             of mesh refinement); False plots W per voxel, which changes if
             you re-mesh.
    """
    sp = openmc.StatePoint(statepoint_path)
    tally = sp.get_tally(name='power_distribution')
    nx, ny, nz = mesh.dimension
    ll, ur = mesh.lower_left, mesh.upper_right

    q = mesh_data(tally, mesh)[..., 0, 0]          # (nx, ny, nz) eV/src
    q = q.reshape(nx, ny, nz)

    if power_W is not None:
        vol = regular_mesh_volumes(mesh) if density else None
        out = fission_q_to_power(q, power_W, total_eV=total_eV,
                                 volumes_cm3=vol)
        field = out[2] if density else out[0]
        if density:
            label = 'Fission energy release (W/cm$^3$)'
        else:
            field = field / 1e6
            label = 'Fission energy release (MW per voxel)'
        norm_label = f'P = {power_W * 1e-9:g} GW'
    else:
        field = q
        label = 'Fission-Q recoverable (eV/source particle)'
        norm_label = 'per source neutron'

    z_idx = int(z_fraction * nz)
    z_cm = ll[2] + (z_idx + 0.5) * (ur[2] - ll[2]) / nz
    xy_slice = field[:, :, z_idx].T                # (ny, nx) for imshow
    yz_slice = field[nx // 2, :, :].T              # (nz, ny), z vertical

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    im0 = axes[0].imshow(xy_slice, origin='lower',
                         extent=[ll[0], ur[0], ll[1], ur[1]],
                         cmap='hot', aspect='equal')
    plt.colorbar(im0, ax=axes[0], label=label)
    axes[0].set_title(f'XY power distribution  |  z = {z_cm:.1f} cm')
    axes[0].set_xlabel('x (cm)')
    axes[0].set_ylabel('y (cm)')

    # Fixed x index -> this is a Y-Z slice, not X-Z; the original title and
    # axis label said XZ / X while the extent was already y.
    im1 = axes[1].imshow(yz_slice, origin='lower',
                         extent=[ll[1], ur[1], ll[2], ur[2]],
                         cmap='hot', aspect='equal')
    plt.colorbar(im1, ax=axes[1], label=label)
    axes[1].set_title('YZ power distribution  |  x = 0')
    axes[1].set_xlabel('y (cm)')
    axes[1].set_ylabel('z (cm)')

    plt.suptitle(f'GCR power distribution ({norm_label})', fontsize=13, y=1.01)
    plt.tight_layout()
    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out_path = os.path.join(figures_dir, 'power_distribution.pdf')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out_path}')
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Total flux distribution
# ---------------------------------------------------------------------------

def plot_flux_distribution(cfg: GCRConfig, mesh: openmc.RegularMesh,
                           statepoint_path: str, cavities=(),
                           z_fraction: float = 0.5, save: bool = True,
                           figures_dir: str = 'figures'):
    """XY and XZ total-flux maps from the 'flux_distribution' tally."""
    sp = openmc.StatePoint(statepoint_path)
    tally = sp.get_tally(name='flux_distribution')
    nx, ny, nz = mesh.dimension
    ll, ur = mesh.lower_left, mesh.upper_right

    flux = mesh_data(tally, mesh)[..., 0, 0].reshape(nx, ny, nz)

    z_idx = int(z_fraction * nz)
    z_cm = ll[2] + (z_idx + 0.5) * (ur[2] - ll[2]) / nz
    xy_slice = flux[:, :, z_idx].T
    xz_slice = flux[:, ny // 2, :].T

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(xy_slice, origin='lower',
                         extent=[ll[0], ur[0], ll[1], ur[1]],
                         cmap='viridis', aspect='equal')
    plt.colorbar(im0, ax=axes[0], label='Flux (n/source particle*cm2)')
    axes[0].set_title(f'XY flux distribution  |  z = {z_cm:.1f} cm')
    axes[0].set_xlabel('x (cm)')
    axes[0].set_ylabel('y (cm)')

    # Markers at the axis position IN THIS PLANE.  The old code used
    # cavity.translation, which is the axis only at the placement reference
    # plane: the ring cavities are tilted and carry an axial offset, so their
    # axes move radially with z.
    for cx, cy in cavity_centres(cavities, z_cm):
        axes[0].plot(cx, cy, 'w+', markersize=8, markeredgewidth=1.5)

    im1 = axes[1].imshow(xz_slice, origin='lower',
                         extent=[ll[0], ur[0], ll[2], ur[2]],
                         cmap='viridis', aspect='auto')
    plt.colorbar(im1, ax=axes[1], label='Flux (n/source particle*cm2)')
    axes[1].set_title('XZ flux distribution  |  y = 0')
    axes[1].set_xlabel('x (cm)')
    axes[1].set_ylabel('z (cm)')

    plt.suptitle('GCR neutron flux distribution', fontsize=13, y=1.01)
    plt.tight_layout()
    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out = os.path.join(figures_dir, 'flux_distribution.pdf')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Midplane flux
# ---------------------------------------------------------------------------

def plot_midplane_flux(cfg: GCRConfig, mesh: openmc.RegularMesh, meta: dict,
                       statepoint_path: str, cavities=(), save: bool = True,
                       power_W: float = 4.6e9, figures_dir: str = 'figures',
                       half: bool = True, cumulative_cut_eV: float = 8.32,
                       lineout_norm: str = 'global', group_edges=None):
    """Group midplane flux maps + line-outs, normalised to reactor power.

    power_W:            None leaves the tally output in n/cm2/src.
    half:               line-outs show the positive side only (the geometry
                        is mirror-symmetric about both x = 0 and y = 0).
    cumulative_cut_eV:  if this energy is a bin edge of the tally's
                        EnergyFilter, a third figure is produced showing the
                        flux summed over every group below it.  Post-
                        processing cannot subdivide a bin, so with the usual
                        [0, 0.625, 1e5, 2e7] structure this edge does not
                        exist; add it in the tally definition,

                            openmc.EnergyFilter([0.0, 0.625, 8.32, 1e5, 2e7])

                        and re-run.  Every existing boundary is preserved, so
                        the three-group figures are unchanged and the new
                        group simply appears.  Pass None to skip.
    lineout_norm:       None keeps the absolute logarithmic line-outs.
                        'global' or 'group' switches them to a LINEAR axis
                        normalised to a maximum -- see lineout_normalisation.
    """
    sp = openmc.StatePoint(statepoint_path)
    tally = sp.get_tally(name='midplane_flux_groups')

    nx, ny, _ = mesh.dimension
    ll, ur = mesh.lower_left, mesh.upper_right
    edges = energy_edges(tally, meta)
    labels = group_labels(edges)
    n_g = len(labels)

    # (nx, ny, nz, n_E, n_nuc, n_score) -> (nx, ny, n_E), correctly oriented
    flux = mesh_data(tally, mesh)[..., 0, 0]
    flux = flux.reshape(nx, ny, -1, n_g)[:, :, 0, :]

    if group_eges is not None:
        flux, edges = collapse_to(flux, edges, group_edges)
        labels = group_labels(edges)
        n_g = len(labels)

    if power_W is not None:
        S = _power_normalisation(statepoint_path, power_W)
        flux = flux * S
        flux_unit = 'n*cm/s'
        norm_label = f'P = {power_W * 1e-9:g} GW'
    else:
        flux_unit = 'n/cm2/src'
        norm_label = 'per source neutron'

    extent = [ll[0], ur[0], ll[1], ur[1]]
    z_slice = 0.5 * (ll[2] + ur[2])
    z_label = f'z = {z_slice:.1f} cm, dz = {ur[2] - ll[2]:.1f} cm'
    centres = cavity_centres(cavities, z_slice)

    # -- Figure 1: heatmaps ---------------------------------------------------
    fig1, axes = plt.subplots(1, n_g, figsize=(6 * n_g, 6))
    axes = np.atleast_1d(axes)
    for g, (ax, label) in enumerate(zip(axes, labels)):
        im = ax.imshow(flux[:, :, g].T, origin='lower', extent=extent,
                       cmap=_GROUP_CMAPS[g % len(_GROUP_CMAPS)],
                       aspect='equal', interpolation='nearest')
        plt.colorbar(im, ax=ax, label=f'Flux ({flux_unit})')
        ax.set_title(label)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        for cx, cy in centres:
            ax.plot(cx, cy, 'w+', markersize=10, markeredgewidth=1.5)

    plt.suptitle(f'Midplane flux  ({z_label}, {norm_label})', y=1.02)
    plt.tight_layout()
    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out = os.path.join(figures_dir, 'midplane_flux_2D.pdf')
        fig1.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')

    # -- line-out coordinates -------------------------------------------------
    x_e = np.linspace(ll[0], ur[0], nx + 1)
    y_e = np.linspace(ll[1], ur[1], ny + 1)
    x_c = 0.5 * (x_e[:-1] + x_e[1:])
    y_c = 0.5 * (y_e[:-1] + y_e[1:])
    i_x0, j_y0 = int(np.argmin(np.abs(x_c))), int(np.argmin(np.abs(y_c)))

    def _lineouts(values):
        """(x, along y=0), (y, along x=0), trimmed to the positive side."""
        a = (x_c, values[:, j_y0])
        b = (y_c, values[i_x0, :])
        return (positive_half(*a), positive_half(*b)) if half else (a, b)

    # -- Figure 2: per-group line-outs ---------------------------------------
    fig2, axes = plt.subplots(
        1, 2, figsize=(13, 5),
        sharey=(lineout_norm_scope == 'figure' or not lineout_norm))
    cuts = [_lineouts(flux[:, :, g]) for g in range(n_g)]
    den_a, den_b, com_a, com_b = lineout_denominators(
        cuts, lineout_norm, lineout_norm_scope)
    for g, (label, ((xa, fa), (yb, fb))) in enumerate(zip(labels, cuts)):
        c = _GROUP_COLOURS[g % len(_GROUP_COLOURS)]
        # In 'group' mode the denominator differs per curve, so carry it in
        # the legend -- otherwise the absolute magnitudes vanish completely.
        # With scope='panel' the two panels differ as well, hence two labels.
        lab_a = label if com_a is not None or not lineout_norm else \
            f'{label}  (max {den_a[g]:.2e} {flux_unit})'
        lab_b = label if com_b is not None or not lineout_norm else \
            f'{label}  (max {den_b[g]:.2e} {flux_unit})'
        axes[0].plot(xa, fa / den_a[g], color=c, label=lab_a)
        axes[1].plot(yb, fb / den_b[g], color=c, label=lab_b)
    axes[0].set_xlabel('x (cm)')
    axes[0].set_title('Line-out along y = 0')
    axes[1].set_xlabel('y (cm)')
    axes[1].set_title('Line-out along x = 0')
    _style_lineout(axes, lineout_norm, flux_unit, [com_a, com_b])
    plt.tight_layout()
    if save:
        out = os.path.join(figures_dir, 'midplane_flux_lineout.pdf')
        fig2.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')

    # -- Figure 3: cumulative flux below a cut --------------------------------
    fig3 = None
    if cumulative_cut_eV is not None:
        j = int(np.argmin(np.abs(edges - cumulative_cut_eV)))
        if abs(edges[j] - cumulative_cut_eV) > 1e-6 * max(cumulative_cut_eV, 1.0):
            print(f'Skipping the E < {fmt_E(cumulative_cut_eV)} line-out: '
                  f'{cumulative_cut_eV:g} eV is not a bin edge of '
                  f'midplane_flux_groups. Edges present: '
                  f'{np.array2string(edges, precision=4)}. Add it to the '
                  f'EnergyFilter and re-run.')
        else:
            below = flux[:, :, :j].sum(axis=2)
            cut_s = fmt_E(cumulative_cut_eV)
            fig3, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
            cuts = [_lineouts(below)] + [_lineouts(flux[:, :, g])
                                         for g in range(j)]
            denom, common = lineout_normalisation(
                [[a[1], b[1]] for a, b in cuts], lineout_norm)
            (xa, ca), (yb, cb) = cuts[0]
            axes[0].plot(xa, ca / denom[0], color='k', lw=2,
                         label=f'E < {cut_s} (total)')
            axes[1].plot(yb, cb / denom[0], color='k', lw=2,
                         label=f'E < {cut_s} (total)')
            for g in range(j):                     # contributing groups
                (xa, fa), (yb, fb) = cuts[g + 1]
                d = denom[g + 1]
                c = _GROUP_COLOURS[g % len(_GROUP_COLOURS)]
                axes[0].plot(xa, fa / d, color=c, ls='--', lw=1, label=labels[g])
                axes[1].plot(yb, fb / d, color=c, ls='--', lw=1, label=labels[g])
            axes[0].set_xlabel('x (cm)')
            axes[0].set_title('Line-out along y = 0')
            axes[1].set_xlabel('y (cm)')
            axes[1].set_title('Line-out along x = 0')
            _style_lineout(axes, lineout_norm, flux_unit, common)
            plt.suptitle(f'Midplane flux below {cut_s}  ({norm_label})', y=1.02)
            plt.tight_layout()
            if save:
                tag = f'{cumulative_cut_eV:g}eV'.replace('.', 'p')
                out = os.path.join(figures_dir,
                                   f'midplane_flux_lineout_below_{tag}.pdf')
                fig3.savefig(out, dpi=150, bbox_inches='tight')
                print(f'Saved -> {out}')

    plt.show()
    return (fig1, fig2, fig3) if fig3 is not None else (fig1, fig2)


# ---------------------------------------------------------------------------
# Axial flux
# ---------------------------------------------------------------------------
def plot_axial_flux(cfg: GCRConfig, mesh: openmc.RegularMesh, meta: dict,
                    statepoint_path: str, save: bool = True,
                    power_W: float = 4.6e9, figures_dir: str = 'figures',
                    half: bool = True, lineout_norm: str = None,
                    group_edges=None,                        # <-- NEW
                    lineout_norm_scope: str = 'panel'):      # <-- NEW
    """Group axial flux on the x = 0 slab + line-outs, power-normalised.

    The mesh is [1, ny, nz], i.e. thin in X, so the slab lies in the x = 0
    plane and its in-plane coordinates are y and z.  The original labelled
    this 'y = 0' with an 'x (cm)' axis; both are corrected here.

    group_edges:         interior energy edges to KEEP, e.g. [1.86, 1e5].
                         Every group below the lowest kept edge is summed
                         into one thermal group.  Each value must already be
                         an edge of the tally's EnergyFilter -- post-
                         processing merges bins, it cannot split them.
                         None leaves the tally's own group structure.
    lineout_norm_scope:  'panel'  normalises the axial and radial panels
                         independently, so each fills its own axis and the
                         two carry different scales (they do not share a y
                         axis, and each denominator appears in its own
                         label).  'figure' restores the old shared-maximum
                         behaviour, where the axial panel sets the scale for
                         both.  Irrelevant when lineout_norm is None.
    """
    sp = openmc.StatePoint(statepoint_path)
    tally = sp.get_tally(name='axial_flux_groups')

    nx, ny, nz = mesh.dimension
    ll, ur = mesh.lower_left, mesh.upper_right
    edges = energy_edges(tally, meta)
    labels = group_labels(edges)
    n_g = len(labels)

    flux = mesh_data(tally, mesh)[..., 0, 0]       # (nx, ny, nz, n_E)
    flux = flux.reshape(nx, ny, nz, n_g)[0]        # -> (ny, nz, n_E)

    # Collapse AFTER the reshape -- the reshape needs the tally's own n_g.
    if group_edges is not None:                                  # <-- NEW
        flux, edges = collapse_to(flux, edges, group_edges)      # <-- NEW
        labels = group_labels(edges)                             # <-- NEW
        n_g = len(labels)                                        # <-- NEW

    if power_W is not None:
        flux = flux * _power_normalisation(statepoint_path, power_W)
        flux_unit = 'n*cm/s'
        norm_label = f'P = {power_W * 1e-9:g} GW'
    else:
        flux_unit = 'n/cm2/src'
        norm_label = 'per source neutron'

    L = cfg.L
    slab_label = f'x = 0, dx = {ur[0] - ll[0]:.1f} cm'

    # -- Figure 1: heatmaps, z vertical --------------------------------------
    extent = [ll[1], ur[1], ur[2], ll[2]]
    panel_h = 6.5
    panel_w = max(6.0, panel_h * (ur[1] - ll[1]) / (ur[2] - ll[2]))
    fig1, axes = plt.subplots(1, n_g, figsize=(n_g * panel_w + 3, panel_h + 1.5),
                              constrained_layout=True)
    axes = np.atleast_1d(axes)
    for g, (ax, label) in enumerate(zip(axes, labels)):
        im = ax.imshow(flux[:, :, g].T, origin='upper', extent=extent,
                       cmap=_GROUP_CMAPS[g % len(_GROUP_CMAPS)],
                       aspect='equal', interpolation='nearest')
        plt.colorbar(im, ax=ax, label=f'Flux ({flux_unit})', shrink=0.85)
        ax.set_title(label)
        ax.set_xlabel('y (cm)')
        ax.set_ylabel('z (cm)')
        if ll[2] <= 0.0 <= ur[2]:
            ax.axhline(0.0, color='w', lw=0.6, ls='--', alpha=0.6)
        if ll[2] <= L <= ur[2]:
            ax.axhline(L, color='w', lw=0.6, ls='--', alpha=0.6)

    fig1.suptitle(f'Axial flux ({slab_label}, {norm_label})')
    if save:
        os.makedirs(figures_dir, exist_ok=True)
        out = os.path.join(figures_dir, 'axial_flux_2D.pdf')
        fig1.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')

    # -- Figure 2: line-outs --------------------------------------------------
    y_e = np.linspace(ll[1], ur[1], ny + 1)
    z_e = np.linspace(ll[2], ur[2], nz + 1)
    y_c = 0.5 * (y_e[:-1] + y_e[1:])
    z_c = 0.5 * (z_e[:-1] + z_e[1:])
    j_y0 = int(np.argmin(np.abs(y_c)))
    k_zL2 = int(np.argmin(np.abs(z_c - L / 2)))

    # Panels share a y axis only when they share a denominator.       <-- CHANGED
    fig2, axes = plt.subplots(
        1, 2, figsize=(13, 5),
        sharey=(lineout_norm_scope == 'figure' or not lineout_norm))
    # Axial profile: z is NOT halved -- the geometry is not symmetric about
    # the midplane (inlet, nozzle, the -4.6 cm ring offset).
    cuts = [((z_c, flux[j_y0, :, g]),
             positive_half(y_c, flux[:, k_zL2, g]) if half
             else (y_c, flux[:, k_zL2, g])) for g in range(n_g)]
    den_a, den_b, com_a, com_b = lineout_denominators(          # <-- CHANGED
        cuts, lineout_norm, lineout_norm_scope)
    for g, (label, ((za, fa), (yb, fb))) in enumerate(zip(labels, cuts)):
        c = _GROUP_COLOURS[g % len(_GROUP_COLOURS)]
        # In 'group' mode the denominator differs per curve, and with
        # scope='panel' it differs per panel too, so each panel needs its
        # own legend text -- otherwise the magnitudes vanish entirely.
        lab_a = label if com_a is not None or not lineout_norm else \
            f'{label}  (max {den_a[g]:.2e} {flux_unit})'
        lab_b = label if com_b is not None or not lineout_norm else \
            f'{label}  (max {den_b[g]:.2e} {flux_unit})'
        axes[0].plot(za, fa / den_a[g], color=c, label=lab_a)   # <-- CHANGED
        axes[1].plot(yb, fb / den_b[g], color=c, label=lab_b)   # <-- CHANGED
    axes[0].set_xlabel('z (cm)')
    axes[0].set_title('Axial profile at y = 0')
    if ll[2] <= 0.0 <= ur[2]:
        axes[0].axvline(0.0, color='k', lw=0.5, ls='--')
    if ll[2] <= L <= ur[2]:
        axes[0].axvline(L, color='k', lw=0.5, ls='--')
    axes[1].set_xlabel('y (cm)')
    axes[1].set_title(f'Radial profile at z = L/2 = {L / 2:.1f} cm')
    _style_lineout(axes, lineout_norm, flux_unit, [com_a, com_b])  # <-- CHANGED
    plt.tight_layout()
    if save:
        out = os.path.join(figures_dir, 'axial_flux_lineout.pdf')
        fig2.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')
    plt.show()
    return fig1, fig2

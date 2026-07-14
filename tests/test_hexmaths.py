"""Pure-mathematics tests for the hexagon/ridge/placement geometry.

These run in milliseconds and need NO OpenMC and NO nuclear data --
just numpy.  They encode the facts established during the tie-rod
investigation, so any future change that breaks them is caught in
seconds instead of after a cluster run:

  * at zero tilt, all six hexagon walls sit exactly at the apothem;
  * the six INNER ridge lines (central cavity + two neighbours) coincide
    to machine precision across all three owning cavities;
  * the six OUTER peripheral-peripheral vertices do NOT coincide exactly
    (tapered hexagons cannot tile) but agree within the documented ~3 mm
    -- inside the tie-rod merge tolerance;
  * merging by line yields exactly 24 unique ridges (6 triple-shared,
    6 double-shared, 12 single);
  * a config survives a JSON round trip.

Run with:  pytest tests/ -v
"""

import json
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcr.config import GCRConfig
from gcr.geometry.hexmaths import (HEX_ANGLES, cavity_placements,
                                   hex_plane_coefficients, ridge_vertices)

_SIN60 = np.sin(np.pi / 3)


@pytest.fixture
def cfg():
    return GCRConfig()


def _global_ridge_segments(cfg):
    """All 42 ridge segments (6 per cavity) in global coordinates."""
    L = cfg.L
    segments = []
    for cav_idx, placement in enumerate(cavity_placements(cfg)):
        R = placement.rotation
        t = np.asarray(placement.translation)
        for (x0, y0, xL, yL) in ridge_vertices(cfg.tilt, cfg.hex_side_length, L):
            g0 = R @ np.array([x0, y0, 0.0]) + t
            gL = R @ np.array([xL, yL, L]) + t
            segments.append((g0, gL, cav_idx))
    return segments


# ---------------------------------------------------------------------------


def test_apothem_at_zero_tilt(cfg):
    """With no tilt, every wall plane sits exactly at the regular-hexagon
    apothem hl * sin(60 deg), for every wall angle."""
    hl = cfg.hex_side_length
    for angle in HEX_ANGLES:
        A, B, C, D = hex_plane_coefficients(angle, tilt=0.0, hl=hl)
        # For a unit normal (A,B,C), D is the signed distance from origin.
        norm = np.sqrt(A * A + B * B + C * C)
        assert np.isclose(norm, 1.0)
        assert np.isclose(D, hl * _SIN60), (
            f'wall at angle {angle}: distance {D} != apothem {hl * _SIN60}')


def test_invalid_angle_raises(cfg):
    with pytest.raises(ValueError):
        hex_plane_coefficients(0.1234, tilt=cfg.tilt, hl=cfg.hex_side_length)


def test_inner_ridges_coincide_exactly(cfg):
    """The six inner vertices are each shared by THREE cavities; the three
    independently computed ridge lines must be identical to machine
    precision -- both endpoints."""
    segments = _global_ridge_segments(cfg)

    # Group by rounded start point
    groups = {}
    for g0, gL, ci in segments:
        key = tuple(np.round(g0, 9))
        groups.setdefault(key, []).append((g0, gL, ci))

    triple_groups = [g for g in groups.values() if len(g) == 3]
    assert len(triple_groups) == 6, (
        f'expected 6 triple-shared inner vertices, found {len(triple_groups)}')

    for members in triple_groups:
        g0s = np.array([m[0] for m in members])
        gLs = np.array([m[1] for m in members])
        assert np.allclose(g0s, g0s[0], atol=1e-9)
        assert np.allclose(gLs, gLs[0], atol=1e-9)


def test_outer_vertices_mismatch_is_small_but_nonzero(cfg):
    """Documented geometric frustration: adjacent PERIPHERAL hexagons cannot
    mate exactly.  Their shared vertices differ by ~3 mm (order tilt^2*hl):
    strictly nonzero, but safely inside the tie-rod merge tolerance."""
    segments = _global_ridge_segments(cfg)

    near_pairs = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments[i][2] == segments[j][2]:
                continue
            d0 = np.linalg.norm(segments[i][0] - segments[j][0])
            if 1e-6 < d0 < cfg.tierod_merge_distance:
                near_pairs.append(d0)

    assert len(near_pairs) == 6, (
        f'expected 6 near-coincident outer pairs, found {len(near_pairs)}')
    for d0 in near_pairs:
        # Order-of-magnitude check against the tilt^2 * hl estimate
        scale = cfg.tilt ** 2 * cfg.hex_side_length
        assert scale < d0 < 5 * scale, (
            f'outer mismatch {d0:.4f} cm outside the expected ~tilt^2*hl band')


def test_line_merge_yields_24_unique_ridges(cfg):
    """42 segments -> 24 unique ridge lines (6 triple + 6 double + 12 single).

    The merge criterion mirrors gcr.geometry.tie_rods but stays pure numpy
    so this test needs no OpenMC.
    """
    tol_dist = cfg.tierod_merge_distance
    tol_ang = np.deg2rad(cfg.tierod_merge_angle_deg)

    lines = []   # (g0, direction, member_count)
    for g0, gL, ci in _global_ridge_segments(cfg):
        d = gL - g0
        d = d / np.linalg.norm(d)
        for line in lines:
            lg0, ld, _ = line
            if np.arccos(np.clip(abs(np.dot(d, ld)), -1, 1)) > tol_ang:
                continue
            v = g0 - lg0
            perp = v - np.dot(v, ld) * ld
            if np.linalg.norm(perp) < tol_dist:
                line[2] += 1
                break
        else:
            lines.append([g0, d, 1])

    counts = sorted(line[2] for line in lines)
    assert len(lines) == 24
    assert counts == [1] * 12 + [2] * 6 + [3] * 6


def test_config_json_round_trip(cfg, tmp_path):
    """A config must survive JSON serialisation exactly, with derived
    fields recomputed rather than trusted from the file."""
    path = tmp_path / 'config.json'
    cfg.to_json(str(path))

    # The file must not smuggle stale derived values past __post_init__:
    data = json.loads(path.read_text())
    assert 'tilt' in data          # present in the snapshot...
    loaded = GCRConfig.from_json(str(path))
    assert loaded == cfg           # ...but recomputed on load

    # A physics-relevant field changed on disk must survive the trip
    data['th_atom_fraction'] = 0.20
    path.write_text(json.dumps(data))
    loaded = GCRConfig.from_json(str(path))
    assert loaded.th_atom_fraction == 0.20
    assert np.isclose(loaded.tilt, cfg.tilt)

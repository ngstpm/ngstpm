"""I.points(side, intorder) and I.segments(): the numpy diagnostics of an
Interface (API-PROPOSAL D19), on TransferPaths, h = 0.3, k = 2."""

from pathlib import Path
import sys

import numpy as np
import pytest
from ngsolve import VOL, ElementId

from ngstpm import ParametricCurve, pair, transfer_paths
from ngstpm.interface import InterfacePoints, InterfaceSegments

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson_interface import domains  # noqa: E402
from test_pairing import disk, facet_data, flower_xy  # noqa: E402


def flower_interface(family="unfitted", maxh=0.3):
    geo = ParametricCurve(flower_xy)
    mesh, D1, D2 = domains(family, maxh, geo)
    p1, p2 = transfer_paths(mesh, geo, D1), transfer_paths(mesh, geo, D2)
    return pair(p1, p2), p1, p2


def kinks(P):
    """|sin| of the angle between xbar - x and xbar - x' at each point."""
    a, b = P.xbar - P.x, P.xbar - P.xp
    return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)


def test_shapes_and_dtypes():
    I, p1, p2 = flower_interface()
    S = I.segments()
    n = I.nsegments
    assert isinstance(S, InterfaceSegments) and n > 0
    for name in ("facet", "el", "loc"):
        a = getattr(S, name)
        assert a.shape == (n, 2) and np.issubdtype(a.dtype, np.integer)
    assert S.t.shape == (n, 2, 2) and S.t.dtype == np.float64
    assert S.x.shape == S.xbar.shape == (n, 2, 2, 2) and S.x.dtype == np.float64
    for side in (1, 2):
        P = I.points(side, 4)
        assert isinstance(P, InterfacePoints)
        m = len(P.weight)
        assert m >= n
        for name in ("x", "xbar", "xp", "sigmap"):
            a = getattr(P, name)
            assert a.shape == (m, 2) and a.dtype == np.float64
        for name in ("el", "loc", "facet", "pel", "ploc", "pfacet", "seg"):
            a = getattr(P, name)
            assert a.shape == (m,) and np.issubdtype(a.dtype, np.integer)
        assert P.weight.shape == (m,) and P.weight.dtype == np.float64
        assert np.allclose(np.linalg.norm(P.sigmap, axis=1), 1, atol=1e-12)


@pytest.mark.parametrize("family", ["fitted", "snapped", "unfitted"])
def test_weights_sum_to_length(family):
    """sum of weights over side i = length of Gamma_h^i (to 1e-12)."""
    geo = ParametricCurve(flower_xy) if family != "fitted" else disk
    mesh, D1, D2 = domains(family, 0.3, geo)
    p1, p2 = transfer_paths(mesh, geo, D1), transfer_paths(mesh, geo, D2)
    I = pair(p1, p2)
    for side, p in ((1, p1), (2, p2)):
        length = sum(np.linalg.norm(b - a) for a, b in facet_data(mesh, p.facets).values())
        assert abs(I.points(side, 4).weight.sum() - length) < 1e-12 * length


def test_points_agree_with_segments():
    """Each point lies in the piece of its segment on its facet; its partner x'
    lies on the segment's facet of the other side, owned by the segment's element."""
    I, p1, p2 = flower_interface()
    S = I.segments()
    mesh = p1.mesh
    worst_in = worst_on = 0.0
    for side in (1, 2):
        i, j = side - 1, 2 - side
        P = I.points(side, 6)
        fd = [facet_data(mesh, p.facets) for p in (p1, p2)]
        assert np.all(P.facet == S.facet[P.seg, i]) and np.all(P.el == S.el[P.seg, i])
        assert np.all(P.loc == S.loc[P.seg, i])
        assert np.all(P.pfacet == S.facet[P.seg, j]) and np.all(P.pel == S.el[P.seg, j])
        assert np.all(P.ploc == S.loc[P.seg, j])
        for k in range(len(P.seg)):
            s = P.seg[k]
            a, b = fd[i][P.facet[k]]
            t = np.dot(P.x[k] - a, b - a) / np.dot(b - a, b - a)
            lo, hi = sorted(S.t[s, i])
            worst_in = max(worst_in, lo - t, t - hi, np.linalg.norm(a + t * (b - a) - P.x[k]))
            a, b = fd[j][P.pfacet[k]]
            tp = np.dot(P.xp[k] - a, b - a) / np.dot(b - a, b - a)
            worst_on = max(worst_on, np.linalg.norm(a + tp * (b - a) - P.xp[k]),
                           -tp - 0.05, tp - 1 - 0.05)                 # GAP_TOL: merged breakpoints
        for el, loc, f in zip(P.pel, P.ploc, P.pfacet):
            assert mesh[ElementId(VOL, int(el))].facets[loc].nr == f
    assert worst_in < 1e-12 and worst_on < 1e-12


def test_flower_closest_kink_small():
    """Closest-point paths are normal to Gamma at xbar, so x -> xbar <- x' is
    straight up to the fit of sigma: max |sin| = 1.26e-2 at h = 0.3 (measured
    2026-11-20; 8.7e-4 at h = 0.2, 9.0e-5 at h = 0.15). Before the bracketed
    partner search it was 0.37 (a spurious partner)."""
    I, p1, p2 = flower_interface()
    assert max(kinks(I.points(side, 6)).max() for side in (1, 2)) < 2e-2


def test_no_spurious_far_partner():
    """Regression (Task 1, D20): at h = 0.3 the previous pairing picked a partner 1.9
    away, across Omega_1, whose sigma line passed through xbar by accident.
    Now |x' - xbar| <= 0.28 (O(h); the reach of the flower is 0.30)."""
    I, p1, p2 = flower_interface()
    for side in (1, 2):
        P = I.points(side, 6)
        assert np.linalg.norm(P.xbar - P.xp, axis=1).max() < 0.5

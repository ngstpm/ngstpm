"""pair(p1, p2): the common subdivision of Gamma by two sides' transfer paths
(API-PROPOSAL D6, D10, D14, Q10, Q12). Successor of test_gap_pairing and
test_gap_pairing_with_kink, on TransferPaths.

Tolerances (ngstpm.interface, relative to the mean facet length h of
Gamma_h^1), measured on fitted/snapped/unfitted x disk/flower, quasi-normal and
closest paths, h = 0.3 and 0.15, k = 2:
- vertex images of one side (one per facet at the vertex, Q10) differ by at
  most 7.7e-3 h (quasi-normal, snapped flower, h = 0.3) and 1.7e-2 h (closest,
  unfitted flower, h = 0.3); 2.7e-4 h at h = 0.15. These are merged by
  topology; GAP_TOL = 5e-2 h only accepts a side-2 vertex image falling into
  such a gap of side 1 (margin 3 over the worst);
- on one mesh, a vertex on both Gamma_h^1 and Gamma_h^2 (fitted, snapped) is
  merged by its number, not by a tolerance; on two meshes, coinciding vertices
  (netgen's spline points) are merged by MERGE_TOL = 1e-10 h, which works from
  1e-14 h to 1e-3 h. The shortest genuine segment is 2.4e-3 h.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from netgen.geom2d import SplineGeometry
from ngsolve import BitArray, Mesh, VOL, BND, atan2, cos, sqrt

from ngstpm import ParametricCurve, pair, transfer_paths
from ngstpm.interface import GAP_TOL, MERGE_TOL

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson_interface import domains  # noqa: E402


def disk(X, Y):
    return sqrt(X * X + Y * Y) - 1


def flower(X, Y):
    return sqrt(X * X + Y * Y) - 1 - 0.12 * cos(5 * atan2(Y, X))


def disk_np(P):
    return np.hypot(P[:, 0], P[:, 1]) - 1


def flower_np(P):
    return np.hypot(P[:, 0], P[:, 1]) - 1 - 0.12 * np.cos(5 * np.arctan2(P[:, 1], P[:, 0]))


def circle_xy(t):
    return np.cos(t), np.sin(t)


def flower_xy(t):
    r = 1 + 0.12 * np.cos(5 * t)
    return r * np.cos(t), r * np.sin(t)


def facet_data(mesh, F):
    """{facet nr: (p0, p1)} with the vertex order of the mesh (t = 0 at p0)."""
    return {f.nr: tuple(np.array(mesh[v].point) for v in f.vertices) for f in mesh.facets if F[f.nr]}


def check(I, p1, p2, phi_np):
    """The checks of the pairing; returns the numbers (see the asserts)."""
    paths = (p1, p2)
    fd = [facet_data(p.mesh, p.facets) for p in paths]
    h = np.mean([np.linalg.norm(b - a) for a, b in fd[0].values()])
    S = I._segments()
    n = I.nsegments
    tiling = 0.0
    for i in (0, 1):
        length = {f: 0.0 for f in fd[i]}
        for s in range(n):
            a, b = fd[i][S["facet"][s, i]]
            length[S["facet"][s, i]] += abs(S["t"][s, i, 1] - S["t"][s, i, 0]) * np.linalg.norm(b - a)
            assert paths[i].elements[int(S["el"][s, i])]            # K in D_i, K' in D_j
        tiling = max(tiling, max(abs(L - np.linalg.norm(b - a)) for f, (a, b) in fd[i].items()
                                 for L in [length[f]]))
    ends = np.abs(S["xbar"][:, 0] - S["xbar"][:, 1]).max() / h        # the same arc of Gamma
    par = on = over = phi = 0.0
    for side in (1, 2):                                               # rules exist whenever pair did
        for intorder in (4, 12):
            assert len(I._rule(side, intorder)["seg"]) >= n
    for side in (1, 2):
        R = I._rule(side, 6)
        j = 2 - side
        assert np.all(np.abs(R["weight"]) > 0)
        for s in range(n):                                            # one (K, K') per segment
            m = R["seg"] == s
            assert set(R["el"][m]) == {S["el"][s, side - 1]} and set(R["pel"][m]) == {S["el"][s, j]}
        assert all(paths[j].elements[int(e)] for e in R["pel"])
        for k in range(len(R["seg"])):
            a, b = fd[j][S["facet"][R["seg"][k], j]]
            xp = R["xp"][k]
            t = np.dot(xp - a, b - a) / np.dot(b - a, b - a)
            on = max(on, np.linalg.norm(a + t * (b - a) - xp))
            over = max(over, max(-t, t - 1, 0) * np.linalg.norm(b - a) / h)
            w, sg = R["xbar"][k] - xp, R["sigmap"][k]
            par = max(par, abs(w[0] * sg[1] - w[1] * sg[0]))
        phi = max(phi, np.abs(phi_np(R["xbar"])).max())
    return dict(nseg=n, tiling=tiling, ends=ends, on=on, over=over, par=par, phi=phi)


def assert_pairing(r, phi_tol):
    assert r["tiling"] < 1e-12            # the pieces of side i tile Gamma_h^i
    assert r["ends"] < GAP_TOL            # both pieces of a segment reach the same arc of Gamma
    assert r["on"] < 1e-12                # x' on the line of F'
    assert r["over"] < GAP_TOL            # ... and in F', up to merged breakpoints
    assert r["par"] < 1e-10               # xbar - x' parallel to sigma(x')
    assert r["phi"] < phi_tol             # xbar on Gamma


CASES = [(fam, name) for name in ("disk", "flower") for fam in ("fitted", "snapped", "unfitted")]


@pytest.mark.parametrize("family,name", CASES)
def test_pairing_quasi_normal(family, name):
    phi, phi_np = (disk, disk_np) if name == "disk" else (flower, flower_np)
    mesh, D1, D2 = domains(family, 0.3, phi)
    p1, p2 = transfer_paths(mesh, phi, D1), transfer_paths(mesh, phi, D2)
    r = check(pair(p1, p2), p1, p2, phi_np)
    assert r["nseg"] >= max(sum(p1.facets), sum(p2.facets))
    if family == "fitted":                                     # shared vertices: one breakpoint each
        assert r["nseg"] == sum(p1.facets) == sum(p2.facets)
    assert_pairing(r, 1e-5 if name == "disk" else 5e-3)     # flower: 2.8e-3 (snapped)


@pytest.mark.parametrize("family,name", CASES)
def test_pairing_closest(family, name):
    geo = ParametricCurve(circle_xy if name == "disk" else flower_xy)
    mesh, D1, D2 = domains(family, 0.3, geo)
    p1, p2 = transfer_paths(mesh, geo, D1), transfer_paths(mesh, geo, D2)
    r = check(pair(p1, p2), p1, p2, lambda P: geo.signed_distance(P))
    assert r["nseg"] >= max(sum(p1.facets), sum(p2.facets))
    if family == "fitted":                                     # shared vertices: one breakpoint each
        assert r["nseg"] == sum(p1.facets) == sum(p2.facets)
    assert_pairing(r, 1e-5 if name == "disk" else 5e-3)     # flower: 3.7e-3 (unfitted)


def interface_facets(mesh):
    F = BitArray(mesh.nfacet)
    F.Clear()
    for el in mesh.Elements(BND):
        if el.mat == "interface":
            F[el.edges[0].nr] = True
    return F


def two_meshes():
    g1 = SplineGeometry()
    g1.AddCircle((0, 0), 1, bc="interface")
    g2 = SplineGeometry()
    g2.AddRectangle((-1.5, -1.5), (1.5, 1.5), bc="outer")
    g2.AddCircle((0, 0), 1, bc="interface", leftdomain=0, rightdomain=1)
    return Mesh(g1.GenerateMesh(maxh=0.25)), Mesh(g2.GenerateMesh(maxh=0.3))


def test_pairing_two_meshes():
    """Two independent straight meshes: Gamma_h^1, Gamma_h^2 are non-matching
    polygons on the bc "interface"; D_h^i is the whole mesh i."""
    paths = []
    for mesh in two_meshes():
        paths.append(transfer_paths(mesh, disk, facets=interface_facets(mesh)))      # D: the whole mesh
    p1, p2 = paths
    r = check(pair(p1, p2), p1, p2, disk_np)
    verts = [{tuple(np.round(q, 9)) for ab in facet_data(p.mesh, p.facets).values() for q in ab} for p in paths]
    assert r["nseg"] == len(verts[0] | verts[1])            # breakpoints: the vertices of both polygons
    assert_pairing(r, 1e-6)


def test_unpaired_piece_raises():
    mesh, D1, D2 = domains("unfitted", 0.3, disk)
    p1, p2 = transfer_paths(mesh, disk, D1), transfer_paths(mesh, disk, D2)
    F2 = BitArray(p2.facets)
    gone = [f.nr for f in mesh.facets if F2[f.nr] and np.mean([mesh[v].point[0] for v in f.vertices]) > 1.0]
    assert 2 <= len(gone) <= 8
    for f in gone:
        F2[f] = False
    p2.facets = F2                                             # side 2 misses the stretch x > 1
    with pytest.raises(Exception, match=r"unpaired piece of Gamma_h\^1 \(side 1\).*length"):
        pair(p1, p2)
    with pytest.raises(Exception, match=r"unpaired piece of Gamma_h\^2 \(side 2\)"):
        pair(p2, p1)                                           # and the other way round


def test_shared_elements_raise():
    mesh, D1, D2 = domains("unfitted", 0.3, disk)
    p1, p2 = transfer_paths(mesh, disk, D1), transfer_paths(mesh, disk, D2)
    p2.elements = D2 | D1
    with pytest.raises(Exception, match="share element"):
        pair(p1, p2)


def test_point_cloud_rules():
    """Closest-point paths to a coarse cloud (400 points, unfitted flower,
    h = 0.15) are not continuous; pair succeeds, and so do the rules at every
    order. Where no point of the partner piece has its path line through xbar
    (2 of 679 points at intorder 12, side 1), x' is the point of the piece whose
    line passes closest: |(xbar - x') x sigma(x')| <= 1.4e-3; elsewhere 3e-16."""
    from test_geometries import cloud
    geo = cloud(n=400)
    mesh, D1, D2 = domains("unfitted", 0.15, geo)
    I = pair(transfer_paths(mesh, geo, D1), transfer_paths(mesh, geo, D2))
    for side in (1, 2):
        for intorder in (4, 12):
            R = I._rule(side, intorder)
            w, sg = R["xbar"] - R["xp"], R["sigmap"]
            par = np.abs(w[:, 0] * sg[:, 1] - w[:, 1] * sg[:, 0])
            assert np.sum(par > 1e-10) <= 0.01 * len(par) and par.max() < 5e-3


def test_absorption_beyond_gap_tol_raises():
    """The D14 absorption of a rule point without an exact partner is bounded by
    gap_tol: the coarse cloud at h = 0.15 needs 9.7e-3 h (the largest of all test
    geometries, h = 0.3 and 0.15); with gap_tol = 5e-3 h pair raises."""
    from ngstpm import Interface
    from test_geometries import cloud
    geo = cloud(n=400)
    mesh, D1, D2 = domains("unfitted", 0.15, geo)
    p1, p2 = transfer_paths(mesh, geo, D1), transfer_paths(mesh, geo, D2)
    args = [a for p in (p1, p2) for a in (p.mesh, p.elements, p.facets, p.target, p.sigma)]
    assert Interface(*args, merge_tol=MERGE_TOL, gap_tol=GAP_TOL).nsegments > 0
    with pytest.raises(Exception, match=r"unpaired piece of Gamma_h\^1 \(side 1\) at .* miss by"):
        Interface(*args, merge_tol=MERGE_TOL, gap_tol=5e-3)

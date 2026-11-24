"""Gamma as a parametrized curve, a B-spline or a point cloud (closest-point
paths), one- and two-sided, against the level-set description."""

import sys
from math import log2
from pathlib import Path

import numpy as np
import pytest
from netgen.csg import CSGeometry, OrthoBrick, Pnt
from ngsolve import BND, CF, ElementId, Integrate, Mesh, Norm, VOL, cos, dx, pi, sin, sqrt, x, y

from ngstpm import (BSplineCurve, ParametricCurve, PointCloud, boundary_ds, pair, snapped_domains,
                    transfer_paths, unfitted_domain)

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import poisson_interface as itf  # noqa: E402
import poisson_tpm as dirichlet  # noqa: E402


def disk(X, Y):
    return sqrt(X * X + Y * Y) - 1


def circle_xy(t):
    return np.cos(t), np.sin(t)


def flower_xy(t):
    r = 1 + 0.12 * np.cos(5 * t)
    return r * np.cos(t), r * np.sin(t)


def spline():
    t = 2 * np.pi * np.arange(12) / 12
    r = 1 + 0.15 * np.cos(3 * t)
    return BSplineCurve(np.column_stack((r * np.cos(t), r * np.sin(t))))


def cloud(xy=flower_xy, n=20000):
    return PointCloud(np.column_stack(xy(2 * np.pi * np.arange(n) / n)))


GEOMETRIES = {
    "parametric": lambda: ParametricCurve(flower_xy),
    "clockwise": lambda: ParametricCurve(lambda t: flower_xy(-t)),
    "bspline": spline,
    "cloud": cloud,
}


@pytest.mark.parametrize("curve", [ParametricCurve(flower_xy), spline()])
def test_closest_point_is_orthogonal_projection(curve):
    rng = np.random.default_rng(0)
    t = rng.uniform(0, curve.T, 200)
    d = rng.uniform(-0.1, 0.1, 200)
    P, N = curve.point(t), curve.normal(t)
    X = P + d[:, None] * N
    Pc, Nc = curve.closest(X)
    assert np.abs(Pc - P).max() < 1e-9 and np.abs(Nc - N).max() < 1e-7
    assert np.abs(curve.signed_distance(X) - d).max() < 1e-9          # N points outwards
    assert np.abs((-curve).signed_distance(X) + d).max() < 1e-9


def test_bspline_is_closed_and_smooth():
    c = spline()
    for f in (c._gamma, c._dgamma, c._d2gamma):
        a, b = np.array(f(np.array([1e-12]))), np.array(f(np.array([c.T - 1e-12])))
        assert np.abs(a - b).max() < 1e-9
    t = np.linspace(0, c.T, 50)
    e = 1e-5
    fd = (np.array(c._gamma(t + e)) - np.array(c._gamma(t - e))) / (2 * e)
    assert np.abs(fd - np.array(c._dgamma(t))).max() < 1e-8
    # degree 1: the polygon of the control points
    poly = BSplineCurve([[0, 0], [1, 0], [1, 1], [0, 1]], degree=1)
    assert np.allclose(poly.point(np.array([0.5, 1.25])), [[0.5, 0], [1, 0.25]]) \
        or np.allclose(poly.point(np.array([0.5 + 1, 1.25 + 1])), [[0.5, 0], [1, 0.25]])


def test_point_cloud_normals_from_ordering():
    t = 2 * np.pi * np.arange(500) / 500
    for s in (1, -1):
        pc = PointCloud(np.column_stack(circle_xy(s * t)))
        assert np.abs(pc.normals - pc.points).max() < 1e-12
    with pytest.raises(ValueError):
        PointCloud(np.zeros((4, 3)))


@pytest.mark.parametrize("make", [lambda: ParametricCurve(circle_xy), lambda: cloud(circle_xy)])
def test_same_subdomains_as_level_set(make):
    mesh = dirichlet.background_mesh(0.2)
    geo = make()
    assert list(unfitted_domain(mesh, geo)[0]) == list(unfitted_domain(mesh, disk)[0])
    assert list(unfitted_domain(mesh, -geo)[0]) == list(unfitted_domain(mesh, lambda *X: -disk(*X))[0])
    outside = list(unfitted_domain(mesh, -geo)[0])            # side=2: {phi > 0}, any geometry
    assert sum(outside) > 100
    assert list(unfitted_domain(mesh, geo, side=2)[0]) == outside == list(unfitted_domain(mesh, disk, side=2)[0])


@pytest.mark.parametrize("side", [1, 2])
def test_prune_infeasible_elements(side):
    """unfitted_domain(..., max_normal_deviation=45) on the disk: a subset of the carved D_h,
    the same for the level set and the parametrized circle; every facet of the pruned
    Gamma_h within 2 diameters of Gamma has n_F . nu >= cos 45 (nu = +-x/|x|, checked here
    independently), and some elements are dropped."""
    mesh = dirichlet.background_mesh(0.2)
    D0 = unfitted_domain(mesh, disk, side)[0]
    D = unfitted_domain(mesh, disk, side, max_normal_deviation=45)[0]
    assert list(D) == list(unfitted_domain(mesh, ParametricCurve(circle_xy), side, max_normal_deviation=45)[0])
    assert all(D0[e] for e in range(mesh.ne) if D[e]) and sum(D) < sum(D0)
    assert list(unfitted_domain(mesh, disk, side, max_normal_deviation=None)[0]) == list(D0)
    with pytest.raises(ValueError, match=r"\[0, 90\]"):
        unfitted_domain(mesh, disk, side, max_normal_deviation=95)
    s = 1 if side == 1 else -1
    worst = 1.0
    for f in mesh.facets:
        els = [e.nr for e in f.elements]
        if len(els) != 2 or D[els[0]] == D[els[1]]:
            continue
        P = np.array([mesh[v].point for v in f.vertices])
        c = P.mean(0)
        if abs(np.linalg.norm(c) - 1) > 2 * np.linalg.norm(P[1] - P[0]):
            continue
        e = els[0] if D[els[0]] else els[1]
        ce = np.mean([mesh[v].point for v in mesh[ElementId(VOL, e)].vertices], axis=0)
        n = np.array([P[1, 1] - P[0, 1], P[0, 0] - P[1, 0]])
        n = n / np.linalg.norm(n) * np.sign(n @ (c - ce))
        worst = min(worst, s * n @ c / np.linalg.norm(c))
    assert worst >= np.cos(np.pi / 4) - 1e-9


# -- Dirichlet (one-sided paths) --------------------------------------------------

@pytest.mark.parametrize("name", GEOMETRIES)
def test_dirichlet_quadratic_reproduced(name):
    g = lambda X, Y: X * X - 2 * X * Y + Y
    mesh = dirichlet.background_mesh(0.3)
    (qh, uh, _), inner = dirichlet.solve(mesh, 2, CF(-2.0), g, GEOMETRIES[name]())
    dV = dx(definedonelements=inner)
    assert sqrt(Integrate((uh - g(x, y)) ** 2 * dV, mesh)) < 1e-11
    assert sqrt(Integrate((qh + CF((2 * x - 2 * y, -2 * x + 1))) ** 2 * dV, mesh)) < 1e-10


def dirichlet_errors(phi, k, maxh):
    g = lambda X, Y: sin(pi * X) * cos(pi * Y)
    qex = CF((-pi * cos(pi * x) * cos(pi * y), pi * sin(pi * x) * sin(pi * y)))
    mesh = dirichlet.background_mesh(maxh)
    (qh, uh, _), inner = dirichlet.solve(mesh, k, 2 * pi**2 * g(x, y), g, phi)
    dV = dx(definedonelements=inner)
    return (sqrt(Integrate((uh - g(x, y)) ** 2 * dV, mesh)),
            sqrt(Integrate((qh - qex) ** 2 * dV, mesh)))


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("make, rel", [(lambda: ParametricCurve(circle_xy), 1e-6),
                                       (lambda: cloud(circle_xy), 1e-3)],
                         ids=["parametric", "cloud"])
def test_dirichlet_disk_matches_level_set(make, rel, k):
    """On the disk grad phi is radial: same paths as the closest-point ones."""
    ref = dirichlet_errors(disk, k, 0.2)
    assert dirichlet_errors(make(), k, 0.2) == pytest.approx(ref, rel=rel)


@pytest.mark.parametrize("name", ["bspline", "cloud"])
def test_dirichlet_convergence(name):
    e = [dirichlet_errors(GEOMETRIES[name](), 2, maxh) for maxh in (0.2, 0.1)]
    assert log2(e[0][0] / e[1][0]) > 2.6 and log2(e[0][1] / e[1][1]) > 2.6


def test_sphere_point_cloud_3d():
    """3D cloud with normals: same D_h as the level set, paths end on the sphere."""
    geo = CSGeometry()
    geo.Add(OrthoBrick(Pnt(-1.3, -1.3, -1.3), Pnt(1.3, 1.3, 1.3)))
    mesh = Mesh(geo.GenerateMesh(maxh=0.4))
    i = np.arange(200000) + 0.5                                       # Fibonacci sphere
    th, ph = np.arccos(1 - 2 * i / len(i)), np.pi * (1 + 5**0.5) * i
    S = np.column_stack((np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th)))
    pc = PointCloud(S, normals=S)
    inner = unfitted_domain(mesh, pc)[0]
    assert list(inner) == list(unfitted_domain(mesh, lambda X, Y, Z: sqrt(X * X + Y * Y + Z * Z) - 1)[0])
    p = transfer_paths(mesh, pc, inner)
    r = Norm(p.target)
    dB = boundary_ds(mesh, inner, bonus_intorder=4)
    area = Integrate(1 * dB, mesh)
    assert sqrt(Integrate((r - 1) ** 2 * dB, mesh) / area) < 1e-4
    assert 0 < Integrate(p.ell * dB, mesh) / area < 0.4


# -- interface (two-sided paths) ----------------------------------------------------

def test_closest_paths_kink_only_with_cloud():
    """x on Gamma_h^i and its partner x' on Gamma_h^j reach the same xbar on
    Gamma from opposite sides. Closest-point paths are normal to Gamma at
    xbar, so on a curve the two segments are collinear; with the nearest
    point of a coarse cloud (400 points) they meet with a kink."""
    for geo, kinked in ((ParametricCurve(flower_xy), False), (cloud(n=400), True)):
        mesh, D1, D2 = itf.domains("unfitted", 0.15, geo)
        I = pair(transfer_paths(mesh, geo, D1), transfer_paths(mesh, geo, D2))
        exact = ParametricCurve(flower_xy)
        kink = worst = 0.0
        for side in (1, 2):
            P = I.points(side, 6)
            assert len(P.el) > 50
            worst = max(worst, np.abs(exact.signed_distance(P.xbar)).max())
            s = exact.signed_distance(P.x)
            assert np.all(s < 0.01) if side == 1 else np.all(s > -0.01)     # P1 classification
            sp = exact.signed_distance(P.xp)
            assert np.all(sp > -0.01) if side == 1 else np.all(sp < 0.01)
            a, b = P.xbar - P.x, P.xbar - P.xp
            kink = max(kink, np.max(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
                                    / np.maximum(np.linalg.norm(a, axis=1)
                                                 * np.linalg.norm(b, axis=1), 1e-300)))
        assert worst < (1e-3 if kinked else 1e-4)                     # xbar on Gamma
        assert (kink > 1e-2) == kinked


KAPPA = (1.0, 10.0)
KAPPA_CF = tuple((lambda X, Y, c=c: CF(c)) for c in KAPPA)


def polynomial_solution(k):
    k1, k2 = KAPPA
    u1 = lambda X, Y: X**k - 2 * X * Y**(k - 1) + Y
    u2 = lambda X, Y: 0.5 * Y**k + 3 * X**(k - 1) * Y - X + 2
    q1 = lambda X, Y: -k1 * CF((k * X**(k - 1) - 2 * Y**(k - 1), -2 * (k - 1) * X * Y**(k - 2) + 1))
    q2 = lambda X, Y: -k2 * CF((3 * (k - 1) * X**(k - 2) * Y - 1, 0.5 * k * Y**(k - 1) + 3 * X**(k - 1)))
    return u1, u2, q1, q2


@pytest.mark.parametrize("family", ["fitted", "snapped", "unfitted"])
@pytest.mark.parametrize("name", ["parametric", "bspline", "cloud"])
@pytest.mark.parametrize("k", [2, 3])
def test_interface_piecewise_polynomials_reproduced(family, name, k):
    """Patch test with two-sided closest-point paths, whatever the gap."""
    if family == "fitted":
        if name != "parametric":
            pytest.skip("the fitted mesh is that of the disk")
        geo = ParametricCurve(circle_xy)
    else:
        geo = GEOMETRIES[name]()
    u1, u2, q1, q2 = polynomial_solution(k)
    f1 = q1(x, y)[0].Diff(x) + q1(x, y)[1].Diff(y)
    f2 = q2(x, y)[0].Diff(x) + q2(x, y)[1].Diff(y)
    mesh, D1, D2 = itf.domains(family, 0.3, geo)
    sol = itf.solve(mesh, k, geo, D1, D2, *KAPPA_CF, f1, f2,
                    lambda X, Y: u1(X, Y) - u2(X, Y), lambda X, Y: q1(X, Y) - q2(X, Y), u2(x, y))
    eu, eq = itf.errors(mesh, sol, D1, D2, u1, u2, q1, q2)
    assert eu < 1e-10 and eq < 1e-9


# ||u - u_h|| of the level-set disk at maxh = 0.1, k = 2 (test_interface.REFERENCE)
# (closest-point paths of the circle; measured 4.1024e-04, 6.3374e-04, 9.6852e-04)
@pytest.mark.parametrize("family, ref", [("fitted", 4.102e-04),
                                         ("snapped", 6.337e-04),     # old 6.374e-04 (one rule per facet)
                                         ("unfitted", 9.685e-04)])   # old 1.140e-03 (one rule per facet)
def test_interface_disk_matches_level_set(family, ref):
    from ngsolve import exp
    k1, k2 = KAPPA
    u1 = lambda X, Y: sin(pi * X) * cos(pi * Y)
    u2 = lambda X, Y: exp(0.5 * X) * cos(0.5 * Y) + X * Y
    q1 = lambda X, Y: -k1 * CF((pi * cos(pi * X) * cos(pi * Y), -pi * sin(pi * X) * sin(pi * Y)))
    q2 = lambda X, Y: -k2 * CF((0.5 * exp(0.5 * X) * cos(0.5 * Y) + Y,
                                -0.5 * exp(0.5 * X) * sin(0.5 * Y) + X))
    geo = ParametricCurve(circle_xy)
    mesh, D1, D2 = itf.domains(family, 0.1, geo)
    sol = itf.solve(mesh, 2, geo, D1, D2, *KAPPA_CF, k1 * 2 * pi**2 * u1(x, y), CF(0.0),
                    lambda X, Y: u1(X, Y) - u2(X, Y), lambda X, Y: q1(X, Y) - q2(X, Y), u2(x, y))
    assert itf.errors(mesh, sol, D1, D2, u1, u2, q1, q2)[0] == pytest.approx(ref, rel=1e-3)


def wave(X, Y):                         # y = 0.5 + 0.05 sin(2 pi x), crosses the sides x = 0, 1 obliquely
    return 0.5 + 0.05 * sin(2 * pi * X) - Y


def unit_square(maxh):
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bc="outer")
    return Mesh(geo.GenerateMesh(maxh=maxh))


@pytest.mark.parametrize("maxh", [0.2, 0.05])
def test_snapped_boundary_vertices_stay_on_their_edge(maxh):
    """A moved vertex on a side of the square stays on it and lands on Gamma (Newton along the side)."""
    mesh = unit_square(maxh)
    snapped, _, _ = snapped_domains(mesh, wave)
    V0 = np.array([v.point for v in mesh.vertices])
    V1 = np.array([v.point for v in snapped.vertices])
    side = (V0[:, 0] == 0) | (V0[:, 0] == 1)
    moved = np.any(V0 != V1, axis=1)
    assert np.sum(side & moved) >= 2
    assert np.array_equal(V0[side, 0], V1[side, 0])
    Y = 0.5 + 0.05 * np.sin(2 * np.pi * V1[side & moved, 0])
    assert np.max(np.abs(V1[side & moved, 1] - Y)) < 1e-14


def test_snapped_interior_unchanged():
    """With no cut element at the mesh boundary every moved vertex is the plain Newton projection
    along grad phi (4 steps), bit for bit, and no boundary vertex moves."""
    mesh = itf.background_mesh(0.3)
    snapped, _, _ = snapped_domains(mesh, disk)
    p = disk(x, y)
    grad = CF((p.Diff(x), p.Diff(y)))
    vals = [p(mesh(*v.point)) for v in mesh.vertices]
    cut = set()
    for el in mesh.Elements():
        s = [vals[v.nr] < 0 for v in el.vertices]
        if any(s) and not all(s):
            cut.update(v.nr for v in el.vertices)
    assert cut
    for v0, v1 in zip(mesh.vertices, snapped.vertices):
        c = np.array(v0.point)
        if v0.nr in cut:
            for _ in range(4):
                mp = mesh(*c)
                g = np.array(grad(mp))
                c = c - p(mp) * g / g.dot(g)
        assert np.array_equal(c, np.array(v1.point))


def test_snapped_gamma_parallel_to_the_boundary():
    """Gamma = {y = 0.05} parallel to the bottom: no tangential gradient there, so the vertices take
    the unconstrained Newton step (as before the boundary treatment) and land on Gamma."""
    mesh = unit_square(0.2)
    snapped, _, _ = snapped_domains(mesh, lambda X, Y: 0.05 - Y)
    V0 = np.array([v.point for v in mesh.vertices])
    V1 = np.array([v.point for v in snapped.vertices])
    moved = np.any(V0 != V1, axis=1)
    assert moved.sum() > 0 and not np.isnan(V1).any()
    assert np.max(np.abs(V1[moved, 1] - 0.05)) < 1e-14 and np.array_equal(V0[moved, 0], V1[moved, 0])


def test_snapped_curved_boundary():
    """On the polygon of the unit disk the boundary vertices of cut elements slide along the mean
    tangent onto Gamma (not frozen as corners); they leave the circle by O(h * step); the elements
    of D1, D2 keep a positive area."""
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddCircle((0, 0), 1, bc="outer")
    mesh = Mesh(geo.GenerateMesh(maxh=0.2))
    line = lambda X, Y: 0.1 + 0.3 * X - Y
    snapped, D1, D2 = snapped_domains(mesh, line)
    V0 = np.array([v.point for v in mesh.vertices])
    V1 = np.array([v.point for v in snapped.vertices])
    bnd = {v.nr for el in mesh.Elements(BND) for v in el.vertices}
    vals = line(V0[:, 0], V0[:, 1])
    cut = set()
    for el in mesh.Elements():
        s = [vals[v.nr] < 0 for v in el.vertices]
        if any(s) and not all(s):
            cut.update(v.nr for v in el.vertices)
    on = sorted(cut & bnd)
    assert len(on) >= 2
    assert np.all(np.any(V0[on] != V1[on], axis=1))
    assert np.max(np.abs(line(V1[on, 0], V1[on, 1]))) < 1e-12
    assert np.max(np.abs(np.hypot(V1[on, 0], V1[on, 1]) - 1)) < 0.05
    for el in snapped.Elements():
        if D1[el.nr] or D2[el.nr]:
            a, b, c = V1[[v.nr for v in el.vertices]]
            assert (b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0] > 0

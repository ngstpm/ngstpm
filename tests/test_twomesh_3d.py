"""Interface problems on two meshes in 3D (ngstpm/twomesh.py, src/ifcmeasure.cpp):
the weak form of examples/poisson_interface_two_meshes.py (its solve, generalised
to the mesh dimension) on the unit sphere, Omega_1 = the ball and Omega_2 = a box
minus the ball, each meshed on its own with straight tetrahedra
(examples/poisson_interface_two_meshes_3d.py: meshes)."""

from pathlib import Path
import sys

import numpy as np
import pytest
from ngsolve import (CF, FESpace, FacetFESpace, GridFunction, Integrate, L2, LinearForm, dx, exp, sqrt, x, y, z)

from ngstpm import pair, transfer_paths
from ngstpm._ngstpm import Interface

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import poisson_interface_two_meshes_3d as ex  # noqa: E402
from poisson_interface_two_meshes import solve  # noqa: E402

H = (0.5, 1.23 * 0.5)                            # the coarsest level of the example


def paths(mesh1, mesh2):
    return [transfer_paths(m, ex.phi, facets=m.Boundaries("interface")) for m in (mesh1, mesh2)]


def affine_solution():
    """u_i affine, q_i = -kappa_i grad u_i constant, jumps in u and in the flux."""
    k1, k2 = 1.0, 10.0
    u1 = lambda X, Y, Z: X - 2 * Y + 3 * Z + 1
    u2 = lambda X, Y, Z: 0.5 * X + Y - Z + 2
    q1 = lambda X, Y, Z: CF((-k1, 2 * k1, -3 * k1))
    q2 = lambda X, Y, Z: CF((-0.5 * k2, -k2, k2))
    return u1, u2, q1, q2


def run(sol_ex, order=1):
    u1, u2, q1, q2 = sol_ex
    X = (x, y, z)
    f1 = sum(q1(*X)[i].Diff(c) for i, c in enumerate(X))
    f2 = sum(q2(*X)[i].Diff(c) for i, c in enumerate(X))
    gD = lambda *P: u1(*P) - u2(*P)
    gN = lambda *P: q1(*P) - q2(*P)
    mesh1, mesh2 = ex.meshes(*H)
    sol = solve(mesh1, mesh2, order, ex.phi, ex.kappa1, ex.kappa2, f1, f2, gD, gN, u2(*X))
    e = lambda a, b, m: Integrate((a - b) ** 2 * dx(bonus_intorder=4), m)
    return sol, (sqrt(e(sol[1], u1(*X), mesh1) + e(sol[4], u2(*X), mesh2)),
                 sqrt(e(sol[0], q1(*X), mesh1) + e(sol[3], q2(*X), mesh2)))


def test_affine_solution_is_reproduced():
    """Consistency of the 3D forms: piecewise affine u_i (constant q_i) solve the
    discrete problem, k = 1. The data are evaluated at the ends xbar of the paths
    (on Gamma) and the transfer along the paths is exact for affine u_i, so neither
    Gamma_h != Gamma nor the non-matching meshes leave an error: measured
    |u - uh| = 8.5e-15, |q - qh| = 6.1e-13."""
    _, (eu, eq) = run(affine_solution())
    print(f"affine: |u-uh|={eu:.1e} |q-qh|={eq:.1e}")
    assert eu < 1e-12 and eq < 1e-11


def test_condensed_equals_uncondensed(monkeypatch):
    """Condensing the interface coupling (D16) gives the uncondensed solution in 3D:
    measured max |difference| = 1.7e-14 relative to the largest coefficient."""
    sol_ex = (ex.u1, ex.u2, ex.q1, ex.q2)
    a = [c.vec.FV().NumPy().copy() for c in run(sol_ex)[0]]
    bilinear = Interface.BilinearForm
    monkeypatch.setattr(Interface, "BilinearForm", lambda I, fes, condense=False: bilinear(I, fes, condense=False))
    b = [c.vec.FV().NumPy().copy() for c in run(sol_ex)[0]]
    rel = max(abs(ca - cb).max() for ca, cb in zip(a, b)) / max(abs(cb).max() for cb in b)
    print(f"condensed - uncondensed: {rel:.1e}")
    assert rel <= 1e-12


@pytest.fixture(scope="module")
def interface():
    mesh1, mesh2 = ex.meshes(*H)
    p1, p2 = paths(mesh1, mesh2)
    return p1, p2, pair(p1, p2)


def g(X, Y, Z):                                  # data, evaluated at the end of the path
    return X * X - 2 * X * Y + 3 * Z + exp(0.3 * Z) + 1


def poly(X, Y, Z):                               # degree 1: in the spaces of order 1
    return 2 * X - Y + 0.5 * Z + 1


@pytest.mark.parametrize("side", [1, 2])
def test_integrate_is_the_piece_rule(interface, side):
    """Integrate(g(xbar) * I.ds(side)) = sum over I.points(side, 5 + bonus) of weight * g(xbar)."""
    p1, p2, I = interface
    p = (p1, p2)[side - 1]
    P = I.points(side, 5 + 2)
    ref = np.sum(P.weight * g(*P.xbar.T))
    assert abs(ref) > 1
    assert abs(Integrate(g(*p.target) * I.ds(side=side, bonus_intorder=2), p.mesh) - ref) < 1e-12 * abs(ref)


@pytest.mark.parametrize("side", [1, 2])
def test_linear_form_is_the_piece_rule(interface, side):
    """lf(u) = sum over I.points(side, 2k + bonus) of weight * g(xbar) * u(x), for u
    in the test space (L2 of order k = 1)."""
    p1, p2, I = interface
    p = (p1, p2)[side - 1]
    k, bonus = 1, 3
    fes = L2(p.mesh, order=k)
    lf = LinearForm(fes)
    lf += g(*p.target) * fes.TestFunction() * I.ds(side=side, bonus_intorder=bonus)
    lf.Assemble()
    u = GridFunction(fes)
    u.Set(poly(x, y, z))
    P = I.points(side, 2 * k + bonus)
    ref = np.sum(P.weight * g(*P.xbar.T) * poly(*P.x.T))
    assert abs(ref) > 1
    assert abs(lf.vec.InnerProduct(u.vec) - ref) < 1e-12 * abs(ref)


@pytest.mark.parametrize("side", [1, 2])
def test_bilinear_form_is_the_piece_rule(interface, side):
    """Two meshes, mu = 1 the facet test function of side i: mu^T A u =
    sum_points w p(x) for the own uhat_i, w p(x') for the partner's
    u_j.Extrapolate(), w p(xbar) for u_j.Extrapolate(xbar) (p of degree 1, in the spaces)."""
    p1, p2, I = interface
    i, j = side - 1, 2 - side
    own, other = (p1, p2)[i], (p1, p2)[j]
    k = 1
    M, W = FESpace([FacetFESpace(own.mesh, order=k)]), FESpace([L2(other.mesh, order=k)])   # product spaces
    fes = I.Space(*((M, W) if side == 1 else (W, M)))
    (a1, a2), (b1, b2) = fes.TnT()
    uhat, w, mu = (a1, a2, b1) if side == 1 else (a2, a1, b2)
    P = I.points(side, 2 * k)
    expect = {"own": np.sum(P.weight * poly(*P.x.T)), "partner": np.sum(P.weight * poly(*P.xp.T)),
              "xbar": np.sum(P.weight * poly(*P.xbar.T))}
    for name, integrand in (("own", uhat * mu), ("partner", w.Extrapolate() * mu),
                            ("xbar", w.Extrapolate(own.target) * mu)):
        a = I.BilinearForm(fes)
        a += integrand * I.ds(side=side)
        a.Assemble()
        u = I.GridFunction(fes)
        c = u.components
        c[i].Set(poly(x, y, z), dual=True) if name == "own" else c[j].Set(poly(x, y, z))
        one = I.GridFunction(fes)
        one.components[i].Set(1.0, dual=True)
        Au = a.mat.CreateColVector()
        Au.data = a.mat * u.vec
        val = one.vec.InnerProduct(Au)
        assert abs(expect[name]) > 1
        assert abs(val - expect[name]) < 1e-12 * abs(expect[name]), name


def test_example_coarsest_level():
    """The example runs (k = 1, its coarsest level): 583 tetrahedra,
    |u - uh| = 6.726e-02 and |q - qh| = 4.283 measured."""
    N, eu, eq = ex.run(1, ex.LEVELS[0])
    assert N == 583 and eu == pytest.approx(6.726e-2, rel=1e-2) and eq == pytest.approx(4.283, rel=1e-2)

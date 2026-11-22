import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, Integrate, dx, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from stokes_darcy import errors, norm, solve, stokes_darcy_data  # noqa: E402
from stokes_darcy_curved import (curved_data, gamma, meshes, phi, prune_teeth, snapped, square,  # noqa: E402
                                 stokes_mean, unfitted, vertical_paths)
from ngstpm import boundary_facets, pair, transfer_paths  # noqa: E402

KAPPA = CF((2, 0.5, 0.5, 1), dims=(2, 2))
NU, OMEGA, TAU = 0.5, 0.7, 3.0


def paths_of(mesh1, mesh2):
    return [transfer_paths(m, phi, facets=m.Boundaries("interface")) for m in (mesh1, mesh2)]


@pytest.mark.parametrize("maxh,nonmatching,tol", [(0.1, False, 1e-9), (0.05, True, 1e-11)])
def test_area(maxh, nonmatching, tol):
    """|mesh i| + the sliver of its paths (with negative parts) = |Omega_i| = 1/2."""
    mesh1, mesh2 = meshes(maxh, nonmatching)
    paths = paths_of(mesh1, mesh2)
    I = pair(*paths)
    for i, (mesh, p) in enumerate(zip((mesh1, mesh2), paths)):
        area = Integrate(CF(1), mesh) + Integrate(CF(1).RegionIntegral(p) * I.ds(side=i + 1), mesh)
        assert abs(area - 0.5) < tol


@pytest.mark.parametrize("darcy_dirichlet", [None, "bottom"])
def test_polynomial_reproduced_exactly(darcy_dirichlet):
    """Quadratic u_s, P_s (zero mean over the curved Omega_s), p_d; non-matching vertices on Gamma.
    With the level, ptilde_s alone is not exact (the polygon does not have the mean of Omega_s);
    the remaining error is that of the paths (about 1e-11 at maxh = 0.1)."""
    P_s = x - 0.5 + y * y - stokes_mean(lambda X, Y: X - 0.5 + Y ** 2)
    data = stokes_darcy_data(CF((x * x + y, -2 * x * y + x)), P_s, x * x - x * y + y, phi,
                             nu=NU, kappa=KAPPA, omega=OMEGA)
    mesh1, mesh2 = meshes(0.1, nonmatching=True)
    sol, _, alpha = solve(mesh1, mesh2, 2, phi, data, nu=NU, kappa=KAPPA, omega=OMEGA, tau=TAU,
                          darcy_dirichlet=darcy_dirichlet)
    assert max(errors(mesh1, mesh2, sol, data)) < 1e-10
    if darcy_dirichlet is None:
        assert Integrate((sol[2] - alpha - P_s) ** 2, mesh1) ** 0.5 > 1e-4


def test_zero_mean_of_the_stokes_pressure():
    """P_s,h over mesh 1 plus its extrapolation over the sliver of the Stokes paths vanishes
    (with the solver's facet quadrature, bonus 6: Integrate's default differs by 1e-10)."""
    mesh1, mesh2 = meshes(0.1)
    sol, _, _ = solve(mesh1, mesh2, 2, phi, curved_data())
    p1, p2 = paths_of(mesh1, mesh2)
    dI = pair(p1, p2).ds(side=1, bonus_intorder=6)
    total = Integrate(sol[2], mesh1) + Integrate(sol[2].RegionIntegral(p1) * dI, mesh1)
    assert abs(total) < 1e-12


@pytest.mark.parametrize("nonmatching", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_convergence_rate(k, nonmatching):
    """maxh 0.1 -> 0.05; observed EOC 1.98-2.33 (k = 1), 2.93-3.35 (k = 2)."""
    data = curved_data()
    errs = []
    for maxh in (0.1, 0.05):
        mesh1, mesh2 = meshes(maxh, nonmatching)
        sol, _, _ = solve(mesh1, mesh2, k, phi, data)
        errs.append(errors(mesh1, mesh2, sol, data))
    for e0, e1 in zip(*errs):
        assert log2(e0 / e1) > k + 1 - 0.15


# Snapped family: one background mesh, D1 (Stokes) and D2 (Darcy) on either side of the snapped slivers.

def snapped_paths(mesh, D1, D2):
    return [transfer_paths(mesh, phi, D) for D in (D1, D2)]


def test_snapped_boundary_vertices_stay_on_the_square():
    """The snapped vertices on the sides x = 0, 1 move along them, onto (x, gamma(x))."""
    background = square(0.05)
    mesh, _, _ = snapped(0.05)
    X = [v.point[0] for v in mesh.vertices]
    assert min(X) == 0 and max(X) == 1
    moved = [v.point for v0, v in zip(background.vertices, mesh.vertices)
             if v0.point != v.point and v0.point[0] in (0.0, 1.0)]
    assert moved
    for X, Y in moved:
        assert X in (0.0, 1.0) and abs(Y - gamma(X)) < 1e-12


@pytest.mark.parametrize("maxh,tol", [(0.1, 1e-8), (0.05, 1e-10)])
def test_snapped_area(maxh, tol):
    """|D_h^i| + the sliver of its paths = |Omega_i| = 1/2 (up to the paths: 5e-10, 2e-9 at maxh 0.1)."""
    mesh, D1, D2 = snapped(maxh)
    paths = snapped_paths(mesh, D1, D2)
    I = pair(*paths)
    for i, (D, p) in enumerate(zip((D1, D2), paths)):
        area = Integrate(CF(1) * dx(definedonelements=D), mesh) + Integrate(CF(1).RegionIntegral(p) * I.ds(side=i + 1), mesh)
        assert abs(area - 0.5) < tol


@pytest.mark.parametrize("darcy_dirichlet", [None, "bottom"])
def test_snapped_polynomial_reproduced_exactly(darcy_dirichlet):
    """As test_polynomial_reproduced_exactly on the snapped mesh (maxh 0.05; with the level the
    error is that of the paths: 1e-10 at maxh 0.1, 5e-13 at 0.05)."""
    P_s = x - 0.5 + y * y - stokes_mean(lambda X, Y: X - 0.5 + Y ** 2)
    data = stokes_darcy_data(CF((x * x + y, -2 * x * y + x)), P_s, x * x - x * y + y, phi,
                             nu=NU, kappa=KAPPA, omega=OMEGA)
    mesh, D1, D2 = snapped(0.05)
    sol, _, alpha = solve(mesh, mesh, 2, phi, data, nu=NU, kappa=KAPPA, omega=OMEGA, tau=TAU,
                          darcy_dirichlet=darcy_dirichlet, D1=D1, D2=D2)
    assert max(errors(mesh, mesh, sol, data, D1, D2)) < 1e-11
    if darcy_dirichlet is None:
        assert norm(mesh, sol[2] - alpha - P_s, D1) > 1e-5


def test_snapped_zero_mean_of_the_stokes_pressure():
    """P_s,h over D_h^1 plus its extrapolation over the sliver of the Stokes paths vanishes."""
    mesh, D1, D2 = snapped(0.1)
    sol, _, _ = solve(mesh, mesh, 2, phi, curved_data(), D1=D1, D2=D2)
    p1, p2 = snapped_paths(mesh, D1, D2)
    dI = pair(p1, p2).ds(side=1, bonus_intorder=6)
    total = Integrate(sol[2] * dx(definedonelements=D1), mesh) + Integrate(sol[2].RegionIntegral(p1) * dI, mesh)
    assert abs(total) < 1e-12


@pytest.mark.parametrize("k", [1, 2])
def test_snapped_convergence_rate(k):
    """maxh 0.1 -> 0.05; observed EOC 2.08-2.44 (k = 1), 3.31-3.64 (k = 2)."""
    data = curved_data()
    errs = []
    for maxh in (0.1, 0.05):
        mesh, D1, D2 = snapped(maxh)
        sol, _, _ = solve(mesh, mesh, k, phi, data, D1=D1, D2=D2)
        errs.append(errors(mesh, mesh, sol, data, D1, D2))
    for e0, e1 in zip(*errs):
        assert log2(e0 / e1) > k + 1 - 0.15


# Unfitted family: D1, D2 the elements entirely on either side, an O(h) staircase gap, vertical paths.
# No rate test: the rates are irregular (stokes_darcy_curved docstring).

def unfitted_paths(mesh, D1, D2):
    return [vertical_paths(mesh, phi, D) for D in (D1, D2)]


@pytest.mark.parametrize("maxh", [0.1, 0.05])
def test_unfitted_area(maxh):
    """|D_h^i| + the region swept by its vertical paths = 1/2 (5e-11 at maxh 0.1)."""
    mesh, D1, D2 = unfitted(maxh)
    paths = unfitted_paths(mesh, D1, D2)
    I = pair(*paths)
    for i, (D, p) in enumerate(zip((D1, D2), paths)):
        area = Integrate(CF(1) * dx(definedonelements=D), mesh) + Integrate(CF(1).RegionIntegral(p) * I.ds(side=i + 1), mesh)
        assert abs(area - 0.5) < 1e-9


def teeth_of(mesh, D):
    gamma_h = boundary_facets(mesh, D)[0]
    return [el.nr for el in mesh.Elements() if D[el.nr] and sum(gamma_h[f.nr] for f in el.facets) >= 2]


def test_unfitted_teeth():
    """prune_teeth leaves no element of D with two or more facets on Gamma_h (maxh 0.05 has a tooth)."""
    mesh, D1, D2 = unfitted(0.05)
    assert teeth_of(mesh, D1) + teeth_of(mesh, D2)
    assert not teeth_of(mesh, prune_teeth(mesh, D1)) and not teeth_of(mesh, prune_teeth(mesh, D2))


@pytest.mark.parametrize("darcy_dirichlet", [None, "bottom"])
@pytest.mark.parametrize("teeth", [True, False])
def test_unfitted_polynomial_reproduced_exactly(darcy_dirichlet, teeth):
    """As test_polynomial_reproduced_exactly on the unfitted mesh of maxh 0.05, with and without its tooth."""
    P_s = x - 0.5 + y * y - stokes_mean(lambda X, Y: X - 0.5 + Y ** 2)
    data = stokes_darcy_data(CF((x * x + y, -2 * x * y + x)), P_s, x * x - x * y + y, phi,
                             nu=NU, kappa=KAPPA, omega=OMEGA)
    mesh, D1, D2 = unfitted(0.05, teeth)
    sol, _, _ = solve(mesh, mesh, 2, phi, data, nu=NU, kappa=KAPPA, omega=OMEGA, tau=TAU,
                      darcy_dirichlet=darcy_dirichlet, D1=D1, D2=D2, paths=vertical_paths)
    assert max(errors(mesh, mesh, sol, data, D1, D2)) < 1e-10


def test_unfitted_zero_mean_of_the_stokes_pressure():
    """P_s,h over D_h^1 plus its extrapolation over the region swept by the Stokes paths vanishes."""
    mesh, D1, D2 = unfitted(0.1)
    sol, _, _ = solve(mesh, mesh, 2, phi, curved_data(), D1=D1, D2=D2, paths=vertical_paths)
    p1, p2 = unfitted_paths(mesh, D1, D2)
    dI = pair(p1, p2).ds(side=1, bonus_intorder=6)
    total = Integrate(sol[2] * dx(definedonelements=D1), mesh) + Integrate(sol[2].RegionIntegral(p1) * dI, mesh)
    assert abs(total) < 1e-12


POLY = {1: (CF((x + 2 * y, x - y)), lambda X, Y: X + Y, x + 2 * y),
        3: (CF((x ** 3 + y ** 2, -3 * x * x * y + x)), lambda X, Y: X * X * Y + Y ** 3, x ** 3 - x * y * y + y)}


@pytest.mark.parametrize("darcy_dirichlet", [None, "bottom"])
@pytest.mark.parametrize("k", [1, 3])
def test_unfitted_degree_k_reproduced_exactly(k, darcy_dirichlet):
    """Polynomials of degree k (u_s divergence free) at k = 1, 3 on the unfitted mesh of maxh 0.141
    (one of the meshes where the errors jump); measured 5e-14 (k = 1), 2e-12 (k = 3)."""
    u_s, P, p_d = POLY[k]
    P_s = P(x, y) - stokes_mean(P)
    data = stokes_darcy_data(u_s, P_s, p_d, phi, nu=NU, kappa=KAPPA, omega=OMEGA)
    mesh, D1, D2 = unfitted(0.141)
    sol, _, _ = solve(mesh, mesh, k, phi, data, nu=NU, kappa=KAPPA, omega=OMEGA, tau=TAU,
                      darcy_dirichlet=darcy_dirichlet, D1=D1, D2=D2, paths=vertical_paths)
    assert max(errors(mesh, mesh, sol, data, D1, D2)) < 1e-10

import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, InnerProduct, Integrate, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from stokes_darcy import DELTA, errors, meshes, paper_data, phi, solve, stokes_darcy_data  # noqa: E402
from ngstpm import pair, transfer_paths  # noqa: E402

KAPPA = CF((2, 0.5, 0.5, 1), dims=(2, 2))
NU, OMEGA, TAU = 0.5, 0.7, 3.0


def polynomial_data():
    """Quadratic div-free u_s, quadratic P_s with zero mean on Omega_s, quadratic p_d: the
    interface data T_mass, T_force are nonzero."""
    return stokes_darcy_data(CF((x * x + y, -2 * x * y + x)), x - 0.5 + y * y - 7 / 12, x * x - x * y + y,
                             phi, nu=NU, kappa=KAPPA, omega=OMEGA)


@pytest.mark.parametrize("darcy_dirichlet", [None, "bottom"])
def test_polynomial_reproduced_exactly(darcy_dirichlet):
    """Hanging nodes on the interface; Darcy all Neumann (rho, alpha) or with a Dirichlet side."""
    data = polynomial_data()
    mesh1, mesh2 = meshes(4, hanging=True)
    sol, _, _ = solve(mesh1, mesh2, 2, phi, data, nu=NU, kappa=KAPPA, omega=OMEGA, tau=TAU,
                      darcy_dirichlet=darcy_dirichlet)
    assert max(errors(mesh1, mesh2, sol, data)) < 1e-10


def test_condensed_equals_uncondensed():
    data = paper_data()
    mesh1, mesh2 = meshes(8, hanging=True)
    a, _, alpha_a = solve(mesh1, mesh2, 2, phi, data)
    b, _, alpha_b = solve(mesh1, mesh2, 2, phi, data, condense=False)
    for i, mesh in ((0, mesh1), (1, mesh1), (2, mesh1), (4, mesh2), (5, mesh2)):
        assert Integrate(InnerProduct(a[i] - b[i], a[i] - b[i]), mesh) ** 0.5 < 1e-10
    assert abs(alpha_a - alpha_b) < 1e-12


def test_zero_mean_of_the_stokes_pressure():
    """The level is active (Darcy all Neumann): P_s,h = ptilde_s + alpha has zero mean on Omega_s."""
    mesh1, mesh2 = meshes(8)
    sol, _, _ = solve(mesh1, mesh2, 1, phi, paper_data())
    assert abs(Integrate(sol[2], mesh1)) < 1e-13


@pytest.mark.parametrize("hanging", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_convergence_rate(k, hanging):
    """BMS Section 6 solution; observed EOC on n = 8 -> 16: 1.95-2.07 (k = 1), 2.96-3.03 (k = 2)."""
    data = paper_data()
    errs = []
    for n in (8, 16):
        mesh1, mesh2 = meshes(n, hanging)
        sol, _, _ = solve(mesh1, mesh2, k, phi, data)
        errs.append(errors(mesh1, mesh2, sol, data))
    for e0, e1 in zip(*errs):
        assert log2(e0 / e1) > k + 1 - 0.15



@pytest.mark.parametrize("delta", [0.05, 1 / 8])
def test_polynomial_reproduced_exactly_with_gap(delta):
    """A gap delta on each side of y = 1/2 (1/8 = h for n = 8): P_s has zero mean on
    Omega_s = (0,1) x (1/2,1), not on mesh 1, so ptilde_s alone is not exact."""
    data = polynomial_data()
    mesh1, mesh2 = meshes(8, hanging=True, delta=delta)
    sol, _, alpha = solve(mesh1, mesh2, 2, phi, data, nu=NU, kappa=KAPPA, omega=OMEGA, tau=TAU)
    assert max(errors(mesh1, mesh2, sol, data)) < 1e-10
    assert Integrate((sol[2] - alpha - data["P_s"]) ** 2, mesh1) ** 0.5 > 1e-2


def test_area_with_gap():
    """|mesh i| + the sliver swept by its paths = 1/2 on each side."""
    mesh1, mesh2 = meshes(8, hanging=True, delta=0.05)
    paths = [transfer_paths(m, phi, facets=m.Boundaries("interface")) for m in (mesh1, mesh2)]
    I = pair(*paths)
    for i, (mesh, p) in enumerate(zip((mesh1, mesh2), paths)):
        area = Integrate(CF(1), mesh) + Integrate(CF(1).RegionIntegral(p) * I.ds(side=i + 1), mesh)
        assert abs(area - 0.5) < 1e-13


@pytest.mark.parametrize("hanging", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_convergence_rate_gap(k, hanging):
    """delta = h^{7/4}; observed EOC on n = 8 -> 16: 1.90-2.80 (k = 1), 3.11-3.80 (k = 2)."""
    data = paper_data()
    errs = []
    for n in (8, 16):
        mesh1, mesh2 = meshes(n, hanging, DELTA["h74"](1 / n))
        sol, _, _ = solve(mesh1, mesh2, k, phi, data)
        errs.append(errors(mesh1, mesh2, sol, data))
    for e0, e1 in zip(*errs):
        assert log2(e0 / e1) > k + 1 - 0.15

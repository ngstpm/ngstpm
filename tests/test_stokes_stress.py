import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, InnerProduct, Integrate, Mesh, cos, pi, sin, specialcf, sqrt, x, y
from ngsolve.meshes import MakeStructured2DMesh

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import elasticity  # noqa: E402
from stokes_stress import P, U, errors, manufactured, solve  # noqa: E402

BCS = {"dirichlet": (".*", False), "mixed": ("left|bottom", True)}


def run(mesh, k, mu, u, p, bc="dirichlet"):
    dirichlet, traction = BCS[bc]
    sigma, gamma, f = manufactured(u, p, mu)
    gN = sigma * specialcf.normal(2) if traction else None
    sol, p_h, m_h = solve(mesh, k, mu, f, u, gN, dirichlet)
    return errors(mesh, sol, p_h, u, p, sigma, gamma), m_h, sol


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("k", [1, 2])
def test_patch(k, bc):
    """Divergence-free velocity of degree k, zero-mean pressure of degree k - 1: sigma, p,
    u, gamma to roundoff (observed at most 1.7e-13); m_h = 0 (observed 3e-16)."""
    u, p = {1: (CF((x + 2 * y, 3 * x - y)), CF(0.0)),
            2: (CF((x * x - 2 * x * y + y * y + 1, -2 * x * y + y * y + x)), x - 0.5 + 2 * (y - 0.5))}[k]
    e, m_h, _ = run(MakeStructured2DMesh(quads=False, nx=3, ny=3), k, 1.0, u, p, bc)
    assert max(e[:4]) < 1e-10
    assert m_h is None or abs(m_h) < 1e-12


# Unstructured meshes of the unit square, mu = 1, Dirichlet, maxh 0.1 -> 0.05. Observed EOC
# (sigma, p, u, gamma; u*): k = 1: 2.03, 1.94, 2.07, 1.90; 2.97;
# k = 2: 3.09, 3.22, 3.08, 2.99; 4.05. m_h is roundoff (observed below 4e-16).
@pytest.mark.parametrize("k", [1, 2])
def test_rates(k):
    from netgen.geom2d import unit_square
    out = [run(Mesh(unit_square.GenerateMesh(maxh=h)), k, 1.0, U, P) for h in (0.1, 0.05)]
    (e1, m1, _), (e2, m2, _) = out
    for a, b in zip(e1[:4], e2[:4]):
        assert log2(a / b) > k + 0.7
    assert log2(e1[5] / e2[5]) > k + 1.8
    assert abs(m1) < 1e-12 and abs(m2) < 1e-12


@pytest.mark.parametrize("k", [1, 2])
def test_limit_of_elasticity(k):
    """The elastic solver with data bounded in lambda (u = curl psi + u_1/lambda) tends to
    the Stokes solver with u = curl psi, p = -div u_1 = -2 pi cos(pi x) cos(pi y) and the same
    mu: lambda |sigma_e - sigma_s| = 2.10, lambda |u_e - u_s| = 0.71 at lambda = 1.7e3 and
    1.7e5 (N = 8, k = 1, 2)."""
    mesh = MakeStructured2DMesh(quads=False, nx=8, ny=8)
    us = CF((sin(pi * x) * cos(pi * y), -cos(pi * x) * sin(pi * y)))
    for nu in (0.4999, 0.499999):
        lam, mu = elasticity.lame(1.0, nu)
        ue = elasticity.bounded_data(lam)
        se, ge, fe = elasticity.manufactured(ue, lam, mu)
        sol_e = elasticity.solve(mesh, k, lam, mu, fe, ue)
        _, _, sol_s = run(mesh, k, mu, us, -2 * pi * cos(pi * x) * cos(pi * y))
        d = [sqrt(Integrate(InnerProduct(a - b, a - b), mesh)) for a, b in zip(sol_e[:2], sol_s[:2])]
        assert lam * d[0] < 2.5 and lam * d[1] < 1.0


@pytest.mark.parametrize("k", [1, 2])
def test_mu_scaling(k):
    """With p proportional to mu (sigma and f proportional to mu, tau_F = mu) the discrete
    velocity does not depend on mu and the stress scales with it."""
    mesh = MakeStructured2DMesh(quads=False, nx=4, ny=4)
    e1, _, _ = run(mesh, k, 1.0, U, P)
    e2, _, _ = run(mesh, k, 1e-3, U, 1e-3 * P)
    assert abs(e2[2] / e1[2] - 1) < 1e-6 and abs(e2[3] / e1[3] - 1) < 1e-6
    assert abs(e2[0] / e1[0] - 1e-3) < 1e-9 and abs(e2[1] / e1[1] - 1e-3) < 1e-9


@pytest.mark.parametrize("k", [1, 2])
def test_bordered_solve(k, monkeypatch):
    """The bordered solve (multiplier m and one element mean of tr sigma eliminated by a 2 x 2
    Schur complement) gives the solution of the factorization of the whole system."""
    import stokes_stress
    mesh = MakeStructured2DMesh(quads=False, nx=4, ny=4)
    e1, m1, _ = run(mesh, k, 1.0, U, P)
    monkeypatch.setattr(stokes_stress, "multiplier_inverse",
                        lambda fes, rho, m, factor=None: (lambda mat, free: mat.Inverse(free)))
    e2, m2, _ = run(mesh, k, 1.0, U, P)
    assert max(abs(a / b - 1) for a, b in zip(e1, e2)) < 1e-10 and abs(m1 - m2) < 1e-12

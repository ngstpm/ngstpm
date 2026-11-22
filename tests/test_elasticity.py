import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, InnerProduct, Integrate, cos, pi, sin, specialcf, x, y
from ngsolve.meshes import MakeStructured2DMesh

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from elasticity import bounded_data, errors, lame, manufactured, solve  # noqa: E402

LAM, MU = lame(1.0, 0.3)
BCS = {"dirichlet": (".*", False), "mixed": ("left|bottom", True)}


def run(N, k, u, bc):
    dirichlet, traction = BCS[bc]
    mesh = MakeStructured2DMesh(quads=False, nx=N, ny=N)
    sigma, gamma, f = manufactured(u, LAM, MU)
    gN = sigma * specialcf.normal(2) if traction else None
    sol = solve(mesh, k, LAM, MU, f, u, gN, dirichlet)
    return mesh, sol, errors(mesh, sol, u, sigma, gamma)


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("k", [1, 2])
def test_patch(k, bc):
    """A displacement of degree k is reproduced to roundoff."""
    u = {1: CF((1 + 2 * x - y, -3 + x + 4 * y)),
         2: CF((x * x - 2 * x * y + y, 3 * y * y + x * y - x))}[k]
    _, _, (es, eu, eg, *_) = run(3, k, u, bc)
    assert max(es, eu, eg) < 1e-10


# Asymptotic pairs on structured meshes, E = 1, nu = 0.3, Dirichlet, tau_S = mu.
# Observed EOC (sigma, u, gamma): k = 1, N 16 -> 32: 1.96, 1.99, 1.79 (gamma is
# still preasymptotic, 1.91 at 32 -> 64); k = 2, N 8 -> 16: 2.99, 2.99, 2.91.
@pytest.mark.parametrize("k, N", [(1, 16), (2, 8)])
def test_rates(k, N):
    u = CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y)))
    e1 = run(N, k, u, "dirichlet")[2]
    e2 = run(2 * N, k, u, "dirichlet")[2]
    for a, b in zip(e1[:3], e2[:3]):
        assert log2(a / b) > k + 0.7


def test_weak_symmetry():
    """On each element the skew part of sigma_h is L2-orthogonal to P_k, yet nonzero."""
    u = CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y)))
    for k, tests in ((1, (CF(1), x, y)), (2, (CF(1), x, y, x * x, x * y, y * y))):
        mesh, (sigma_h, *_), _ = run(4, k, u, "dirichlet")
        skew = sigma_h[0, 1] - sigma_h[1, 0]
        for q in tests:
            assert max(abs(v) for v in Integrate(skew * q, mesh, element_wise=True)) < 1e-12
        assert Integrate(skew * skew, mesh) > 1e-8


@pytest.mark.parametrize("k", [1, 2])
def test_nearly_incompressible(k):
    """With data bounded in lambda (u = curl psi + u_1/lambda) and tau_S = mu, the errors
    at nu = 0.49999999 are those at nu = 0.3 up to a factor (observed ratios: 0.50 to 1.44)."""
    for N in (8, 16):
        mesh = MakeStructured2DMesh(quads=False, nx=N, ny=N)
        errs = []
        for nu in (0.3, 0.49999999):
            lam, mu = lame(1.0, nu)
            u = bounded_data(lam)
            sigma, gamma, f = manufactured(u, lam, mu)
            errs.append(errors(mesh, solve(mesh, k, lam, mu, f, u), u, sigma, gamma))
        for a, b in zip(*errs):
            assert b < 2 * a


# Unstructured meshes of the unit square, E = 1, nu = 0.3, Dirichlet. Observed EOC of
# |u - u*_h|: k = 1, maxh 0.05 -> 0.025: 2.99 (3.13 at 0.1 -> 0.05; on structured meshes
# k = 1 approaches 3 slowly, 2.63 and 2.82 at N 8 -> 16 -> 32); k = 2, 0.1 -> 0.05: 4.18.
@pytest.mark.parametrize("k, h", [(1, 0.05), (2, 0.1)])
def test_postprocessing_rate(k, h):
    from netgen.geom2d import unit_square
    from ngsolve import Mesh

    u = CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y)))
    sigma, gamma, f = manufactured(u, LAM, MU)
    e = []
    for maxh in (h, h / 2):
        mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
        e.append(errors(mesh, solve(mesh, k, LAM, MU, f, u), u, sigma, gamma)[5])
    assert log2(e[0] / e[1]) > k + 2 - 0.2


@pytest.mark.parametrize("nu", [0.3, 0.49])
def test_pressure_split_is_exact(nu):
    """Keeping the element mean of tr(sigma) global (MATLAB's pressure_split, cf. its
    tests/test_solid_lambda_forms.m: k = 2, nu = 0.49) gives the same discrete solution."""
    lam, mu = lame(1.0, nu)
    u = CF((sin(x) * y, x * x * cos(y)))
    sigma, gamma, f = manufactured(u, lam, mu)
    mesh = MakeStructured2DMesh(quads=False, nx=4, ny=4)
    a = solve(mesh, 2, lam, mu, f, u)
    b = solve(mesh, 2, lam, mu, f, u, split=True)
    for p, q in zip(a[:3], b[:3]):
        assert Integrate(InnerProduct(p - q, p - q), mesh) ** 0.5 < 1e-10

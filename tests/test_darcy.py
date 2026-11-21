import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, Integrate, cos, pi, sin, specialcf, sqrt, x, y
from ngsolve.meshes import MakeStructured2DMesh

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import poisson  # noqa: E402
from darcy import darcy_data, half_square, solve  # noqa: E402

KAPPA = CF((2, 0.5, 0.5, 1), dims=(2, 2))
I2 = CF((1, 0, 0, 1), dims=(2, 2))


def l2(a, b, mesh):
    return sqrt(Integrate((a - b) ** 2, mesh, order=10))


def test_polynomial_reproduced_exactly_mixed():
    """Quadratic p, full tensor kappa: Dirichlet on the top side, u . n = gN (nonzero) on the others."""
    p = x * x - 2 * x * y + y + 1
    u, f = darcy_data(p, KAPPA)
    mesh = half_square(0.2)
    uh, ph, _ = solve(mesh, 2, KAPPA, f, p, u * specialcf.normal(2), dirichlet="top", tau=3.0)
    assert max(l2(uh, u, mesh), l2(ph, p, mesh)) < 1e-10


def test_identity_tensor_equals_poisson():
    """kappa = I: (u, p, phat) of Darcy are (q, u, uhat) of poisson.py."""
    p = sin(pi * x) * cos(pi * y)
    u, f = darcy_data(p, I2)
    mesh = MakeStructured2DMesh(quads=False, nx=6, ny=6)
    uh, ph, _ = solve(mesh, 2, I2, f, p, tau=2.0)
    qh, vh, _ = poisson.solve(mesh, 2, f, p, tau=2.0)
    assert max(l2(uh, qh, mesh), l2(ph, vh, mesh)) < 1e-12


@pytest.mark.parametrize("k", [1, 2])
def test_convergence_rate(k):
    p = cos(pi * x) * cos(pi * y)
    u, f = darcy_data(p, KAPPA)
    errs = []
    for maxh in (0.05, 0.025):
        mesh = half_square(maxh)
        uh, ph, _ = solve(mesh, k, KAPPA, f, p, u * specialcf.normal(2), dirichlet="top")
        errs.append((l2(uh, u, mesh), l2(ph, p, mesh)))
    for e0, e1 in zip(*errs):
        assert log2(e0 / e1) > k + 1 - 0.2

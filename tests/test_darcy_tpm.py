import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, Integrate, cos, dx, pi, sqrt, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from darcy import darcy_data  # noqa: E402
from darcy_tpm import GEOMETRIES, background_mesh, solve, solve_neumann  # noqa: E402

KAPPA = CF((2, 0.5, 0.5, 1), dims=(2, 2))


def l2(a, b, mesh, inner):
    return sqrt(Integrate((a - b) ** 2 * dx(definedonelements=inner, bonus_intorder=4), mesh))


def quadratic():
    p = lambda X, Y: X * X - 2 * X * Y + Y + 1
    u = lambda X, Y: -KAPPA * CF((2 * X - 2 * Y, -2 * X + 1))
    f = darcy_data(p(x, y), KAPPA)[1]
    return p, u, f


def test_polynomial_reproduced_exactly_dirichlet():
    p, u, f = quadratic()
    mesh = background_mesh(0.3)
    (uh, ph, _), inner = solve(mesh, 2, KAPPA, f, p, GEOMETRIES["disk"], tau=3.0)
    assert max(l2(uh, u(x, y), mesh, inner), l2(ph, p(x, y), mesh, inner)) < 1e-10


def test_polynomial_reproduced_exactly_neumann():
    """u . nu on the hole, p on the box."""
    p, u, f = quadratic()
    mesh = background_mesh(0.3)
    (uh, ph, _), inner = solve_neumann(mesh, 2, KAPPA, f, p, u, lambda X, Y: 0.6 - sqrt(X * X + Y * Y), tau=3.0)
    assert max(l2(uh, u(x, y), mesh, inner), l2(ph, p(x, y), mesh, inner)) < 1e-10


@pytest.mark.parametrize("k", [1, 2])
def test_convergence_rate(k):
    p = lambda X, Y: cos(pi * X) * cos(pi * Y)
    u, f = darcy_data(p(x, y), KAPPA)
    errs = []
    for maxh in (0.1, 0.05):
        mesh = background_mesh(maxh)
        (uh, ph, _), inner = solve(mesh, k, KAPPA, f, p, GEOMETRIES["disk"])
        errs.append((l2(uh, u, mesh, inner), l2(ph, p(x, y), mesh, inner)))
    for e0, e1 in zip(*errs):
        assert log2(e0 / e1) > k + 1 - 0.2

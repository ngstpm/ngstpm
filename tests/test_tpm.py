import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, Integrate, atan2, cos, dx, pi, sin, sqrt, x, y

import ngstpm  # noqa: F401  (adds the proxy verbs Extrapolate, PathIntegral, Transfer)

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson_tpm import background_mesh, solve  # noqa: E402


def disk(X, Y):
    return sqrt(X * X + Y * Y) - 1


def flower(X, Y):
    return sqrt(X * X + Y * Y) - 1 - 0.12 * cos(5 * atan2(Y, X))


@pytest.mark.parametrize("phi", [disk, flower])
def test_quadratic_reproduced_exactly(phi):
    """The transferred datum is exact for u in the discrete space."""
    g = lambda X, Y: X * X - 2 * X * Y + Y
    mesh = background_mesh(0.3)
    (qh, uh, _), inner = solve(mesh, 2, CF(-2.0), g, phi)     # -Delta g = -2
    err = sqrt(Integrate((uh - g(x, y)) ** 2 * dx(definedonelements=inner), mesh))
    assert err < 1e-11


def l2_error(uh, u, mesh, inner):
    return sqrt(Integrate((uh - u) ** 2 * dx(definedonelements=inner, bonus_intorder=4), mesh))


# ||u - u_h||_{L2(D_h)} at maxh = 0.1, cross-checked to 4e-14 relative
# against the independent auxiliary-unknown implementation (ngshdg.tpm).
REFERENCE = {
    (disk, 1): 1.1759e-02, (disk, 2): 3.9310e-04, (disk, 3): 2.1277e-05,
    (flower, 1): 1.1662e-02, (flower, 2): 3.8197e-04, (flower, 3): 2.6787e-05,
}


@pytest.mark.parametrize("phi,k", list(REFERENCE))
def test_reference_errors(phi, k):
    g = lambda X, Y: sin(pi * X) * cos(pi * Y)
    mesh = background_mesh(0.1)
    (qh, uh, _), inner = solve(mesh, k, 2 * pi**2 * g(x, y), g, phi)
    err = l2_error(uh, g(x, y), mesh, inner)
    assert err == pytest.approx(REFERENCE[(phi, k)], rel=1e-4)


@pytest.mark.parametrize("k", [1, 2])
def test_convergence_rate(k):
    g = lambda X, Y: sin(pi * X) * cos(pi * Y)
    errs = []
    for maxh in (0.1, 0.05):
        mesh = background_mesh(maxh)
        (qh, uh, _), inner = solve(mesh, k, 2 * pi**2 * g(x, y), g, disk)
        errs.append(l2_error(uh, g(x, y), mesh, inner))
    assert log2(errs[0] / errs[1]) > k + 0.8


@pytest.mark.parametrize("name,rate_from", [("annulus", 0.1), ("square with a hole", 0.2)])
def test_convergence_rate_geometries(name, rate_from):
    """Multiply connected domain; domain with a fitted and an unfitted boundary."""
    from poisson_tpm import GEOMETRIES
    g = lambda X, Y: sin(pi * X) * cos(pi * Y)
    errs = []
    for maxh in (rate_from, rate_from / 2):
        mesh = background_mesh(maxh)
        (qh, uh, _), inner = solve(mesh, 2, 2 * pi**2 * g(x, y), g, GEOMETRIES[name])
        errs.append(l2_error(uh, g(x, y), mesh, inner))
    assert log2(errs[0] / errs[1]) > 2.8


def test_quadratic_reproduced_exactly_3d():
    from ngsolve import z
    from poisson_tpm_3d import GEOMETRIES, background_mesh as background_mesh_3d
    g = lambda X, Y, Z: X * X - 2 * X * Y + Y * Z + Z
    mesh = background_mesh_3d(0.5)
    (qh, uh, _), inner = solve(mesh, 2, CF(-2.0), g, GEOMETRIES["ball"])
    assert sum(inner) > 20
    err = sqrt(Integrate((uh - g(x, y, z)) ** 2 * dx(definedonelements=inner), mesh))
    assert err < 1e-11


def test_neumann_quadratic_reproduced_exactly():
    """Transferred Neumann condition on the hole, Dirichlet on the box."""
    from poisson_tpm_neumann import solve as solve_neumann
    g = lambda X, Y: X * X - 2 * X * Y + Y
    gN = lambda X, Y: -CF((2 * X - 2 * Y, -2 * X + 1))
    phi = lambda X, Y: 0.6 - sqrt(X * X + Y * Y)
    mesh = background_mesh(0.3)
    (qh, uh, _), inner = solve_neumann(mesh, 2, CF(-2.0), g(x, y), gN, phi)
    err = sqrt(Integrate((uh - g(x, y)) ** 2 * dx(definedonelements=inner), mesh))
    assert err < 1e-11

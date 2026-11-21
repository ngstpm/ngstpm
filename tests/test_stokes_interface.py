import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, VOL, BitArray, InnerProduct, Integrate, Mesh, cos, dx, pi, sin, sqrt, x, y
from ngsolve.meshes import MakeStructured2DMesh

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import stokes  # noqa: E402
import stokes_interface  # noqa: E402
from poisson_interface import domains  # noqa: E402
from stokes import stokes_data  # noqa: E402
from stokes_interface import disk  # noqa: E402
from stokes_interface_two_meshes import meshes, phi, solve  # noqa: E402
from ngstpm import pair, transfer_paths  # noqa: E402

UEX = CF((sin(pi * x) * sin(pi * y), cos(pi * x) * cos(pi * y)))
PEX = sin(2 * pi * x) * sin(2 * pi * y)


def l2(a, b, mesh, D=None):
    return sqrt(Integrate(InnerProduct(a - b, a - b) * dx(definedonelements=D, bonus_intorder=4), mesh))


def test_matching_interface_equals_single_domain():
    """Zero path length on a mesh fitted to y = 1/2: the transmission conditions are
    the single-domain HDG's single trace and conservativity."""
    nu, tau = 0.7, 2.0
    L, f = stokes_data(UEX, PEX, nu)
    mesh = MakeStructured2DMesh(quads=False, nx=4, ny=4)       # no internal boundary on y = 1/2
    for i in range(4):
        mesh.ngmesh.SetBCName(i, "outer")
    mesh = Mesh(mesh.ngmesh)
    D1 = BitArray([sum(mesh[v].point[1] for v in el.vertices) > 1.5 for el in mesh.Elements(VOL)])
    sol, *_ = stokes_interface.solve(mesh, 2, phi, D1, ~D1, f, UEX, nu=nu, tau=tau)
    ref = stokes.solve(mesh, 2, f, UEX, nu=nu, tau=tau)
    for D, part in ((D1, sol[0:3]), (~D1, sol[4:7])):
        assert max(l2(a, b, mesh, D) for a, b in zip(part, ref[:3])) < 1e-12


def test_polynomial_reproduced_exactly_and_global_mean():
    """Two non-matching meshes; p = x + y - 1 has means -1/8 (lower half) and +1/8
    (upper half): only their sum is constrained."""
    nu, tau = 0.5, 3.0
    u = CF((x * x + y, -2 * x * y + x))
    p = x + y - 1
    L, f = stokes_data(u, p, nu)
    mesh1, mesh2 = meshes(0.25, 1.23 * 0.25)
    sol, *_ = solve(mesh1, mesh2, 2, phi, f, u, nu=nu, tau=tau)
    means = []
    for mesh, part in ((mesh1, sol[0:3]), (mesh2, sol[4:7])):
        assert max(l2(a, b, mesh) for a, b in zip(part, (L, u, p))) < 1e-10
        means.append(Integrate(part[2], mesh))
    assert means[0] == pytest.approx(0.125, abs=1e-12) and means[1] == pytest.approx(-0.125, abs=1e-12)
    assert abs(sum(means)) < 1e-13


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("gap", [0, 2])
def test_convergence_rate(k, gap):
    """No gap and a gap delta = h^2."""
    L, f = stokes_data(UEX, PEX, 1.0)
    errs = []
    for maxh in (0.1, 0.05):
        mesh1, mesh2 = meshes(maxh, 1.23 * maxh, maxh ** gap if gap else 0)
        sol, *_ = solve(mesh1, mesh2, k, phi, f, UEX)
        e = [l2(a, b, mesh1) ** 2 + l2(c, b, mesh2) ** 2
             for a, c, b in zip(sol[0:3], sol[4:7], (L, UEX, PEX))]
        errs.append([sqrt(ei) for ei in e])
    for e0, e1 in zip(*errs):
        assert log2(e0 / e1) > k + 1 - 0.2


def region_area(pieces, paths):
    """|D_1| + |D_2| + the two slivers."""
    I = pair(*paths)
    return sum(Integrate(CF(1) * dx(definedonelements=D), mesh) + Integrate(CF(1).RegionIntegral(p) * I.ds(side=i + 1), mesh)
               for i, ((mesh, D), p) in enumerate(zip(pieces, paths)))


@pytest.mark.parametrize("family,tol", [("fitted", 1e-9), ("unfitted", 1e-9)])
def test_area_disk(family, tol):
    """The slivers fill the box (|Omega| = 9); on the fitted mesh they have opposite signs."""
    mesh, D1, D2 = domains(family, 0.2, disk)
    paths = [transfer_paths(mesh, disk, D) for D in (D1, D2)]
    assert abs(region_area(((mesh, D1), (mesh, D2)), paths) - 9) < tol


def test_area_flat_gap():
    mesh1, mesh2 = meshes(0.2, 0.246, 0.1)
    paths = [transfer_paths(m, phi, facets=m.Boundaries("interface")) for m in (mesh1, mesh2)]
    assert abs(region_area(((mesh1, None), (mesh2, None)), paths) - 1) < 1e-13


def polynomial_data(p):
    nu, tau = 0.5, 3.0
    u = CF((x * x + y, -2 * x * y + x))
    return u, p, stokes_data(u, p, nu), nu, tau


def test_polynomial_reproduced_exactly_flat_gap():
    """An O(h) gap between two meshes: p = y^2 - 1/3 has zero mean on the unit square but not
    on the meshes, so ptilde alone is not exact; ptilde + alpha (the algebraic row) is."""
    u, p, (L, f), nu, tau = polynomial_data(y * y - 1 / 3)
    mesh1, mesh2 = meshes(0.25, 1.23 * 0.25, 0.125)
    sol, _, alpha = solve(mesh1, mesh2, 2, phi, f, u, nu=nu, tau=tau)
    for mesh, part in ((mesh1, sol[0:3]), (mesh2, sol[4:7])):
        assert max(l2(a, b, mesh) for a, b in zip(part, (L, u, p))) < 1e-10
        assert l2(part[2] - alpha, p, mesh) > 1e-3


def test_polynomial_reproduced_exactly_disk_unfitted():
    """The same on one mesh, with alpha an unknown of the system."""
    u, p, (L, f), nu, tau = polynomial_data(x * x - 0.75)             # zero mean on (-1.5, 1.5)^2
    mesh, D1, D2 = domains("unfitted", 0.3, disk)
    sol, _, alpha = stokes_interface.solve(mesh, 2, disk, D1, D2, f, u, nu=nu, tau=tau)
    for D, part in ((D1, sol[0:3]), (D2, sol[4:7])):
        assert max(l2(a, b, mesh, D) for a, b in zip(part, (L, u, p))) < 5e-9
        assert l2(part[2] - alpha, p, mesh, D) > 1e-2

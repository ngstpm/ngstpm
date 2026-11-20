"""Interface problems on two meshes (ngstpm/twomesh.py): I.Space, I.BilinearForm,
I.LinearForm, I.GridFunction with the weak form of examples/poisson_interface.py.
Omega_1 = the unit disk and Omega_2 = the box minus the disk, each meshed on its
own with straight elements: Gamma_h^1, Gamma_h^2 are non-matching polygons."""

from math import log2
from pathlib import Path
import sys

import pytest
from netgen.geom2d import SplineGeometry
from ngsolve import (BND, CF, BitArray, BilinearForm, FacetFESpace, Integrate, L2, Mesh, VectorL2, cos,
                     div, dx, exp, grad, pi, sin, specialcf, sqrt, x, y)

from ngstpm import interior_ds, pair, transfer_paths

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson import solve_linear_system  # noqa: E402

KAPPA = (1.0, 10.0)


def phi(X, Y):
    return sqrt(X * X + Y * Y) - 1


def meshes(maxh1, maxh2):
    g1 = SplineGeometry()
    g1.AddCircle((0, 0), 1, bc="interface")
    g2 = SplineGeometry()
    g2.AddRectangle((-1.5, -1.5), (1.5, 1.5), bc="outer")
    g2.AddCircle((0, 0), 1, bc="interface", leftdomain=0, rightdomain=1)
    return Mesh(g1.GenerateMesh(maxh=maxh1)), Mesh(g2.GenerateMesh(maxh=maxh2))


def boundary(mesh, name):
    """The facets of mesh on the boundary condition name (in 2D a BND element's facet is its edge)."""
    F = BitArray(mesh.nfacet)
    F.Clear()
    for el in mesh.Elements(BND):
        if el.mat == name:
            F[el.edges[0].nr] = True
    return F


def solve(mesh1, mesh2, order, f1, f2, gD, gN, g, condense=True, tau=1.0):
    kappa1, kappa2 = (lambda X, Y, c=c: CF(c) for c in KAPPA)
    p1 = transfer_paths(mesh1, phi, facets=mesh1.Boundaries("interface"))   # D: the whole mesh
    p2 = transfer_paths(mesh2, phi, facets=mesh2.Boundaries("interface"))
    I = pair(p1, p2)
    bonus = max(6, 2 * order)
    dI = [I.ds(side=i, bonus_intorder=bonus) for i in (1, 2)]

    fes1 = VectorL2(mesh1, order=order) * L2(mesh1, order=order) * FacetFESpace(mesh1, order=order)
    fes2 = (VectorL2(mesh2, order=order) * L2(mesh2, order=order)
            * FacetFESpace(mesh2, order=order, dirichlet="outer"))
    fes = I.Space(fes1, fes2)
    (q1, u1, uhat1, q2, u2, uhat2), (r1, v1, mu1, r2, v2, mu2) = fes.TnT()
    lhs = I.BilinearForm(fes, condense=condense)
    rhs = I.LinearForm(fes)
    n = specialcf.normal(2)

    for q, u, uhat, r, v, mu, kappa, f, D in ((q1, u1, uhat1, r1, v1, mu1, kappa1, f1, p1.elements),
                                              (q2, u2, uhat2, r2, v2, mu2, kappa2, f2, p2.elements)):
        mesh = ComponentMesh(q)
        dV = dx(definedonelements=D)
        dS = dx(element_boundary=True, definedonelements=D)
        qhat_n = q*n + tau*(u - uhat)
        lhs += 1/kappa(x, y)*q*r*dV - u*div(r)*dV + uhat*(r*n)*dS
        lhs += -q*grad(v)*dV + qhat_n*v*dS
        rhs += f*v*dV
        lhs += qhat_n*mu*interior_ds(mesh, D)

    xbar, nubar = p1.target, p1.target_normal
    qn1 = q1.Extrapolate(xbar)*nubar + tau*(u1 - uhat1)
    qn2 = q2.Extrapolate(xbar)*nubar - tau*(u2.Extrapolate() - uhat2.Extrapolate())
    lhs += (qn1 - qn2) * mu1 * dI[0]
    rhs += gN(*xbar) * nubar * mu1 * dI[0]

    xbar = p2.target
    ub1 = uhat1.Transfer(p1, gradient=-q1/kappa1(x, y))
    ub2 = uhat2.Transfer(p2, gradient=-q2/kappa2(x, y))
    lhs += (ub1 - ub2) * mu2 * dI[1]
    rhs += gD(*xbar) * mu2 * dI[1]

    gfu = I.GridFunction(fes)
    gfu.components[5].Set(g, definedon=mesh2.Boundaries("outer"))
    solve_linear_system(lhs, rhs, gfu)
    return gfu.components


def ComponentMesh(proxy):
    from ngstpm import ComponentSpace
    return ComponentSpace(proxy).mesh


def errors(mesh1, mesh2, sol, u1, u2, q1, q2):
    e = lambda a, b, m: Integrate((a - b) ** 2 * dx(bonus_intorder=4), m)
    return (sqrt(e(sol[1], u1(x, y), mesh1) + e(sol[4], u2(x, y), mesh2)),
            sqrt(e(sol[0], q1(x, y), mesh1) + e(sol[3], q2(x, y), mesh2)))


def data(u1, u2, q1, q2):
    f1 = q1(x, y)[0].Diff(x) + q1(x, y)[1].Diff(y)
    f2 = q2(x, y)[0].Diff(x) + q2(x, y)[1].Diff(y)
    return f1, f2, (lambda X, Y: u1(X, Y) - u2(X, Y)), (lambda X, Y: q1(X, Y) - q2(X, Y)), u2(x, y)


def polynomial_solution(k):
    k1, k2 = KAPPA
    u1 = lambda X, Y: X**k - 2 * X * Y**(k - 1) + Y
    u2 = lambda X, Y: 0.5 * Y**k + 3 * X**(k - 1) * Y - X + 2
    q1 = lambda X, Y: -k1 * CF((k * X**(k - 1) - 2 * Y**(k - 1), -2 * (k - 1) * X * Y**(k - 2) + 1))
    q2 = lambda X, Y: -k2 * CF((3 * (k - 1) * X**(k - 2) * Y - 1, 0.5 * k * Y**(k - 1) + 3 * X**(k - 1)))
    return u1, u2, q1, q2


def smooth_solution():
    k1, k2 = KAPPA
    u1 = lambda X, Y: sin(pi * X) * cos(pi * Y)
    u2 = lambda X, Y: exp(0.5 * X) * cos(0.5 * Y) + X * Y
    q1 = lambda X, Y: -k1 * CF((pi * cos(pi * X) * cos(pi * Y), -pi * sin(pi * X) * sin(pi * Y)))
    q2 = lambda X, Y: -k2 * CF((0.5 * exp(0.5 * X) * cos(0.5 * Y) + Y, -0.5 * exp(0.5 * X) * sin(0.5 * Y) + X))
    return u1, u2, q1, q2


@pytest.mark.parametrize("k", [1, 2, 3])
def test_patch_two_meshes(k):
    """Piecewise polynomials of degree k are reproduced: |u - uh| <= 5e-13 and
    |q - qh| <= 7e-12 measured (maxh 0.4 and 0.5)."""
    sol_ex = polynomial_solution(k)
    mesh1, mesh2 = meshes(0.4, 0.5)
    sol = solve(mesh1, mesh2, k, *data(*sol_ex))
    eu, eq = errors(mesh1, mesh2, sol, *sol_ex)
    assert eu < 1e-10 and eq < 1e-9


def test_condensed_equals_uncondensed():
    """Condensing the interface coupling (D16) gives the uncondensed solution."""
    sol_ex = smooth_solution()
    mesh1, mesh2 = meshes(0.3, 0.37)
    a = solve(mesh1, mesh2, 2, *data(*sol_ex), condense=True)
    a = [c.vec.FV().NumPy().copy() for c in a]
    b = solve(mesh1, mesh2, 2, *data(*sol_ex), condense=False)
    b = [c.vec.FV().NumPy() for c in b]
    for ca, cb in zip(a, b):
        assert abs(ca - cb).max() <= 1e-10 * max(abs(cb).max(), 1)


def test_convergence_rate():
    """Order k + 1 for u, k = 2 (measured rate 3.12 between maxh 0.2 and 0.1,
    |u - uh| = 4.12e-4 at maxh 0.1)."""
    sol_ex = smooth_solution()
    e = []
    for maxh in (0.2, 0.1):
        mesh1, mesh2 = meshes(maxh, 1.23 * maxh)
        e.append(errors(mesh1, mesh2, solve(mesh1, mesh2, 2, *data(*sol_ex)), *sol_ex)[0])
    assert 2.8 < log2(e[0] / e[1]) < 3.4


def test_compound_of_two_meshes_raises():
    mesh1, mesh2 = meshes(0.5, 0.5)
    with pytest.raises(ValueError, match="two meshes"):
        L2(mesh1, order=1) * L2(mesh2, order=1)


def test_one_mesh_gives_ngsolve_objects():
    """On one mesh I.Space is the compound space and I.BilinearForm NGSolve's."""
    from netgen.geom2d import unit_square
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    D1 = BitArray([sum(mesh[v].point[0] for v in el.vertices) / 3 < 0.5 for el in mesh.Elements()])
    p1 = transfer_paths(mesh, lambda X, Y: X - 0.5, D1)
    p2 = transfer_paths(mesh, lambda X, Y: X - 0.5, ~D1)
    I = pair(p1, p2)
    fes = I.Space(L2(mesh, order=1), L2(mesh, order=1))
    assert len(fes.components) == 2
    assert isinstance(I.BilinearForm(fes), BilinearForm)


def test_region_facets_equal_bitarray_facets():
    """facets=mesh.Boundaries("interface") gives the same paths and Interface as
    the facet BitArray: identical segments (exactly)."""
    import numpy as np
    from ngstpm import boundary_ds
    mesh1, mesh2 = meshes(0.25, 0.3)
    a = pair(*(transfer_paths(m, phi, facets=m.Boundaries("interface")) for m in (mesh1, mesh2)))
    b = pair(*(transfer_paths(m, phi, facets=boundary(m, "interface")) for m in (mesh1, mesh2)))
    Sa, Sb = a.segments(), b.segments()
    assert a.nsegments == b.nsegments == 40
    for name in Sa._fields:
        assert np.array_equal(getattr(Sa, name), getattr(Sb, name))
    for m in (mesh1, mesh2):
        h = CF(x * x + y)
        assert (Integrate(h * boundary_ds(m, facets=m.Boundaries("interface")), m)
                == Integrate(h * boundary_ds(m, facets=boundary(m, "interface")), m) != 0)

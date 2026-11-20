"""The local parts of the interface measure (src/ifclinear.cpp): a linear form and
Integrate over the pieces of Gamma_h^side, on the points and weights of the
piece rules (I.points), with the binding set (path data at the integration
point). Through the private helpers until I.ds is wired to them."""

from pathlib import Path
import sys

import numpy as np
import pytest
from ngsolve import L2, FacetFESpace, GridFunction, Integrate, LinearForm, x, y

from ngstpm import ParametricCurve, boundary_ds, pair, transfer_paths
from ngstpm._ngstpm import _AddInterfaceLF, _IntegrateInterface

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson_interface import domains  # noqa: E402
from test_pairing import disk, flower_xy, interface_facets, two_meshes  # noqa: E402


def g(X, Y):                                     # data, evaluated at the end of the path
    return X * X - 2 * X * Y + 3 * Y + 1


def poly(X, Y, k):                               # in the test space of order k
    return X**k - 2 * X * Y**(k - 1) + 1


def setups():
    geo = ParametricCurve(flower_xy)
    mesh, D1, D2 = domains("unfitted", 0.3, geo)
    yield "unfitted flower", transfer_paths(mesh, geo, D1), transfer_paths(mesh, geo, D2)
    m1, m2 = two_meshes()
    yield ("two meshes", transfer_paths(m1, disk, facets=interface_facets(m1)),
           transfer_paths(m2, disk, facets=interface_facets(m2)))


@pytest.mark.parametrize("case", ["unfitted flower", "two meshes"])
@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("k", [1, 2])
def test_linear_form_is_the_piece_rule(case, side, k):
    """lf(p) = sum over I.points(side, 2k + bonus) of weight * g(xbar) * p(x), to
    1e-12 relative, for p in the space (L2 of order k): the same points and
    weights, and the path data evaluated on K."""
    p1, p2 = dict((c, (a, b)) for c, a, b in setups())[case]
    I = pair(p1, p2)
    p = (p1, p2)[side - 1]
    bonus = 3
    fes = L2(p.mesh, order=k)
    v = fes.TestFunction()
    lf = LinearForm(fes)
    _AddInterfaceLF(lf, I, side, g(*p.target) * v, bonus)
    lf.Assemble()
    u = GridFunction(fes)
    u.Set(poly(x, y, k))
    P = I.points(side, 2 * k + bonus)
    ref = np.sum(P.weight * g(*P.xbar.T) * poly(*P.x.T, k))
    assert abs(ref) > 0.1
    assert abs(lf.vec.InnerProduct(u.vec) - ref) < 1e-12 * abs(ref)


@pytest.mark.parametrize("side", [1, 2])
def test_integrate_is_the_piece_rule(side):
    """Integrate = sum over I.points(side, 5 + bonus) of weight * g(xbar)."""
    for case, p1, p2 in setups():
        I = pair(p1, p2)
        p = (p1, p2)[side - 1]
        P = I.points(side, 5 + 2)
        ref = np.sum(P.weight * g(*P.xbar.T))
        assert abs(_IntegrateInterface(I, side, g(*p.target), 2) - ref) < 1e-12 * abs(ref)


def test_fitted_equals_boundary_ds():
    """On a fitted mesh the pieces tile Gamma_h exactly, so a polynomial integrand
    gives the boundary_ds values, for Integrate and for the linear form (k = 2)."""
    mesh, D1, D2 = domains("fitted", 0.3, disk)
    p1, p2 = transfer_paths(mesh, disk, D1), transfer_paths(mesh, disk, D2)
    I = pair(p1, p2)
    fes = FacetFESpace(mesh, order=2)
    mu = fes.TestFunction()
    h = g(x, y)                                  # degree 2: exact under both rules
    for side, D in ((1, D1), (2, D2)):
        ref = Integrate(h * boundary_ds(mesh, D), mesh)
        assert abs(ref) > 1
        assert abs(_IntegrateInterface(I, side, h) - ref) < 1e-12 * abs(ref)
        a, b = LinearForm(fes), LinearForm(fes)
        _AddInterfaceLF(a, I, side, h * mu, 0)
        b += h * mu * boundary_ds(mesh, D)
        a.Assemble()
        b.Assemble()
        A, B = a.vec.FV().NumPy(), b.vec.FV().NumPy()
        assert np.abs(B).max() > 0.01 and np.abs(A - B).max() < 1e-12 * np.abs(B).max()


def test_trial_function_raises():
    mesh, D1, D2 = domains("fitted", 0.3, disk)
    I = pair(transfer_paths(mesh, disk, D1), transfer_paths(mesh, disk, D2))
    fes = L2(mesh, order=1)
    lf = LinearForm(fes)
    with pytest.raises(Exception, match="trial function"):
        _AddInterfaceLF(lf, I, 1, fes.TrialFunction() * fes.TestFunction())

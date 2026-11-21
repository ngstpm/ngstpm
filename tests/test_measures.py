"""interior_ds / boundary_ds: facet measures of an element set D (API-PROPOSAL D12)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from ngsolve import (
    VOL, BilinearForm, FacetFESpace, GridFunction, Integrate, L2, LinearForm, VectorL2,
    cos, div, dx, exp, grad, pi, sin, specialcf, sqrt, x, y,
)

from ngstpm import boundary_ds, interior_ds, restrict, transfer_paths, unfitted_domain

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson import solve_linear_system  # noqa: E402
from poisson_tpm import GEOMETRIES, background_mesh  # noqa: E402

CASES = ["disk", "square with a hole"]           # the second touches the mesh boundary


def reference_lengths(mesh, D):
    """2 x (length of the faces with both neighbours in D), perimeter of Gamma_h
    (one neighbour in D, one element outside), and the length of the BND faces of D."""
    P = np.array([v.point for v in mesh.vertices])
    inside = np.array([D[i] for i in range(mesh.ne)])
    interior = gamma = bnd = 0.0
    for f in mesh.facets:
        a, b = (v.nr for v in f.vertices)
        length = np.linalg.norm(P[a] - P[b])
        nb = [inside[el.nr] for el in f.elements]
        if len(nb) == 2:
            interior += 2 * length * (nb[0] and nb[1])
            gamma += length * (nb[0] != nb[1])
        else:
            bnd += length * nb[0]
    return interior, gamma, bnd


@pytest.mark.parametrize("name", CASES)
def test_lengths(name):
    """Integrate(1 * dS) is twice the interior face length, Integrate(1 * dG) the
    perimeter of Gamma_h; BND faces are in neither."""
    mesh = background_mesh(0.3)
    D = unfitted_domain(mesh, GEOMETRIES[name])[0]
    interior, gamma, bnd = reference_lengths(mesh, D)
    assert gamma > 1 and interior > 1
    assert (bnd > 1) == (name == "square with a hole")
    assert Integrate(1 * interior_ds(mesh, D), mesh) == pytest.approx(interior, rel=1e-13)
    assert Integrate(1 * boundary_ds(mesh, D), mesh) == pytest.approx(gamma, rel=1e-13)


def dirichlet_tpm(mesh, phi, k, new, tau=1.0):
    """HDG + TPM Dirichlet problem of examples/poisson_tpm.py, with the old
    indicator measures ((1 - ind) * dS, ind * dG) or with interior_ds /
    boundary_ds; the same integrands (uhat.Transfer) under both."""
    g = lambda X, Y: sin(pi * X) * cos(pi * Y)
    f = 2 * pi**2 * g(x, y)
    inner, gamma, ind = unfitted_domain(mesh, phi)
    p = transfer_paths(mesh, phi, inner)
    bonus = max(6, 2 * k)
    n = specialcf.normal(2)
    dV = dx(definedonelements=inner)
    dS = dx(element_boundary=True, definedonelements=inner)
    fes = (restrict(VectorL2(mesh, order=k), inner) * restrict(L2(mesh, order=k), inner)
           * restrict(FacetFESpace(mesh, order=k, dirichlet=".*"), inner))
    (q, u, uhat), (r, v, mu) = fes.TnT()
    qhat_n = q * n + tau * (u - uhat)
    ub = uhat.Transfer(p, gradient=-q)
    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    lhs += q * r * dV - u * div(r) * dV + uhat * (r * n) * dS
    lhs += -q * grad(v) * dV + qhat_n * v * dS
    rhs += f * v * dV
    if new:
        dI, dG = interior_ds(mesh, inner), boundary_ds(mesh, inner, bonus_intorder=bonus)
        lhs += qhat_n * mu * dI
        lhs += ub * mu * dG
        rhs += g(*p.target) * mu * dG
    else:
        dG = dx(element_boundary=True, definedonelements=inner, bonus_intorder=bonus)
        lhs += (1 - ind) * qhat_n * mu * dS
        lhs += ind * ub * mu * dG
        rhs += ind * g(*p.target) * mu * dG
    gfu = GridFunction(fes)
    gfu.components[2].Set(g(x, y), definedon=mesh.Boundaries(".*"))
    solve_linear_system(lhs, rhs, gfu)
    free = np.array(list(fes.FreeDofs(coupling=True)), dtype=bool)
    A = lhs.mat.ToDense().NumPy()[np.ix_(free, free)]
    return A, rhs.vec.FV().NumPy()[free], gfu.vec.FV().NumPy().copy()


@pytest.mark.parametrize("name", CASES)
def test_dirichlet_tpm_same_as_indicator_form(name):
    """The Dirichlet TPM solve with interior_ds / boundary_ds gives the same
    condensed matrix and right-hand side on the free dofs, and the same
    solution, as the indicator form (poisson_tpm.py before D12)."""
    mesh = background_mesh(0.3)
    A0, b0, u0 = dirichlet_tpm(mesh, GEOMETRIES[name], 2, new=False)
    A1, b1, u1 = dirichlet_tpm(mesh, GEOMETRIES[name], 2, new=True)
    assert np.abs(A1 - A0).max() <= 1e-13 * np.abs(A0).max()
    assert np.abs(b1 - b0).max() <= 1e-13 * np.abs(b0).max()
    assert np.abs(u1 - u0).max() <= 1e-11 * np.abs(u0).max()


@pytest.mark.parametrize("measure", [interior_ds, boundary_ds])
def test_bonus_intorder_passes_through(measure):
    """bonus_intorder given to the constructor or to a later call gives the
    same value as dx(element_boundary=True, ..., bonus_intorder) with the
    indicator; it changes the value for a non-polynomial integrand."""
    mesh = background_mesh(0.3)
    D = unfitted_domain(mesh, GEOMETRIES["disk"])[0]
    dM = measure(mesh, D)
    ind = dM.ind
    h = exp(x) * sin(3 * y)
    for b in (0, 5):
        ref = Integrate(ind * h * dx(element_boundary=True, definedonelements=D, bonus_intorder=b), mesh)
        assert Integrate(h * measure(mesh, D, bonus_intorder=b), mesh) == ref
        assert Integrate(h * dM(bonus_intorder=b), mesh) == ref
    assert Integrate(h * dM, mesh) != Integrate(h * dM(bonus_intorder=5), mesh)


def test_boundary_ds_on_given_facets():
    """boundary_ds(mesh, D, facets=F) on one side of two meshes (D the whole
    mesh, F the BND facets of the interface) gives the polygon length of
    Gamma_h, and with F = Gamma_h of D the same as without facets."""
    from netgen.geom2d import SplineGeometry
    from ngsolve import BND, BitArray, Mesh
    from ngstpm import boundary_facets
    g = SplineGeometry()
    g.AddRectangle((-1.5, -1.5), (1.5, 1.5), bc="outer")
    g.AddCircle((0, 0), 1, bc="interface", leftdomain=0, rightdomain=1)
    mesh = Mesh(g.GenerateMesh(maxh=0.3))
    D = BitArray(mesh.ne)
    D.Set()
    F = BitArray(mesh.nfacet)
    F.Clear()
    length = 0.0
    for el in mesh.Elements(BND):
        if el.mat == "interface":
            F[el.edges[0].nr] = True
            a, b = (np.array(mesh[v].point) for v in el.vertices)
            length += np.linalg.norm(b - a)
    assert 6.2 < length < 2 * pi
    assert Integrate(1 * boundary_ds(mesh, D, facets=F), mesh) == pytest.approx(length, rel=1e-13)
    assert Integrate(1 * boundary_ds(mesh, D), mesh) == 0.0                 # no Gamma_h inside the mesh
    mesh = background_mesh(0.3)
    D = unfitted_domain(mesh, GEOMETRIES["disk"])[0]
    F = boundary_facets(mesh, D)[0]
    h = exp(x) * sin(3 * y)
    assert Integrate(h * boundary_ds(mesh, D, facets=F), mesh) == Integrate(h * boundary_ds(mesh, D), mesh)



def test_nan_away_from_gamma_h_does_not_leak():
    """boundary_ds does not evaluate the integrand on elements of D without a facet
    on Gamma_h: a NaN there (e.g. grad phi = 0/0 at the centre of a disk, which
    made the TPM matrix singular on a 13 x 13 quad mesh) must not reach the
    integral as 0 * NaN."""
    mesh = background_mesh(0.2)
    D = unfitted_domain(mesh, GEOMETRIES["disk"])[0]
    far = next(el.nr for el in mesh.Elements(VOL) if D[el.nr]
               and all(D[e.nr] for v in el.vertices for e in mesh[v].elements))   # no facet on Gamma_h
    c = GridFunction(L2(mesh, order=0))
    c.vec[:] = 1.0
    c.vec[far] = float("nan")
    assert np.isfinite(Integrate(c * boundary_ds(mesh, D), mesh))

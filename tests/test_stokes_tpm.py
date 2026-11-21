import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import (CF, BilinearForm, FacetFESpace, GridFunction, InnerProduct, Integrate, L2, LinearForm,
                     MatrixValued, cos, dx, pi, sin, sqrt, x, y)

import ngstpm  # noqa: F401  (adds the proxy verbs Extrapolate, PathIntegral, Transfer)
from ngstpm import boundary_ds, restrict, transfer_paths, unfitted_domain

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from stokes import stokes_data  # noqa: E402
from stokes_tpm import GEOMETRIES, background_mesh, solve  # noqa: E402


def setup(name, maxh, k):
    mesh = background_mesh(maxh)
    inner = unfitted_domain(mesh, GEOMETRIES[name])[0]
    p = transfer_paths(mesh, GEOMETRIES[name], inner, order=k + 4)
    return mesh, inner, p


def test_vector_transfer():
    """uhat.Transfer(p, gradient=L) of the trace of a quadratic field with its exact gradient is the field at xbar."""
    U = lambda X, Y: CF((X * X - X * Y + Y, 2 * X * Y - Y * Y + 1))
    mesh, inner, p = setup("disk", 0.2, 2)
    S = restrict(MatrixValued(L2(mesh, order=2)), inner)
    M = restrict(FacetFESpace(mesh, order=2) ** 2, inner)
    fes = S * M
    (L, uhat), (_, mu) = fes.TnT()
    dB = boundary_ds(mesh, inner, bonus_intorder=6)
    gfu = GridFunction(fes)
    gfu.components[0].Set(stokes_data(U(x, y), CF(0), 1.0)[0], definedonelements=inner)
    gfu.components[1].Set(U(x, y), dual=True, definedonelements=inner)
    a = BilinearForm(fes)
    a += uhat.Transfer(p, gradient=L) * mu * dB
    a.Assemble()
    b = LinearForm(fes)
    b += U(*p.target) * mu * dB
    b.Assemble()
    res = gfu.vec.CreateVector()
    res.data = a.mat * gfu.vec - b.vec
    assert b.vec.Norm() > 1 and res.Norm() < 1e-12


def errors(sol, exact, mesh, inner):
    dV = dx(definedonelements=inner, bonus_intorder=4)
    return [sqrt(Integrate(InnerProduct(a - b, a - b) * dV, mesh)) for a, b in zip(sol, exact)]


def test_polynomial_reproduced_exactly():
    """Divergence-free quadratic u, linear p with zero mean on the disk; the level alpha makes p exact."""
    nu, tau = 0.5, 3.0
    g = lambda X, Y: CF((X * X + Y, -2 * X * Y + X))
    p = x - y
    L, f = stokes_data(g(x, y), p, nu)
    mesh = background_mesh(0.2)
    (Lh, uh, ph, _, _), inner, alpha = solve(mesh, 2, f, g, GEOMETRIES["disk"], nu=nu, tau=tau)
    eL, eu, ep, ep_unshifted = errors((Lh, uh, ph, ph - alpha), (L, g(x, y), p, p), mesh, inner)
    assert max(eL, eu, ep) < 1e-9
    assert ep_unshifted > 1e-4


@pytest.mark.parametrize("k", [1, 2])
def test_convergence_rate(k):
    nu = 1.0
    g = lambda X, Y: CF((sin(pi * X) * sin(pi * Y), cos(pi * X) * cos(pi * Y)))
    p = sin(2 * pi * x) * sin(2 * pi * y)
    L, f = stokes_data(g(x, y), p, nu)
    errs = []
    for maxh in (0.1, 0.05):
        mesh = background_mesh(maxh)
        (Lh, uh, ph, _, _), inner, _ = solve(mesh, k, f, g, GEOMETRIES["disk"], nu=nu)
        errs.append(errors((Lh, uh, ph), (L, g(x, y), p), mesh, inner))
    for e0, e1 in zip(*errs):
        assert log2(e0 / e1) > k + 1 - 0.2


def test_alpha_is_the_mean_shift():
    """|Omega_h| alpha + int_gap E ptilde = 0, |Omega_h| = |D_h| + the gap area: alpha is minus
    the post-solve shift of ptilde."""
    g = lambda X, Y: CF((sin(pi * X) * sin(pi * Y), cos(pi * X) * cos(pi * Y)))
    L, f = stokes_data(g(x, y), sin(2 * pi * x) * sin(2 * pi * y), 1.0)
    mesh = background_mesh(0.1)
    (_, _, ph, _, _), inner, alpha = solve(mesh, 2, f, g, GEOMETRIES["flower"])
    p = transfer_paths(mesh, GEOMETRIES["flower"], inner, order=6)
    dB = boundary_ds(mesh, inner, bonus_intorder=20)
    area = Integrate(CF(1) * dx(definedonelements=inner), mesh) + Integrate(CF(1).RegionIntegral(p) * dB, mesh)
    assert abs(alpha) > 1e-4
    assert abs(area * alpha + Integrate((ph - alpha).RegionIntegral(p) * dB, mesh)) < 1e-12

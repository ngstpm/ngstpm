"""Bilinear assembly under the interface measure I.ds (src/ifcmeasure.cpp): the segment
group with the interface binding, own versus partner (D7, D15), the D16/D17 raises, and
the static condensation of the coupling (D16), on the weak form of API-PROPOSAL 5b."""

from pathlib import Path
import sys

import numpy as np
import pytest
from ngsolve import (
    CF, BilinearForm, FacetFESpace, GridFunction, L2, LinearForm, VectorL2, atan2, cos, div, dx,
    exp, grad, pi, sin, specialcf, sqrt, x, y,
)

from ngstpm import boundary_ds, interior_ds, pair, restrict, transfer_paths
from ngstpm._ngstpm import SideEval, _CondensationInstalled, _SetCondensationHookForTesting

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson import solve_linear_system  # noqa: E402
from poisson_interface import domains, errors  # noqa: E402

KAPPA = (1.0, 10.0)


def disk(X, Y):
    return sqrt(X * X + Y * Y) - 1


def flower(X, Y):
    return sqrt(X * X + Y * Y) - 1 - 0.12 * cos(5 * atan2(Y, X))


def patch(k):
    k1, k2 = KAPPA
    u1 = lambda X, Y: X**k - 2 * X * Y**(k - 1) + Y
    u2 = lambda X, Y: 0.5 * Y**k + 3 * X**(k - 1) * Y - X + 2
    dx1 = lambda X, Y: k * X**(k - 1) - 2 * Y**(k - 1)
    dy1 = lambda X, Y: -2 * (k - 1) * X * Y**(k - 2) + 1 if k > 1 else CF(1.0) + 0 * X
    dx2 = lambda X, Y: 3 * (k - 1) * X**(k - 2) * Y - 1 if k > 1 else CF(-1.0) + 0 * X
    dy2 = lambda X, Y: 0.5 * k * Y**(k - 1) + 3 * X**(k - 1)
    q1 = lambda X, Y: -k1 * CF((dx1(X, Y), dy1(X, Y)))
    q2 = lambda X, Y: -k2 * CF((dx2(X, Y), dy2(X, Y)))
    return u1, u2, q1, q2, q1(x, y)[0].Diff(x) + q1(x, y)[1].Diff(y), q2(x, y)[0].Diff(x) + q2(x, y)[1].Diff(y)


def smooth():
    k1, k2 = KAPPA
    u1 = lambda X, Y: sin(pi * X) * cos(pi * Y)
    u2 = lambda X, Y: exp(0.5 * X) * cos(0.5 * Y) + X * Y
    q1 = lambda X, Y: -k1 * CF((pi * cos(pi * X) * cos(pi * Y), -pi * sin(pi * X) * sin(pi * Y)))
    q2 = lambda X, Y: -k2 * CF((0.5 * exp(0.5 * X) * cos(0.5 * Y) + Y, -0.5 * exp(0.5 * X) * sin(0.5 * Y) + X))
    return u1, u2, q1, q2, k1 * 2 * pi**2 * u1(x, y), CF(0.0)


def solve_5b(family, phi, maxh, k, sol, condense, bonus=6, tau=1.0):
    """The interface problem of examples/poisson_interface.py written as in API-PROPOSAL 5b."""
    u1e, u2e, q1e, q2e, f1, f2 = sol
    kappa1, kappa2 = KAPPA
    gD = lambda X, Y: u1e(X, Y) - u2e(X, Y)
    gN = lambda X, Y: q1e(X, Y) - q2e(X, Y)
    mesh, D1, D2 = domains(family, maxh, phi)
    p1, p2 = transfer_paths(mesh, phi, D1), transfer_paths(mesh, phi, D2)
    I = pair(p1, p2)
    dI = [I.ds(side=i, bonus_intorder=bonus) for i in (1, 2)]
    V1, V2 = (restrict(VectorL2(mesh, order=k), D) for D in (D1, D2))
    W1, W2 = (restrict(L2(mesh, order=k), D) for D in (D1, D2))
    M1, M2 = (restrict(FacetFESpace(mesh, order=k, dirichlet="outer"), D) for D in (D1, D2))
    fes = V1 * W1 * M1 * V2 * W2 * M2
    (q1, u1, uhat1, q2, u2, uhat2), (r1, v1, mu1, r2, v2, mu2) = fes.TnT()
    lhs, rhs = BilinearForm(fes, condense=condense), LinearForm(fes)
    n = specialcf.normal(2)
    for q, u, uhat, r, v, mu, kappa, f, D in ((q1, u1, uhat1, r1, v1, mu1, kappa1, f1, D1),
                                              (q2, u2, uhat2, r2, v2, mu2, kappa2, f2, D2)):
        dV, dS = dx(definedonelements=D), dx(element_boundary=True, definedonelements=D)
        qhat_n = q * n + tau * (u - uhat)
        lhs += 1 / kappa * q * r * dV - u * div(r) * dV + uhat * (r * n) * dS
        lhs += -q * grad(v) * dV + qhat_n * v * dS
        rhs += f * v * dV
        lhs += qhat_n * mu * interior_ds(mesh, D)
    xbar, nubar = p1.target, p1.target_normal
    qn1 = q1.Extrapolate(xbar) * nubar + tau * (u1 - uhat1)
    qn2 = q2.Extrapolate(xbar) * nubar - tau * (u2.Extrapolate() - uhat2.Extrapolate())
    lhs += (qn1 - qn2) * mu1 * dI[0]
    rhs += (gN(*xbar) * nubar) * mu1 * dI[0]
    xbar = p2.target
    ub1 = uhat1.Transfer(p1, gradient=-q1 / kappa1)
    ub2 = uhat2.Transfer(p2, gradient=-q2 / kappa2)
    lhs += (ub1 - ub2) * mu2 * dI[1]
    rhs += gD(*xbar) * mu2 * dI[1]
    gfu = GridFunction(fes)
    for c in (2, 5):
        gfu.components[c].Set(u2e(x, y), definedon=mesh.Boundaries("outer"))
    solve_linear_system(lhs, rhs, gfu)
    eu, eq = errors(mesh, gfu.components, D1, D2, u1e, u2e, q1e, q2e)
    return gfu.vec.FV().NumPy().copy(), eu, eq, lhs


@pytest.mark.parametrize("family,phi,k", [("fitted", disk, 1), ("snapped", flower, 2),
                                          ("unfitted", disk, 3), ("unfitted", flower, 2)])
def test_patch(family, phi, k):
    """Piecewise polynomials of degree k are reproduced to roundoff (condensed).
    Measured at h = 0.3: |u - uh| <= 9e-13, |q - qh| <= 2e-11 on all of fitted/snapped/
    unfitted x disk/flower x k = 1, 2, 3."""
    _, eu, eq, _ = solve_5b(family, phi, 0.3, k, patch(k), condense=True)
    assert eu < 1e-11 and eq < 1e-9


@pytest.mark.parametrize("family", ["unfitted", "snapped"])
def test_condensed_equals_uncondensed(family):
    """All element-local dofs condensed: the coupling is condensed (C_g + C_l H, rhs -C_l S f), and the solve
    agrees with condense=False to roundoff (measured 1.5e-13 unfitted, 3.4e-14 snapped)."""
    U0, eu0, _, _ = solve_5b(family, disk, 0.2, 2, smooth(), condense=False)
    U1, eu1, _, lhs = solve_5b(family, disk, 0.2, 2, smooth(), condense=True)
    assert _CondensationInstalled(lhs)
    assert abs(U1 - U0).max() < 1e-10 * abs(U0).max()
    assert abs(eu1 - eu0) < 1e-9 * eu0


def small(family="unfitted", phi=disk, k=2):
    mesh, D1, D2 = domains(family, 0.3, phi)
    p1, p2 = transfer_paths(mesh, phi, D1), transfer_paths(mesh, phi, D2)
    return mesh, D1, D2, p1, p2, pair(p1, p2)


def test_hook_missing_fails_loudly():
    """If the right-hand-side correction is not in harmonic_extension_trans, Assemble raises
    instead of letting a condensed solve be silently wrong."""
    mesh, D1, D2, p1, p2, I = small()
    fes = restrict(L2(mesh, order=1), D2) * restrict(FacetFESpace(mesh, order=1), D1)
    (u2, uhat1), (_, mu1) = fes.TnT()
    a = BilinearForm(fes, condense=True)
    a += u2 * fes.TestFunction()[0] * dx(definedonelements=D2) + uhat1 * mu1 * dx(element_boundary=True)
    a += u2.Extrapolate() * mu1 * I.ds(side=1)
    _SetCondensationHookForTesting(False)
    try:
        with pytest.raises(Exception, match="not in harmonic_extension_trans"):
            a.Assemble()
    finally:
        _SetCondensationHookForTesting(True)
    a.Assemble()
    assert _CondensationInstalled(a)


def test_local_test_function_raises():
    """D16: interface integrands are tested with facet test functions only."""
    mesh, D1, D2, p1, p2, I = small()
    fes = restrict(L2(mesh, order=1), D1) * restrict(L2(mesh, order=1), D2)
    (u1, u2), (v1, v2) = fes.TnT()
    a = BilinearForm(fes)
    with pytest.raises(Exception, match="D16"):
        a += u2.Extrapolate() * v1 * I.ds(side=1)


def test_partner_outside_interface_measure_raises():
    """D17: a partner evaluation under boundary_ds or dx (no binding) raises."""
    mesh, D1, D2, p1, p2, I = small()
    fes = restrict(L2(mesh, order=1), D2) * restrict(FacetFESpace(mesh, order=1), D1)
    (u2, _), (_, mu1) = fes.TnT()
    for measure in (boundary_ds(mesh, D1), dx(element_boundary=True, definedonelements=D1)):
        a = BilinearForm(fes)
        a += u2.Extrapolate() * mu1 * measure
        with pytest.raises(Exception, match="interface measure"):
            a.Assemble()


def apply(a, gf):
    v = a.mat.CreateColVector()
    v.data = a.mat * gf.vec
    return v.FV().NumPy()


def test_partner_values_by_hand():
    """Under the binding the partner evaluations are those of the piece rule: with mu = 1,
    mu^T A u = sum_points w p(x') for u2.Extrapolate() and SideEval(g2), and
    sum w p(xbar) for u2.Extrapolate(p1.target) (p a polynomial of the space)."""
    k, bonus = 2, 3
    mesh, D1, D2, p1, p2, I = small(k=k)
    poly = lambda X, Y: X * X - 2 * X * Y + 3 * Y + 1
    W2 = restrict(L2(mesh, order=k), D2)
    M1 = restrict(FacetFESpace(mesh, order=k), D1)
    fes = W2 * M1
    (u2, _), (_, mu1) = fes.TnT()
    gf = GridFunction(fes)
    gf.components[0].Set(poly(x, y))
    one = GridFunction(fes)
    one.components[1].Set(1.0, dual=True)
    g2 = GridFunction(L2(mesh, order=k))
    g2.Set(poly(x, y))
    P = I.points(1, 2 * k + bonus)           # the group's rule order 2k + bonus
    at_xp = sum(w * poly(*xp) for w, xp in zip(P.weight, P.xp))
    at_xbar = sum(w * poly(*xb) for w, xb in zip(P.weight, P.xbar))
    assert abs(at_xp) > 0.1 and abs(at_xbar) > 0.1
    gf_one = GridFunction(fes)
    gf_one.components[1].Set(1.0, dual=True)
    for integrand, expect, uvec in ((u2.Extrapolate() * mu1, at_xp, gf),
                                    (u2.Extrapolate(p1.target) * mu1, at_xbar, gf),
                                    (SideEval(g2, None, mesh, D2) * fes.TrialFunction()[1] * mu1, at_xp, gf_one)):
        a = BilinearForm(fes)
        a += integrand * I.ds(side=1, bonus_intorder=bonus)
        a.Assemble()
        val = np.dot(one.vec.FV().NumPy(), apply(a, uvec))
        assert val == pytest.approx(expect, rel=1e-12)


def test_side_override():
    """D15: one space on both sides is ambiguous under I.ds; side=1/2 or side=TransferPath decides."""
    k = 1
    mesh, D1, D2, p1, p2, I = small(k=k)
    fes = L2(mesh, order=k) * FacetFESpace(mesh, order=k)
    (w, _), (_, mu) = fes.TnT()
    dI = I.ds(side=1)
    a = BilinearForm(fes)
    a += w.Extrapolate() * mu * dI
    with pytest.raises(Exception, match="ambiguous"):
        a.Assemble()
    mats = {}
    for name, side in (("1", 1), ("2", 2), ("p1", p1), ("p2", p2)):
        a = BilinearForm(fes)
        a += w.Extrapolate(side=side) * mu * dI
        a.Assemble()
        mats[name] = a.mat.ToDense().NumPy()
    assert abs(mats["1"] - mats["p1"]).max() == 0 and abs(mats["2"] - mats["p2"]).max() == 0
    assert abs(mats["1"] - mats["2"]).max() > 0.1 * abs(mats["1"]).max()


def test_mesh_identity():
    """I.ds(side=i) in a form whose space is on another mesh (even a copy) raises."""
    mesh, D1, D2, p1, p2, I = small()
    other, _, _ = domains("unfitted", 0.3, disk)
    fes = FacetFESpace(other, order=1)
    u, mu = fes.TnT()
    a = BilinearForm(fes)
    with pytest.raises(Exception, match="mesh of that side"):
        a += u * mu * I.ds(side=1)
    f = LinearForm(fes)
    with pytest.raises(Exception, match="mesh of that side"):
        f += mu * I.ds(side=1)


def test_fitted_shared_facet_is_partner():
    """(d) On a fitted mesh F is a facet of K and of K': M2 has dofs on K (those of F) and on K',
    so a rule 'own = the space has dofs on K' would call uhat2 ambiguous (D15) and raise. Own is
    'all its dofs on K' (dofs of the Gamma_h facets alone do not make K own): uhat2.Extrapolate()
    is the partner's, and its values are those at x' (checked; the configuration is asserted)."""
    k = 2
    mesh, D1, D2, p1, p2, I = small("fitted", disk, k)
    M1, M2 = (restrict(FacetFESpace(mesh, order=k), D) for D in (D1, D2))
    some = [el for el in mesh.Elements() if D1[el.nr] and any(d >= 0 for d in M2.GetDofNrs(el))]
    assert some and all(not all(d >= 0 for d in M2.GetDofNrs(el)) for el in some)
    fes = M1 * M2
    (_, uhat2), (mu1, _) = fes.TnT()
    poly = lambda X, Y: X * X - 2 * X * Y + 3 * Y + 1
    gf = GridFunction(fes); gf.components[1].Set(poly(x, y), dual=True)
    one = GridFunction(fes); one.components[0].Set(1.0, dual=True)
    a = BilinearForm(fes)
    a += uhat2.Extrapolate() * mu1 * I.ds(side=1)
    a.Assemble()
    P = I.points(1, 2 * k)
    assert np.all(P.pel >= 0)
    val = np.dot(one.vec.FV().NumPy(), apply(a, gf))
    assert val == pytest.approx(sum(w * poly(*xp) for w, xp in zip(P.weight, P.xp)), rel=1e-12)

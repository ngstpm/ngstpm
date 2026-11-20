"""The interface problem of examples/poisson_interface.py: the measure I.ds,
partner evaluation, the patch test, rates and reference errors.
(Pairing geometry: tests/test_pairing.py, tests/test_interface_diagnostics.py.)"""

import sys
from math import log2
from pathlib import Path

import numpy as np
import pytest
from ngsolve import (
    CF, BilinearForm, FacetFESpace, GridFunction, L2,
    atan2, cos, exp, pi, sin, sqrt, x, y,
)

from ngstpm import boundary_ds, pair, restrict, transfer_paths

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson_interface import domains, errors, solve  # noqa: E402


def disk(X, Y):
    return sqrt(X * X + Y * Y) - 1


def flower(X, Y):
    return sqrt(X * X + Y * Y) - 1 - 0.12 * cos(5 * atan2(Y, X))


def unfitted_interface(maxh=0.3, phi=disk, kind=None):
    mesh, D1, D2 = domains("unfitted", maxh, phi)
    p1, p2 = transfer_paths(mesh, phi, D1, kind=kind), transfer_paths(mesh, phi, D2, kind=kind)
    return mesh, D1, D2, p1, p2, pair(p1, p2)


def test_quasi_normal_paths_kink():
    """Quasi-normal paths are not normal to Gamma at xbar: on the flower the two
    paths x -> xbar <- x' meet with a kink (|sin| > 1e-2 somewhere)."""
    mesh, D1, D2, p1, p2, I = unfitted_interface(phi=flower)
    kink = 0.0
    for side in (1, 2):
        P = I.points(side, 6)
        a, b = P.xbar - P.x, P.xbar - P.xp
        kink = max(kink, np.max(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
                                / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)))
    assert kink > 1e-2


def test_ds_without_partner_is_boundary_ds():
    """Without terms of the other side, I.ds(side=i) is boundary_ds(mesh, D_i)
    for polynomial integrands, which both rules (per segment piece, per facet)
    integrate exactly."""
    mesh, D1, D2, p1, p2, I = unfitted_interface()
    for i, D in ((1, D1), (2, D2)):
        fes = restrict(L2(mesh, order=2), D) * restrict(FacetFESpace(mesh, order=2), D)
        (u, uhat), (v, mu) = fes.TnT()
        form = (u.Extrapolate(CF((x + 0.1, y - 0.05))) + 2 * uhat) * mu
        a = BilinearForm(fes); a += form * I.ds(side=i, bonus_intorder=3); a.Assemble()
        b = BilinearForm(fes); b += form * boundary_ds(mesh, D, bonus_intorder=3); b.Assemble()
        A, B = a.mat.ToDense().NumPy(), b.mat.ToDense().NumPy()
        assert abs(B).max() > 0.01 and abs(A - B).max() < 1e-13 * abs(B).max()


@pytest.mark.parametrize("k", [1, 2, 3])
def test_partner_extrapolates_polynomials(k):
    """Under I.ds(side=1) a trial function of side 2 is the polynomial of the
    partner element K': at xbar with Extrapolate(xbar), at x' with Extrapolate().
    With mu piecewise constant on Gamma_h^1, row F of the matrix applied to p is
    the sum over the points of F of weight * (p(xbar) + p(x')), the rule being
    that of I.points(1, 2k + bonus)."""
    mesh, D1, D2, p1, p2, I = unfitted_interface()
    b = 4
    fes = restrict(L2(mesh, order=k), D2) * restrict(FacetFESpace(mesh, order=k), D2) \
        * restrict(FacetFESpace(mesh, order=0), D1)
    (u, uhat, _), (_, _, mu) = fes.TnT()
    p = lambda X, Y: X**k - 2 * X * Y**(k - 1) + 1
    a = BilinearForm(fes)
    a += (u.Extrapolate(p1.target) + uhat.Extrapolate()) * mu * I.ds(side=1, bonus_intorder=b)
    a.Assemble()
    gf = GridFunction(fes)
    gf.components[0].Set(p(x, y))
    gf.components[1].Set(p(x, y), dual=True)
    Ap = (a.mat * gf.vec).Evaluate().FV().NumPy()
    P = I.points(1, 2 * k + b)
    pn = lambda Z: Z[:, 0]**k - 2 * Z[:, 0] * Z[:, 1]**(k - 1) + 1
    vals = P.weight * (pn(P.xbar) + pn(P.xp))
    rows = {}
    for f, v in zip(P.facet, vals):
        rows[int(f)] = rows.get(int(f), 0.0) + v
    ref = np.zeros_like(Ap)
    for f, v in rows.items():
        ref[fes.GetDofNrs(mesh.facets[f])[-1]] = v              # the order-0 dof of mu on F
    assert np.abs(ref).max() > 0.01 and np.abs(Ap - ref).max() < 1e-10 * np.abs(ref).max()


KAPPA = (1.0, 10.0)
KAPPA_CF = tuple((lambda X, Y, c=c: CF(c)) for c in KAPPA)


@pytest.mark.parametrize("family,phi", [("fitted", disk), ("snapped", disk), ("unfitted", disk),
                                        ("snapped", flower), ("unfitted", flower)])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("kind", ["quasi-normal", "facet-normal"])
def test_piecewise_polynomials_reproduced(family, phi, k, kind):
    """Patch test: exact for solutions in the discrete space, whatever the gap."""
    if kind == "facet-normal" and family == "unfitted":
        pytest.skip("facet normals of the staircase Gamma_h can be nearly tangent to Gamma: "
                    "paths up to 16.8 long at h = 0.3 (disk), roundoff up to 1e-8")
    k1, k2 = KAPPA
    u1 = lambda X, Y: X**k - 2 * X * Y**(k - 1) + Y
    u2 = lambda X, Y: 0.5 * Y**k + 3 * X**(k - 1) * Y - X + 2
    dx1 = lambda X, Y: k * X**(k - 1) - 2 * Y**(k - 1)
    dy1 = lambda X, Y: -2 * (k - 1) * X * Y**(k - 2) + 1 if k > 1 else CF(1.0) + 0 * X
    dx2 = lambda X, Y: 3 * (k - 1) * X**(k - 2) * Y - 1 if k > 1 else CF(-1.0) + 0 * X
    dy2 = lambda X, Y: 0.5 * k * Y**(k - 1) + 3 * X**(k - 1)
    q1 = lambda X, Y: -k1 * CF((dx1(X, Y), dy1(X, Y)))
    q2 = lambda X, Y: -k2 * CF((dx2(X, Y), dy2(X, Y)))
    f1 = q1(x, y)[0].Diff(x) + q1(x, y)[1].Diff(y)
    f2 = q2(x, y)[0].Diff(x) + q2(x, y)[1].Diff(y)
    mesh, D1, D2 = domains(family, 0.3, phi)
    sol = solve(mesh, k, phi, D1, D2, *KAPPA_CF, f1, f2,
                lambda X, Y: u1(X, Y) - u2(X, Y), lambda X, Y: q1(X, Y) - q2(X, Y), u2(x, y), kind=kind)
    eu, eq = errors(mesh, sol, D1, D2, u1, u2, q1, q2)
    assert eu < 1e-10 and eq < 1e-9


def test_space_varying_kappa_reproduced():
    """Patch test with kappa_i varying in space (kappa along the path, D8):
    u_i linear, q_i = -kappa_i grad u_i, kappa_i linear, so q_i is in the
    discrete space for k = 2 and the TPM integral is exact."""
    k = 2
    kappa1 = lambda X, Y: 1 + 0.5 * X
    kappa2 = lambda X, Y: 10 + Y
    u1 = lambda X, Y: 2 * X - Y + 1
    u2 = lambda X, Y: X + 3 * Y - 2
    q1 = lambda X, Y: -kappa1(X, Y) * CF((2, -1))
    q2 = lambda X, Y: -kappa2(X, Y) * CF((1, 3))
    f1 = q1(x, y)[0].Diff(x) + q1(x, y)[1].Diff(y)
    f2 = q2(x, y)[0].Diff(x) + q2(x, y)[1].Diff(y)
    mesh, D1, D2 = domains("unfitted", 0.3, flower)
    sol = solve(mesh, k, flower, D1, D2, kappa1, kappa2, f1, f2,
                lambda X, Y: u1(X, Y) - u2(X, Y), lambda X, Y: q1(X, Y) - q2(X, Y), u2(x, y))
    eu, eq = errors(mesh, sol, D1, D2, u1, u2, q1, q2)
    assert eu < 1e-10 and eq < 1e-9


def smooth_solution():
    k1, k2 = KAPPA
    u1 = lambda X, Y: sin(pi * X) * cos(pi * Y)
    u2 = lambda X, Y: exp(0.5 * X) * cos(0.5 * Y) + X * Y
    q1 = lambda X, Y: -k1 * CF((pi * cos(pi * X) * cos(pi * Y), -pi * sin(pi * X) * sin(pi * Y)))
    q2 = lambda X, Y: -k2 * CF((0.5 * exp(0.5 * X) * cos(0.5 * Y) + Y,
                                -0.5 * exp(0.5 * X) * sin(0.5 * Y) + X))
    return u1, u2, q1, q2, k1 * 2 * pi**2 * u1(x, y), CF(0.0)


def smooth_errors(family, k, maxh):
    u1, u2, q1, q2, f1, f2 = smooth_solution()
    mesh, D1, D2 = domains(family, maxh, disk)
    sol = solve(mesh, k, disk, D1, D2, *KAPPA_CF, f1, f2,
                lambda X, Y: u1(X, Y) - u2(X, Y), lambda X, Y: q1(X, Y) - q2(X, Y), u2(x, y))
    return errors(mesh, sol, D1, D2, u1, u2, q1, q2)


@pytest.mark.parametrize("family", ["fitted", "snapped"])
@pytest.mark.parametrize("k", [1, 2])
def test_convergence_rate(family, k):
    """Order k+1 for u when the gap is O(h^2)."""
    e = [smooth_errors(family, k, maxh)[0] for maxh in (0.2, 0.1)]
    assert log2(e[0] / e[1]) > k + 0.8


# ||u - u_h|| at maxh = 0.1, k = 2, bonus 6 (independent of the bonus to 1e-9);
# the unfitted value guards the O(h)-gap behaviour, which has no regular rate.
REFERENCE = {"fitted": 4.102e-04,    # unchanged
             "snapped": 6.337e-04,   # old 6.374e-04: one Gauss rule per facet across partner breakpoints; the old code at bonus 20-60 settles on the new value (k = 1)
             "unfitted": 9.685e-04}  # old 1.140e-03: same cause; the old code wanders by about 4% with the bonus (k = 1, bonus 6-60) and approaches the new value


@pytest.mark.parametrize("family", list(REFERENCE))
def test_reference_errors(family):
    assert smooth_errors(family, 2, 0.1)[0] == pytest.approx(REFERENCE[family], rel=1e-3)

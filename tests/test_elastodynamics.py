import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, GridFunction, InnerProduct, Integrate, VectorL2, dx, sqrt, x, y
from ngsolve.meshes import MakeStructured2DMesh

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import elastodynamics as ed  # noqa: E402
from elastodynamics import t  # noqa: E402
from elasticity_tpm import EXAMPLES, domain  # noqa: E402

TPM = "Example 2 (unfitted disk)"
POLY = {1: CF((1 + 2 * x - y, -3 + x + 4 * y)),
        2: CF((x * x - 2 * x * y + y, 3 * y * y + x * y - x))}
MAT = ed.material(rho=2.0, b0=1.5)


def setting(case, h):
    if case == "fitted":
        return MakeStructured2DMesh(quads=False, nx=round(1 / h), ny=round(1 / h)), {}
    phi = EXAMPLES[case]
    mesh, D, dirichlet, _ = domain(case, h, phi)
    return mesh, dict(phi=phi, D=D, dirichlet=dirichlet)


def poly(k):
    """Degree k in space, quadratic in t: the elliptic projections and Newmark are exact."""
    return POLY[k] * (1 + t + t * t / 2) + CF((y, -x)) * t * t


@pytest.mark.parametrize("case, h", [("fitted", 1 / 3), (TPM, 0.3)])
@pytest.mark.parametrize("k", [1, 2])
def test_exact_polynomial(case, h, k):
    """Every column at every step to roundoff (sigma, eta, gamma, etahat, eta*, v)."""
    mesh, kw = setting(case, h)
    worst = []
    ed.run(mesh, k, poly(k), MAT, 0.1, 4, monitor=lambda r: worst.append(max(r.errors(poly(k)))), **kw)
    assert len(worst) == 4 and max(worst) < 1e-9


@pytest.mark.parametrize("case, h", [("fitted", 0.25), (TPM, 0.3)])
@pytest.mark.parametrize("k", [1, 2])
def test_consistent_acceleration(case, h, k):
    """MATLAB test_consistent_acceleration_solid (1): with the elliptic projections the eta block
    of A is Pi eta''(0) to roundoff, for smooth (non-polynomial) data."""
    mesh, kw = setting(case, h)
    r = ed.Elastodynamics(mesh, k, MAT, ed.manufactured(ed.SMOOTH, MAT), ed.SMOOTH, 0.1, **kw)
    r.initial_state(ed.SMOOTH, ed.SMOOTH.Diff(t))
    gf = GridFunction(r.fes)
    gf.vec.data = r.A
    P = GridFunction(VectorL2(mesh, order=k))
    P.Set(ed.SMOOTH.Diff(t).Diff(t), definedonelements=r.D, bonus_intorder=4)
    dV = dx(definedonelements=r.D, bonus_intorder=4)
    e = gf.components[3] - P
    norm = sqrt(Integrate(InnerProduct(P, P) * dV, mesh))
    assert sqrt(Integrate(InnerProduct(e, e) * dV, mesh)) < 1e-10 * norm


def eocs(case, k, hs, T=0.5):
    rows = []
    for h in hs:
        mesh, kw = setting(case, h)
        n = round(T / h)
        rows.append(ed.run(mesh, k, ed.SMOOTH, MAT, T / n, n, **kw).errors(ed.SMOOTH))
    return [log2(a / b) for a, b in zip(*rows)]


# k = 1, dt = h, T = 0.5: space-time order 2. Observed EOC (sigma, eta, gamma, v): fitted
# N 8 -> 16: 2.04, 2.15, 2.36, 1.57; unfitted disk h 0.1 -> 0.05: 2.01, 2.15, 2.24, 1.53. v is
# preasymptotic (fitted unit_square, h 0.2 -> 0.025: 1.66, 2.07, 2.64).
@pytest.mark.parametrize("case", ["fitted", TPM])
def test_rates(case):
    r = eocs(case, 1, (1 / 8, 1 / 16) if case == "fitted" else (0.1, 0.05))
    assert min(r[0], r[1], r[2]) > 1.8 and r[8] > 1.4, r

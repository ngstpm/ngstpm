import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, GridFunction, InnerProduct, Integrate, VectorL2, dx, pi, sin, sqrt, x, y
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


# ---- VelocityStress (Crank-Nicolson) -----------------------------------------------------------

def vpoly(k):
    """Degree k in space, quadratic in t (v, sigma' linear): Crank-Nicolson is exact."""
    return POLY[k] * (1 + t + t * t / 2) + CF((y, -x)) * t * t


@pytest.mark.parametrize("case, h", [("fitted", 1 / 3), (TPM, 0.3)])
@pytest.mark.parametrize("k", [1, 2])
def test_velocity_exact_polynomial(case, h, k):
    """Every column (sigma, v, omega, eta, vhat) at every step to roundoff."""
    mesh, kw = setting(case, h)
    worst = []
    ed.run_velocity(mesh, k, vpoly(k), MAT, 0.1, 4, monitor=lambda r: worst.append(max(r.errors(vpoly(k)))), **kw)
    assert len(worst) == 4 and max(worst) < 1e-9


def test_velocity_energy_decays():
    """f = 0, g = 0, b0 > 0: the energy (A sigma, sigma) + rho |v|^2 + b0 |eta|^2 never grows
    (Crank-Nicolson conserves it; tau_S (v - vhat) dissipates)."""
    mesh, _ = setting("fitted", 0.125)
    eta0 = CF((sin(pi * x) * sin(pi * y), x * y * (1 - x) * (1 - y))) * (1 + t)
    r = ed.VelocityStress(mesh, 1, MAT, CF((0, 0)), CF((0, 0)) * t, 0.05)
    r.initial_state(eta0)
    E = [r.energy()]
    r.march(10, lambda r: E.append(r.energy()))
    assert E[0] > 0.1 and all(b <= a * (1 + 1e-12) for a, b in zip(E, E[1:])) and E[-1] < E[0]


def veocs(case, k, hs, T=0.5):
    rows = []
    for h in hs:
        mesh, kw = setting(case, h)
        n = round(T / h)
        rows.append(ed.run_velocity(mesh, k, ed.SMOOTH, MAT, T / n, n, **kw).errors(ed.SMOOTH))
    return [log2(a / b) for a, b in zip(*rows)]


# k = 1, dt = h, T = 0.5: space-time order 2 for sigma, v, eta, vhat. omega is order k only (not
# asserted; fitted unit_square, h 0.2 -> 0.025: 1.14, 0.86, 1.13), also with dt << h and for
# tau from 0.01 to 10; the Newmark gamma has k + 1.
@pytest.mark.parametrize("case", ["fitted", TPM])
def test_velocity_rates(case):
    r = veocs(case, 1, (1 / 8, 1 / 16) if case == "fitted" else (0.1, 0.05))
    assert min(r[0], r[1], r[3]) > 1.8, r


# ---- unfitted meshes: the stable settings ---------------------------------------------------------

STABLE = {"displacement": (ed.Elastodynamics, ed.run, dict(transfer="grad", damping=1.0)),
          "velocity": (ed.VelocityStress, ed.run_velocity, dict(transfer="grad"))}


def unfitted(name, h, prune=True):
    phi = EXAMPLES[name]
    mesh, D, dirichlet, _ = domain(name, h, phi, prune)
    return mesh, dict(phi=phi, D=D, dirichlet=dirichlet)


@pytest.mark.parametrize("form", STABLE)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_stable_settings_exact(form, k):
    """Polynomial solutions stay exact with transfer="grad" (and damping) on the unfitted disk."""
    from ngsolve import CF
    P3 = CF((x ** 3 - 2 * x * y * y + y, 3 * y ** 3 + x * x * y - x))
    eta = {1: POLY[1], 2: POLY[2], 3: P3}[k] * (1 + t + t * t / 2) + CF((y, -x)) * t * t
    mesh, kw = unfitted(TPM, 0.3, prune=False)
    _, runner, extra = STABLE[form]
    worst = []
    runner(mesh, k, eta, MAT, 0.1, 4, monitor=lambda r: worst.append(max(r.errors(eta))), **kw, **extra)
    assert max(worst) < 1e-9


def growth(cls, mesh, k, kw, dt=0.01, T=1.0):
    """Growth rate per unit time of the zero-data march from a random displacement (velocity)."""
    import math
    import numpy as np
    r = cls(mesh, k, ed.material(), CF((0, 0)), CF((0, 0)) * t, dt, **kw)
    gf = GridFunction(r.fes)
    gf.vec[:] = 0
    v = gf.components[3].vec.FV().NumPy()
    v[:] = np.random.default_rng(0).standard_normal(len(v))
    r.U.data = gf.vec
    norms = []
    for _ in range(round(T / dt)):
        r.step()
        norms.append(np.linalg.norm(r.U.FV().NumPy()))
    n = len(norms)
    return math.log(norms[-1] / norms[n // 2]) / ((n - 1 - n // 2) * dt)


# Kidney h = 0.1, k = 2. Observed rates: MATLAB-like settings (no prune, stress transfer, no
# damping) 270 (displacement), 80 (velocity); stable settings -0.46, -0.00. Smooth data to T = 10
# with the stable settings: sigma error bounded (disk and kidney, h 0.1, 0.05, k = 1..3).
@pytest.mark.parametrize("form", STABLE)
def test_stable_settings_no_growth(form):
    cls, _, extra = STABLE[form]
    mesh, kw = unfitted("Example 3 (kidney)", 0.1)
    assert growth(cls, mesh, 2, dict(kw, **extra)) < 0.05
    plain, _ = unfitted("Example 3 (kidney)", 0.1, prune=False)
    assert growth(cls, plain, 2, unfitted("Example 3 (kidney)", 0.1, prune=False)[1], dt=0.005, T=0.4) > 10


# Example 1 (Gamma_h at distance O(h^2)), dt = h, T = 0.5, defaults: EOC of sigma, eta (v) about 2
# (k = 1, 2, 3; h 0.1 -> 0.05: 2.11, 2.28 | 1.97, 1.99 | displacement form).
@pytest.mark.parametrize("form", STABLE)
def test_polygonal_disk_converges(form):
    _, runner, _ = STABLE[form]
    e = []
    for h in (0.1, 0.05):
        mesh, kw = unfitted("Example 1 (fitted disk)", h, prune=False)
        n = round(0.5 / h)
        e.append(runner(mesh, 1, ed.SMOOTH, MAT, 0.5 / n, n, **kw).errors(ed.SMOOTH))
    assert log2(e[0][0] / e[1][0]) > 1.7 and log2(e[0][1] / e[1][1]) > 1.7

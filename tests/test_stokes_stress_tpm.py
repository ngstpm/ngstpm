import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, InnerProduct, Integrate, cos, dx, pi, sin, sqrt, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import elasticity  # noqa: E402
import stokes_stress as st  # noqa: E402
import stokes_stress_tpm_neumann as neumann  # noqa: E402
from elasticity_tpm import EXAMPLES, at, domain, sigma_modulo_pressure, solve as solve_elastic  # noqa: E402
from poisson_tpm import background_mesh  # noqa: E402
from stokes_stress_tpm import solve, table_errors  # noqa: E402

POLY = {1: (CF((x + 2 * y, 3 * x - y)), CF(0.0)),
        2: (CF((x * x - 2 * x * y + y * y + 1, -2 * x * y + y * y + x)), x - 0.5 + 2 * (y - 0.5))}


def run(name, maxh, k, u=st.U, p=st.P, mu=1.0):
    phi = EXAMPLES[name]
    sigma, gamma, f = st.manufactured(u, p, mu)
    mesh, D, dirichlet, _ = domain(name, maxh, phi)
    sol, p_h, m_h, D = solve(mesh, k, mu, f, at(u), phi, D, dirichlet=dirichlet)
    e, c = table_errors(mesh, sol, p_h, u, p, sigma, gamma, D)
    return e, m_h


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("name, maxh", [("Example 1 (fitted disk)", 0.3),
                                        ("Example 2 (unfitted disk)", 0.3),
                                        ("Example 3 (kidney)", 0.15)])
def test_patch_dirichlet(name, maxh, k):
    """Divergence-free velocity of degree k, pressure of degree k - 1: sigma modulo c I, p
    shifted by c, u, gamma, traces, u* to roundoff (observed at most 3.4e-11, kidney k = 2);
    m_h = 0 (observed at most 1.1e-12): the transferred polynomial trace is exact."""
    e, m_h = run(name, maxh, k, *POLY[k])
    assert max(e[1:]) < 1e-10 and abs(m_h) < 1e-10


@pytest.mark.parametrize("k", [1, 2])
def test_patch_traction(k):
    """Raw errors (the traction fixes the pressure level); observed at most 3.8e-11."""
    u, p = POLY[k]
    sigma, gamma, f = st.manufactured(u, p, 1.0)
    mesh = background_mesh(0.3)
    sol, p_h, D = neumann.solve(mesh, k, 1.0, f, u, at(sigma), neumann.PHI)
    assert max(st.errors(mesh, sol, p_h, u, p, sigma, gamma, D)) < 1e-10


def test_multiplier_decreases():
    """m_h (flux defect of the transferred trace) tends to 0 on Example 2, k = 1
    (observed 7.2e-4, 4.0e-4 at maxh 0.2, 0.1)."""
    m = [abs(run("Example 2 (unfitted disk)", h, 1)[1]) for h in (0.2, 0.1)]
    assert m[1] < 0.7 * m[0]


def test_limit_of_elasticity():
    """Elastic TPM solver (nu = 0.499999 and 0.49999999, bounded data) against the Stokes
    TPM solver (u = curl psi, p = -div u_1, same mu) on Example 2, k = 1. The difference does
    not depend on lambda (the limit is reached): it is the term A(c I) of the transferred
    trace of the elastic solver, of the size of m_h times the path length, and it decreases
    with h: sigma modulo c I 1.27e-3 -> 1.02e-4, u 3.90e-4 -> 3.82e-5 at maxh 0.2 -> 0.1."""
    name = "Example 2 (unfitted disk)"
    phi = EXAMPLES[name]
    us = CF((sin(pi * x) * cos(pi * y), -cos(pi * x) * sin(pi * y)))
    diffs = {}
    for h in (0.2, 0.1):
        mesh, D, _, _ = domain(name, h, phi)
        dV = dx(definedonelements=D)
        for nu in (0.499999, 0.49999999):
            lam, mu = elasticity.lame(1.0, nu)
            ue = elasticity.bounded_data(lam)
            se, ge, fe = elasticity.manufactured(ue, lam, mu)
            sol_e, _ = solve_elastic(mesh, 1, lam, mu, fe, at(ue), phi, D)
            ss, gs, fs = st.manufactured(us, -2 * pi * cos(pi * x) * cos(pi * y), mu)
            sol_s, _, _, _ = solve(mesh, 1, mu, fs, at(us), phi, D)
            du = sqrt(Integrate(InnerProduct(sol_e[1] - sol_s[1], sol_e[1] - sol_s[1]) * dV, mesh))
            diffs[h, nu] = (sigma_modulo_pressure(mesh, sol_e[0], sol_s[0], D), du)
    lam = elasticity.lame(1.0, 0.499999)[0]
    for h in (0.2, 0.1):
        for a, b in zip(diffs[h, 0.499999], diffs[h, 0.49999999]):
            assert abs(a - b) < 1 / lam          # O(1/lam): observed 1.4e-6 to 2.6e-6, 1/lam = 6.0e-6
    for a, b in zip(diffs[0.2, 0.49999999], diffs[0.1, 0.49999999]):
        assert a / b > 8


# Observed EOC (sigma modulo c I, p shifted, u, gamma), mu = 1:
#   Example 1, maxh 0.2 -> 0.1:   k = 1: 1.98, 1.84, 2.06, 1.86;   k = 2: 3.17, 3.18, 3.13, 3.21
#   Example 2, maxh 0.1 -> 0.05:  k = 1: 2.03, 2.06, 1.93, 1.91;   k = 2: 3.93, 3.94, 4.42, 3.95
#   (Example 2, k = 2 is only 2.31, 2.38, 2.54, 2.07 on 0.2 -> 0.1.)
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("name, hs", [("Example 1 (fitted disk)", (0.2, 0.1)),
                                      ("Example 2 (unfitted disk)", (0.1, 0.05))])
def test_rates(name, hs, k):
    e1, e2 = (run(name, h, k)[0] for h in hs)
    for j in (1, 2, 3, 4):
        assert log2(e1[j] / e2[j]) > k + 0.7

import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, Integrate, div, dx, sqrt, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import oseen_stress as oseen  # noqa: E402
import stokes_stress as st  # noqa: E402
import stokes_stress_tpm as tpm  # noqa: E402
import stokes_stress_tpm_neumann as neumann  # noqa: E402
from divfree import corrected_beta, divergence_free  # noqa: E402
from elasticity_tpm import EXAMPLES, at, domain  # noqa: E402
from oseen_stress_tpm import SINK, solve  # noqa: E402
from poisson_tpm import GEOMETRIES, background_mesh  # noqa: E402

POLY = {1: (CF((x + 2 * y, 3 * x - y)), CF(0.0)),
        2: (CF((x * x - 2 * x * y + y * y + 1, -2 * x * y + y * y + x)), x - 0.5 + 2 * (y - 0.5))}


def dirichlet(name, maxh, k, u=st.U, p=st.P, mu=1.0, beta=oseen.BETA, stab="global"):
    """Errors of stokes_stress_tpm.table_errors (sigma, sigma - cI, p - c, u, gamma, trace, u*),
    m_h, S and the solution."""
    phi = GEOMETRIES["flower"] if name == "flower" else EXAMPLES[name]
    if name == "flower":
        mesh, D, bc = background_mesh(maxh), None, ".*"
    else:
        mesh, D, bc, _ = domain(name, maxh, phi)
    sigma, gamma, f = oseen.manufactured(u, p, mu, beta)
    sol, p_h, m_h, D, S = solve(mesh, k, mu, f, at(u), phi, beta, D, bc, stab)
    return tpm.table_errors(mesh, sol, p_h, u, p, sigma, gamma, D)[0], m_h, S, (mesh, sol, D)


def traction(maxh, k, u=st.U, p=st.P, mu=1.0, beta=SINK, stab="global"):
    """Raw errors of stokes_stress.errors (sigma, p, u, gamma, trace, u*, div u_h) and S."""
    mesh = background_mesh(maxh)
    sigma, gamma, f = oseen.manufactured(u, p, mu, beta)
    sol, p_h, _, D, S = solve(mesh, k, mu, f, u, neumann.PHI, beta, stab=stab, gN=at(sigma))
    return st.errors(mesh, sol, p_h, u, p, sigma, gamma, D), S


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("name", ["Example 1 (fitted disk)", "Example 2 (unfitted disk)", "flower"])
def test_patch_dirichlet(name, k):
    """Degree-k divergence-free velocity, degree k - 1 pressure, beta = (1 + y, x): all
    columns but raw sigma to roundoff (observed at most 4.9e-12, flower k = 2), m_h = 0
    (observed 5.4e-14); fails if the convective flux enters the transferred rows."""
    e, m_h, _, _ = dirichlet(name, 0.3, k, *POLY[k])
    assert max(e[1:]) < 1e-10 and abs(m_h) < 1e-10


@pytest.mark.parametrize("k", [1, 2])
def test_patch_traction(k):
    """The same with traction on the hole (beta = (1 + y, x), not an outflow: consistency
    only); observed at most 1.1e-11."""
    e, _ = traction(0.3, k, *POLY[k], beta=oseen.BETA)
    assert max(e) < 1e-10


@pytest.mark.parametrize("k", [1, 2])
def test_stokes_limit(k):
    """beta = 0 reproduces stokes_stress_tpm.solve and stokes_stress_tpm_neumann.solve dof for
    dof (observed difference exactly 0)."""
    name = "Example 2 (unfitted disk)"
    phi = EXAMPLES[name]
    sigma, gamma, f = st.manufactured(st.U, st.P, 1.0)
    mesh, D, bc, _ = domain(name, 0.3, phi)
    a = tpm.solve(mesh, k, 1.0, f, at(st.U), phi, D, dirichlet=bc)[0][1].gfu.vec.FV().NumPy()
    b = solve(mesh, k, 1.0, f, at(st.U), phi, CF((0, 0)), D, bc)[0][1].gfu.vec.FV().NumPy()
    assert abs(a - b).max() < 1e-12 * abs(a).max()
    mesh = background_mesh(0.3)
    a = neumann.solve(mesh, k, 1.0, f, st.U, at(sigma), neumann.PHI)[0][1].gfu.vec.FV().NumPy()
    b = solve(mesh, k, 1.0, f, st.U, neumann.PHI, CF((0, 0)), gN=at(sigma))[0][1].gfu.vec.FV().NumPy()
    assert abs(a - b).max() < 1e-12 * abs(a).max()


# Example 2, maxh 0.2, k = 1, mu = 0.1: S, and the errors (sigma - cI, u; m_h) and (traction,
# sink beta: sigma, u) pinned, so that a wrong S, sign or face set fails.
PINNED = {"global": (1.0309816216595464, (6.2221519348e-02, 3.6182766608e-02, 2.9319787916e-04),
                     0.9187469663833309, (7.8431188543e-02, 4.1289227795e-02)),
          "pointwise": (None, (5.9680912037e-02, 4.5095382275e-02, 9.3919753190e-04),
                        None, (7.3942044721e-02, 5.6113568514e-02))}


@pytest.mark.parametrize("stab", PINNED)
def test_pinned(stab):
    S_d, ref_d, S_t, ref_t = PINNED[stab]
    e, m_h, S, _ = dirichlet("Example 2 (unfitted disk)", 0.2, 1, mu=0.1, stab=stab)
    for a, b in zip((e[1], e[3], m_h), ref_d):
        assert abs(a / b - 1) < 1e-6
    et, St = traction(0.2, 1, mu=0.1, stab=stab)
    for a, b in zip((et[0], et[2]), ref_t):
        assert abs(a / b - 1) < 1e-6
    assert (S is None and S_d is None) or abs(S - S_d) < 1e-12
    assert (St is None and S_t is None) or abs(St - S_t) < 1e-12


# Observed EOC (sigma - cI, p - c, u, gamma), mu = 1 (agentB-tables/goal4-stage2-*.txt):
#   Example 1, maxh 0.2 -> 0.1:   k = 1: 1.99, 1.83, 2.06, 1.86;   k = 2: 3.16, 3.17, 3.13, 3.18
#   Example 2, maxh 0.1 -> 0.05:  k = 1: 2.03, 2.07, 1.97, 1.91;   k = 2: 3.50, 3.50, 4.07, 3.52
# (Example 2, k = 2, mu = 0.1 gives 3.56 then 2.59 for sigma - cI; the Stokes solver (beta = 0)
# at mu = 0.1 gives 2.51 then 3.22, the other way round (goal4-stage2-example2-mu0.1-k2-stokes.txt).
# Mesh-to-mesh scatter of the unstructured meshes: on maxh 0.16, 0.08, 0.04 the Oseen rates
# are 2.92 then 3.27, and the error at maxh 0.14 exceeds the one at 0.2 (Fable, stage 2 review).)
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("name, hs", [("Example 1 (fitted disk)", (0.2, 0.1)),
                                      ("Example 2 (unfitted disk)", (0.1, 0.05))])
def test_rates(name, hs, k):
    """A regression pin: one pair of meshes, threshold k + 0.7 on the observed EOC above; not a
    convergence-rate study (the tables are)."""
    e1, e2 = (dirichlet(name, h, k)[0] for h in hs)
    for j in (1, 2, 3, 4):
        assert log2(e1[j] / e2[j]) > k + 0.7


@pytest.mark.parametrize("k", [1, 2])
def test_rates_traction(k):
    """Sink beta, outflow traction on the hole, maxh 0.2 -> 0.1, mu = 1 (sigma, p, u, gamma):
    observed 2.19, 2.20, 2.05, 1.93 (k = 1), 4.26, 4.22, 4.64, 3.98 (k = 2); the next level
    gives 1.95 and 2.78 for sigma, as the Stokes solver (D21, goal-2 traction table)."""
    e1, e2 = (traction(h, k)[0] for h in (0.2, 0.1))
    for j in (0, 1, 2, 3):
        assert log2(e1[j] / e2[j]) > k + 0.7


def test_corrected_beta():
    """Example 2, maxh 0.2, k = 1: div u** = -m_h (about 7e-4), div beta at roundoff
    (observed 1e-14); beta = u** when m_h = 0."""
    e, m_h, _, (mesh, sol, D) = dirichlet("Example 2 (unfitted disk)", 0.2, 1)
    us = divergence_free(mesh, 1, 1.0, *sol[:4], D)
    dV = dx(definedonelements=D, bonus_intorder=2)
    norm = lambda b: sqrt(Integrate(div(b) ** 2 * dV, mesh))
    assert norm(us) > 1e-4 and norm(corrected_beta(mesh, us, m_h, D)) < 1e-11
    same = corrected_beta(mesh, us, 0.0, D)
    assert max(abs(a - b) for a, b in zip(same.vec, us.vec)) == 0

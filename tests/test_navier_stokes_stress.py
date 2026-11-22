import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, GridFunction, Integrate, Mesh, div, dx, specialcf, sqrt, x, y
from ngsolve.meshes import MakeStructured2DMesh

pytest.importorskip("scipy")                  # examples/superlu.py (the Picard solves)

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import navier_stokes_stress as ns  # noqa: E402
import oseen_stress as oseen  # noqa: E402
import oseen_stress_tpm as otpm  # noqa: E402
import stokes_stress as st  # noqa: E402
from elasticity_tpm import EXAMPLES, at, domain  # noqa: E402

POLY = {1: (CF((x + 2 * y, 3 * x - y)), CF(0.0)),
        2: (CF((x * x - 2 * x * y + y * y + 1, -2 * x * y + y * y + x)), x - 0.5 + 2 * (y - 0.5))}
EX2 = "Example 2 (unfitted disk)"
PAPER4_MAXH = 0.2
KIDNEY = "Example 3 (kidney)"


def fitted_step(mesh, k, mu, u, p, bc="dirichlet"):
    """step(beta) of the fitted problem with the Navier-Stokes data of (u, p)."""
    sigma, gamma, f = oseen.manufactured(u, p, mu, u)
    dirichlet, gN = (".*", None) if bc == "dirichlet" else ("left|bottom", sigma * specialcf.normal(2))
    return (lambda beta: oseen.solve(mesh, k, mu, f, u, beta, gN, dirichlet)), (sigma, gamma)


def unfitted_step(maxh, k, mu, u, p, name=EX2):
    """step(beta), mesh and D of Example 2 (or name) with the Navier-Stokes data of (u, p)."""
    phi = EXAMPLES[name]
    mesh, D, bc, _ = domain(name, maxh, phi)
    sigma, gamma, f = oseen.manufactured(u, p, mu, u)

    def step(beta):
        sol, p_h, m_h, _, S = otpm.solve(mesh, k, mu, f, at(u), phi, beta, D, bc)
        return sol, p_h, m_h, S
    return step, mesh, D, (sigma, gamma)


def paper4_step(maxh, k, mu, u, p):
    """step(beta), mesh and D of paper example 4 (disk of radius 0.75 in (-1, 1)^2) with the
    Navier-Stokes data of (u, p)."""
    mesh, D = ns.paper4_domain(maxh)
    sigma, gamma, f = oseen.manufactured(u, p, mu, u)

    def step(beta):
        sol, p_h, m_h, _, S = otpm.solve(mesh, k, mu, f, at(u), ns.PHI_PAPER4, beta, D, ".*")
        return sol, p_h, m_h, S
    return step, mesh, D, (sigma, gamma)


@pytest.mark.parametrize("bc", ["dirichlet", "mixed", "unfitted"])
def test_beta_handoff(bc):
    """Step 1 gets beta = 0 (Stokes); step n + 1 gets the u** of step n (divfree, called here
    independently), corrected by m_h on the unfitted Dirichlet problem; dof for dof."""
    import divfree
    if bc == "unfitted":
        step, mesh, D, _ = unfitted_step(0.3, 1, 0.1, st.U, st.P)
    else:
        mesh, D = MakeStructured2DMesh(quads=False, nx=4, ny=4), None
        step = fitted_step(mesh, 1, 0.1, ns.U_MIXED if bc == "mixed" else st.U, st.P, bc)[0]
    seen = []

    def spy(beta):
        out = step(beta)
        seen.append((beta, out))
        return out
    ns.picard(spy, mesh, 1, 0.1, D, correct=bc == "unfitted", maxit=3)
    assert len(seen) == 3 and isinstance(seen[0][0], CF) and not isinstance(seen[0][0], GridFunction)
    for (_, (sol, _, m_h, _)), (beta, _) in zip(seen, seen[1:]):
        us = divfree.divergence_free(mesh, 1, 0.1, *sol[:4], D)
        ref = divfree.corrected_beta(mesh, us, m_h, D) if bc == "unfitted" else us
        assert bc != "unfitted" or abs(m_h) > 1e-6          # the correction is not trivial
        a, b = beta.vec.FV().NumPy(), ref.vec.FV().NumPy()
        assert a.shape == b.shape and abs(a - b).max() == 0


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("bc", ["dirichlet", "mixed", "unfitted", "kidney", "paper4"])
def test_patch(k, bc):
    """Navier-Stokes patch: u of degree k divergence free, p of degree k - 1, beta = u. The
    exact solution is a fixed point of the Picard map (u** reproduces P_{k+1}, so beta = u,
    and the Oseen patch is exact; the convective integrals of the HDiv(k+1) beta are exact
    because the bubble components of degree k + 2 raise the element integration order, not
    because NGSolve reads the order of beta: agentB-tables/goal4-stage3-quadrature.txt);
    Picard from Stokes reaches it to the tolerance. Observed: 7 to 15 steps, u, gamma, trace, u*, u** at most 5.9e-13, raw sigma
    and p (fitted) at most 2.1e-13. Kidney (maxh 0.15, as test_stokes_stress_tpm.py): the
    increment stalls at about 5e-13 at k = 2 (cap reached with tol 1e-13), so tol 1e-11 and the
    bound 1e-10 of the Stokes kidney patch; observed 6 and 8 steps, at most 8.3e-12 (k = 2).
    Paper example 4 (disk of radius 0.75 in (-1, 1)^2, maxh 0.2, 74 elements in D_h): likewise
    the k = 2 increment stalls at about 4e-12 (cap reached with tol 1e-13), so the kidney's tol
    and bound; observed 7 steps at k = 1 and 2, at most 2.5e-12 (k = 2)."""
    u, p = POLY[k]
    if bc == "kidney":
        step, mesh, D, (sigma, gamma) = unfitted_step(0.15, k, 1.0, u, p, KIDNEY)
    elif bc == "paper4":
        step, mesh, D, (sigma, gamma) = paper4_step(PAPER4_MAXH, k, 1.0, u, p)
    elif bc == "unfitted":
        step, mesh, D, (sigma, gamma) = unfitted_step(0.3, k, 1.0, u, p)
    else:
        mesh, D = MakeStructured2DMesh(quads=False, nx=3, ny=3), None
        step, (sigma, gamma) = fitted_step(mesh, k, 1.0, u, p, bc)
    tol, bound = (1e-11, 1e-10) if bc in ("kidney", "paper4") else (1e-13, 1e-11)
    sol, p_h, m_h, S, us, beta, n, inc = ns.picard(step, mesh, k, 1.0, D,
                                                   correct=bc in ("unfitted", "kidney", "paper4"),
                                                   tol=tol)
    e = st.errors(mesh, sol, p_h, u, p, sigma, gamma, D)
    dV = dx(definedonelements=D, bonus_intorder=2)
    eus = sqrt(Integrate((u - us) ** 2 * dV, mesh))
    assert inc < tol and n < 20
    assert max(e[2:6]) < bound and eus < bound
    assert bc in ("unfitted", "kidney", "paper4") or max(e[:2]) < 1e-11   # raw sigma, p: shifted by c unfitted


# Iteration counts (tol 1e-8, k = 1, maxh 0.2) at Re ~ 1 (mu = 1) and Re ~ 10 (mu = 0.1):
# observed 4, 6 (Dirichlet), 8, 12 (mixed), 6, 13 (Example 2), 5, 8 (kidney); the bound is observed + 2
# (agentB-tables/goal4-stage3-*.txt). Against the mutants (goal4-stage3-mutants.txt) only the
# Dirichlet bounds catch beta = u_h (7, 11 steps); on mixed and Example 2 the counts differ by
# at most 1 and the uncorrected u** gives the same counts: the handoff and rate tests catch them.
ITER = {("dirichlet", 1.0): 6, ("dirichlet", 0.1): 8, ("mixed", 1.0): 10, ("mixed", 0.1): 14,
        (EX2, 1.0): 8, (EX2, 0.1): 15, (KIDNEY, 1.0): 7, (KIDNEY, 0.1): 10}


@pytest.mark.parametrize("case, mu", list(ITER))
def test_iterations(case, mu):
    n = ns.table_row(case, 1, 0.2, mu, verbose=False)[3]["iter"]
    assert n <= ITER[case, mu]


# k = 1, mu = 0.1, maxh 0.2 -> 0.1: errors (sigma or sigma - cI, u, u**) pinned on the
# coarse level, EOC bounds below the observed ones, |div beta| at roundoff (observed 3e-14).
# Observed EOC: Dirichlet (sigma, p, u, gamma; trace, u*, u**) 2.21, 2.31, 2.16, 1.97; 3.09,
# 3.21, 3.22; Example 2 (sigma - cI, p - c, u, gamma; trace, u*, u**) 2.16, 2.27, 2.05, 1.92;
# 2.31, 2.35, 2.34. Columns: navier_stokes_stress.COLUMNS.
PINNED = {"dirichlet": ((2.4075829200e-02, 2.0041148316e-02, 4.4937578084e-03), (0, 2, 7),
                        {0: 1.7, 1: 1.7, 2: 1.7, 3: 1.7, 4: 2.7, 5: 2.7, 7: 2.7}),
          EX2: ((5.0291989859e-02, 3.3359772153e-02, 2.3318533404e-02), (1, 3, 7),
                {1: 1.7, 2: 1.7, 3: 1.7, 4: 1.7, 5: 2.0, 6: 2.0, 7: 2.0})}


@pytest.mark.parametrize("case", list(PINNED))
def test_rates(case):
    (_, _, e1, x1), (_, _, e2, x2) = (ns.table_row(case, 1, h, 0.1, verbose=False) for h in (0.2, 0.1))
    ref, cols, lo = PINNED[case]
    for j, r in zip(cols, ref):
        assert abs(e1[j] / r - 1) < 1e-5
    for j, b in lo.items():
        assert log2(e1[j] / e2[j]) > b
    assert x1["div_beta"] < 1e-12 and x2["div_beta"] < 1e-12


@pytest.mark.parametrize("k", [1, 2])
def test_patch_traction_kidney(k):
    """Navier-Stokes patch on the traction kidney hole (the box minus the kidney, traction on
    Gamma_h, Dirichlet on the box), POLY[k] data, mu = 1, maxh 0.2, pointwise S, beta = u**
    uncorrected (as ns.solve on traction). A divergence-free polynomial has inflow on part of
    the closed curve Gamma_h (min u . n_h -1.58, -1.99), but the do-nothing condition with
    gN = sigma n is consistent for any data, so the exact solution is still a fixed point:
    inflow matters for stability, not for exactness. Observed 12 and 10 steps (tol 1e-11),
    raw sigma, p, u, gamma, trace, u* at most 7.6e-11 (k = 1) and 2.2e-11 (k = 2), u** 3.7e-12
    and 1.7e-12; bound 1e-9."""
    from ngstpm import unfitted_domain
    from poisson_tpm import background_mesh
    u, p = POLY[k]
    mesh = background_mesh(0.2)
    D = unfitted_domain(mesh, ns.PHI_KIDNEY)[0]
    sigma, gamma, f = oseen.manufactured(u, p, 1.0, u)

    def step(beta):
        sol, p_h, m_h, _, S = otpm.solve(mesh, k, 1.0, f, u, ns.PHI_KIDNEY, beta, stab="pointwise",
                                         gN=at(sigma))
        return sol, p_h, m_h, S
    sol, p_h, m_h, S, us, beta, n, inc = ns.picard(step, mesh, k, 1.0, D, tol=1e-11)
    e = st.errors(mesh, sol, p_h, u, p, sigma, gamma, D)[:6]
    eus = sqrt(Integrate((u - us) ** 2 * dx(definedonelements=D, bonus_intorder=2), mesh))
    assert inc < 1e-11 and n < 20
    assert max(e) < 1e-9 and eus < 1e-9


def test_paper4():
    """MATLAB's immersed variant of Solano-Vargas Example 4 (ns.PAPER4), k = 1, mu = 1, maxh
    0.2, through table_row (the case data): e(u), e(u**) pinned (observed 2.5272e-3,
    6.4755e-4, goal4-stage4-paper4-mu1-k1.txt), 4 steps, and |div u**|_{D_h} = |m_h| sqrt|D_h|
    (div u** = -m_h on every element; observed relative difference 6e-10)."""
    from ngstpm import unfitted_domain
    mesh, _, e, extra = ns.table_row(ns.PAPER4, 1, 0.2, 1.0, verbose=False)
    D = unfitted_domain(mesh, ns.PHI_PAPER4)[0]
    area = Integrate(CF(1.0) * dx(definedonelements=D), mesh)
    assert abs(e[3] / 2.527e-3 - 1) < 1e-3 and abs(e[7] / 6.476e-4 - 1) < 1e-3
    assert extra["iter"] == 4 and not ns.failed(extra)
    assert abs(extra["div_us"] / (abs(extra["m_h"]) * sqrt(area)) - 1) < 1e-8


def test_kidney():
    """Kidney Dirichlet (Example 3), k = 1, mu = 1, maxh 0.2: e(u) pinned (observed 1.9189e-2,
    goal4-stage4-kidney-mu1-k1.txt)."""
    _, _, e, extra = ns.table_row(KIDNEY, 1, 0.2, 1.0, verbose=False)
    assert not ns.failed(extra) and abs(e[3] / 1.919e-2 - 1) < 1e-3


# Traction (outflow) on the kidney, u = U + 2 sum_i SINK_{c_i} (exactness: the patch test
# above). Smoke test of the case data (k = 1, maxh 0.2): Picard converges within observed + 2 steps (observed 25 at mu = 1, 19 at mu = 0.1,
# agentB-tables/goal4-stage4-logs/traction-kidney-mu*-k1-h0.2.log), the last beta has outflow
# on Gamma_h (min beta . n_h observed 3.01 and 3.74; at mu = 0.1 the early iterates do not, -19), |div beta| at roundoff (observed 3e-12 and
# 8e-13; u** is not corrected on traction boundaries), u error pinned (1.144e+00, 1.527e-01).
KIDNEY_TRACTION = {1.0: (27, 1.1438e+00), 0.1: (21, 1.5270e-01)}


@pytest.mark.parametrize("mu", list(KIDNEY_TRACTION))
def test_traction_kidney(mu):
    _, _, e, extra = ns.table_row("traction kidney", 1, 0.2, mu, verbose=False)
    steps, eu = KIDNEY_TRACTION[mu]
    assert not ns.failed(extra) and extra["iter"] <= steps
    assert extra["min_bn"] > 1 and extra["div_beta"] < 1e-10
    assert abs(e[2] / eu - 1) < 1e-3


@pytest.mark.parametrize("case", ["dirichlet", "traction"])
def test_superlu(case):
    """SciPy SuperLU (the driver's default; on the Dirichlet case inside the bordered solve)
    gives the UMFPACK solution: k = 1, mu = 1, maxh 0.2, same step count, dofs and errors to
    1e-10 (observed 1.7e-13, agentB-tables/goal4-superlu.txt)."""
    import numpy as np
    r = {s: ns.table_row(case, 1, 0.2, 1.0, "global", verbose=False, solver=s)
         for s in ("umfpack", "superlu")}
    a, b = (r[s][1][1].gfu.vec.FV().NumPy() for s in r)
    assert r["umfpack"][3]["iter"] == r["superlu"][3]["iter"]
    assert np.linalg.norm(a - b) < 1e-10 * np.linalg.norm(a)
    assert max(abs(e / f - 1) for e, f in zip(r["umfpack"][2], r["superlu"][2])) < 1e-10


@pytest.mark.parametrize("case", ns.CASES)
def test_default_stab(case, monkeypatch):
    """Pointwise S by default in the traction cases only (circular hole: decision 23 (2);
    kidney: orchestrator choice under decision 26); stab overrides. Paper example 4
    (Dirichlet): global, as MATLAB."""
    seen = []
    fitted, unfitted = oseen.solve, otpm.solve

    def spy_fitted(mesh, order, mu, f, g, beta, gN=None, dirichlet=".*", stab="global", factor=None):
        seen.append(stab)
        return fitted(mesh, order, mu, f, g, beta, gN, dirichlet, stab, factor)

    def spy_unfitted(*args, stab="global", **kwargs):
        seen.append(stab)
        return unfitted(*args, stab=stab, **kwargs)
    monkeypatch.setattr(oseen, "solve", spy_fitted)
    monkeypatch.setattr(otpm, "solve", spy_unfitted)
    coarse = (KIDNEY, "traction kidney", ns.PAPER4)    # maxh 0.4 is too coarse for these
    for stab in (None, "global", "pointwise"):
        ns.solve(case, 1, 0.2 if case in coarse else 0.4, maxit=1, stab=stab)
    assert seen == [("pointwise" if case in ("traction", "traction kidney") else "global"), "global",
                    "pointwise"]

"""Stokes flow in stress form (stokes_stress.py) on Omega = {phi < 0}, u = g on its
boundary, by the HDG on an unfitted mesh with the Transfer Path Method.

The rows of stokes_stress.py live on the interior subdomain D_h, with the conservativity
row on its interior faces; on Gamma_h the equation of uhat is the trace transferred to the
end xbar of the path, as in elasticity_tpm.py (Cardenas-Solano 2024, (2.2g)), with
grad u = A sigma + S(gamma), A sigma = dev(sigma)/(2 mu) and sigma_h the whole stress (P_k
part and curl of the bubbles; full=False: P_k part only):

    uhat(x) + int_x^xbar (E (A sigma_h + S(gamma_h))) dl = g(xbar).

Pressure mean: the velocity is prescribed on the whole boundary, so p + c is undetermined
and the constraint (rho, 1)_{D_h} = 0 with the multiplier m is used (rule of
stokes_stress.py). The mass-balance rows <uhat . n, psi>_dK + (m, psi)_K = 0 use uhat on
Gamma_h as it is: the flux defect of the transferred trace goes into m, so m_h is not zero
but tends to 0. The exact p has no zero mean on D_h while p_h has: sigma and p are compared
both raw and modulo the constant c = mean over D_h of (p - p_h) (sigma modulo c I).

Examples 1-3 of elasticity_tpm.py (fitted disk, R ~ h; immersed disk; kidney); tau_F = mu;
pruning off by default (prune=True: elasticity_tpm.prune).
"""
from math import log2

from ngsolve import *

from elasticity import compliance, skew, stress
from elasticity_tpm import EXAMPLES, at, domain, gamma_ds, path_ratio, sigma_modulo_pressure
from ngstpm import interior_ds, transfer_paths, unfitted_domain
import stokes_stress as st

INF = float("inf")


def solve(mesh, order, mu, f, g, phi, D=None, tau=1.0, dirichlet=".*", full=True, convect=None,
          factor=None):
    """g(X, Y): the velocity on Gamma (vector CF); dirichlet: the fitted part of the mesh
    boundary; convect(lhs, trial, test, D): added terms (oseen_stress_tpm.py); factor: as in
    stokes_stress.finish.
    Returns (sigma_h, u_h, gamma_h, uhat_h, ustar_h), p_h, m_h and D."""
    D = unfitted_domain(mesh, phi)[0] if D is None else D
    p = transfer_paths(mesh, phi, D, order=order + 4)                 # x -> p.target on Gamma
    dB = gamma_ds(mesh, D, bonus_intorder=max(6, 2 * order))         # Gamma_h

    fes, top, lhs, rhs, trial, test, sigmahat_n = st.forms(mesh, order, mu, f, dirichlet, tau, D)
    if convect is not None:
        convect(lhs, trial, test, D)
    sP, w1, w2, u, gam, uhat = trial[:6]
    muh = test[5]
    lhs += sigmahat_n * muh * interior_ds(mesh, D)
    sigma = stress(sP, w1, w2) if full else sP
    lhs += uhat.Transfer(p, gradient=compliance(sigma, INF, mu) + skew(gam)) * muh * dB
    rhs += g(*p.target) * muh * dB
    st.add_constraint(lhs, trial, test, D)
    bc = (g(x, y), BND, mesh.Boundaries(dirichlet))
    return (*st.finish(lhs, rhs, fes, top, bc, mesh, order, mu, D, factor=factor), D)


def mean_difference(mesh, p, p_h, D):
    """c = mean over D_h of p - p_h (= mean of tr(sigma_h - sigma)/2)."""
    dV = dx(definedonelements=D, bonus_intorder=4)
    return Integrate((p - p_h) * dV, mesh) / Integrate(CF(1) * dV, mesh)


def table_errors(mesh, sol, p_h, u, p, sigma, gamma, D):
    """|sigma - sigma_h|, |sigma - sigma_h - cI|, |p - p_h - c|, |u - u_h|, |gamma - gamma_h|,
    |P_M u - uhat|_h, |u - u*|, and c."""
    e = st.errors(mesh, sol, p_h, u, p, sigma, gamma, D)
    c = mean_difference(mesh, p, p_h, D)
    dV = dx(definedonelements=D, bonus_intorder=4)
    ep = sqrt(Integrate((p - p_h - c) ** 2 * dV, mesh))
    return (e[0], sigma_modulo_pressure(mesh, sol[0], sigma, D), ep, e[2], e[3], e[4], e[5]), c


COLUMNS = ("|s-sh|", "|s-sh-cI|", "|p-ph-c|", "|u-uh|", "|g-gh|", "|Pu-uh^|h", "|u-u*|")
HEADER = (f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS)
          + f"  {'m_h':>10}  {'c':>10}      R   #D_h")
U = st.U
P = st.P


def eoc_line(h, e, prev):
    line = f"  {h:7.4f}"
    for i, ei in enumerate(e):
        r = f"{log2(prev[1][i] / ei) / log2(prev[0] / h):5.2f}" if prev else "   - "
        line += f"  {ei:9.3e} {r}"
    return line


def study(name, order, hs, mu=1.0, full=True, prune=False, record=None):
    """EOC table; record: a study_io.Record that collects the levels."""
    phi = EXAMPLES[name]
    sigma, gamma, f = st.manufactured(U, P, mu)
    print(f"{name}, k = {order}, mu = {mu}" + ("" if full else ", P_k part of sigma transferred")
          + ("" if name.startswith("Example 1") else (", pruned" if prune else ", not pruned")))
    print(HEADER)
    prev, rows = None, []
    for h in hs:
        mesh, D, dirichlet, _ = domain(name, h, phi, prune)
        sol, p_h, m_h, D = solve(mesh, order, mu, f, at(U), phi, D, dirichlet=dirichlet, full=full)
        e, c = table_errors(mesh, sol, p_h, U, P, sigma, gamma, D)
        R = path_ratio(mesh, phi, D)
        print(eoc_line(h, e, prev) + f"  {m_h:10.2e}  {c:10.2e}  {R:5.3f}  {D.NumSet():5d}")
        if record is not None:
            record.add(h, mesh, e, sol[1].gfu, m_h=m_h, c=c, R=R, nD=D.NumSet())
        prev = (h, e)
        rows.append((e, m_h, c, R, D.NumSet()))
    return rows


if __name__ == "__main__":
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(EXAMPLES))
    p.add_argument("--mu", type=float, default=1.0, help="viscosity (default 1)")
    p.add_argument("--prune", action="store_true", help="prune D_h (MATLAB rules)")
    p.add_argument("--pk", dest="full", action="store_false", help="transfer the P_k part of sigma only")
    a = parse(p)
    if a.k is None:
        # one process per case, within 2 GB each: h = 0.025 needs about 3.5-4 GB for k = 1
        # (multiplier m); Example 2, k = 2 at h = 0.05 is 1.9 GB alone, so it stops at 0.1
        cases = [(name, k, (0.2, 0.1, 0.05) if k == 1 or not name.startswith("Example 2") else (0.2, 0.1), 1.0)
                 for name in EXAMPLES for k in (1, 2)]
        cases += [("Example 2 (unfitted disk)", 1, (0.2, 0.1, 0.05), 1e-3),
                  ("Example 2 (unfitted disk)", 2, (0.2, 0.1), 1e-3)]
        run_defaults(__file__, [["--case", n, "--k", k, "--h", *hs, "--mu", mu] for n, k, hs, mu in cases])
    else:
        record = Record(COLUMNS)
        study(a.case, a.k, a.h, mu=a.mu, full=a.full, prune=a.prune, record=record)
        if a.out:
            save(a.out, record, dict(driver="stokes_stress_tpm", case=a.case, k=a.k, mu=a.mu, tau=1.0,
                                     full=a.full, prune=a.prune))

"""Oseen flow in stress form (oseen_stress.py) on an unfitted mesh with the Transfer Path
Method: the rows of oseen_stress.py on the interior subdomain D_h, with the equation of
uhat on Gamma_h that of stokes_stress_tpm.py (Dirichlet) or stokes_stress_tpm_neumann.py
(traction), unchanged:

    Dirichlet:  uhat(x) + int_x^xbar (E (A sigma_h + S(gamma_h))) dl = g(xbar)
    traction:   sP_h(xbar) nubar - S (u_h - uhat)(x) = gN(xbar) nubar     (do-nothing)

beta is kept out of the stress, so grad u = A sigma + S(gamma) still holds on the paths and
the transferred rows carry no beta (NavierStokesProblem.m:6-9, Solano-Vargas 2022 (3.2)).
Convection enters on Gamma_h only through the local face term <(beta . n) uhat, v>_dK and S;
the convective flux -<(beta . n) uhat, mu> is not added to the rows of uhat on Gamma_h
(MATLAB: the transfer rows replace the flux rows there; on a traction face it is the
do-nothing of oseen_stress.py). S = mu + max |beta . n|/2 over the faces of D_h (MATLAB:
over the faces of the immersed mesh, convective_matrix.m:89), or pointwise.

Pressure mean by the rule of stokes_stress.py: Dirichlet on the whole boundary, constraint
and multiplier m (bordered solve), and the flux defect of the transferred trace goes into
m_h; traction on Gamma, neither. The traction case uses a sink beta = -(x, y)/|x|^2 (sink
in the hole, divergence free in Omega): beta . n = 1/r > 0, so Gamma is an outflow
boundary, as the do-nothing condition needs (oseen_stress.py).

Differences from MATLAB (test_transfer_oseen.m): MATLAB transfers the datum radially
(transfer_mapping, :59) on a structured background grid with R = 0.75 (:57); here the
quasi-normal paths of ngstpm and the examples of elasticity_tpm.py. No transferred
traction for the incompressible fluid was found in MATLAB src; the traction row is that of
stokes_stress_tpm_neumann.py, with MATLAB's do-nothing (skip_faces, get_matrices.m:84).
Pressure mean, bubbles and pointwise S as in oseen_stress.py. MATLAB does not check the
stress form at k = 1 ("unstable under refinement", test_transfer_oseen.m:19-20); here k = 1
is run and asserted at k + 0.7 on one pair of meshes up to h = 0.05 only. The plan's
Example 3 (kidney) is not run. corrected_beta (divfree.py) has no MATLAB counterpart.

Columns: Dirichlet as in stokes_stress_tpm.py, where c = mean over D_h of p - p_h is
geometric, not an error: p_h has zero mean on D_h by the constraint, so c is the mean of the
exact p over D_h; traction: the raw errors, c = mean(p_h - p), and min beta . n_h over Gamma_h (outflow
check).
"""
import numpy as np
from ngsolve import *

import oseen_stress as oseen
import stokes_stress as st
import stokes_stress_tpm as tpm
import stokes_stress_tpm_neumann as neumann
from divfree import corrected_beta, divergence_free
from elasticity_tpm import EXAMPLES, at, domain, path_ratio
from ngstpm import boundary_facets, unfitted_domain
from poisson_tpm import background_mesh

SINK = CF((-x, -y)) / (x * x + y * y)


def solve(mesh, order, mu, f, g, phi, beta, D=None, dirichlet=".*", stab="global", gN=None,
          factor=None, inertia=None, full=True):
    """Dirichlet (gN None; g(X, Y) on Gamma, D and dirichlet as in stokes_stress_tpm.solve)
    or traction on Gamma (gN(X, Y) the stress at xbar; g the velocity on the box, a CF).
    Returns (sigma_h, u_h, gamma_h, uhat_h, ustar_h), p_h, m_h (None with traction), D, S.
    factor: as in stokes_stress.finish; inertia = (c, w): as in oseen_stress.solve, on D_h
    (w enters as f + w); full: as in stokes_stress_tpm.solve (Dirichlet only)."""
    assert gN is None or D is None, "traction: D is that of stokes_stress_tpm_neumann.solve"
    D = unfitted_domain(mesh, phi)[0] if D is None else D
    S = oseen.stabilization(mesh, beta, mu, order, D) if stab == "global" else None
    tau = S / mu if S is not None else 1 + Norm(beta * specialcf.normal(2)) / (2 * mu)

    def add(lhs, trial, test, D):
        # no convective flux in the rows of uhat on Gamma_h (transferred rows) nor on the
        # fitted mesh boundary (do-nothing on natural faces; muh vanishes on Dirichlet ones)
        skip = boundary_facets(mesh, D)[1]
        for el in mesh.Elements(BND):
            for d in skip.space.GetDofNrs(el):
                skip.vec[d] = 1
        oseen.convection(lhs, trial, test, beta, skip, D)
        if inertia is not None:
            lhs += inertia[0] * InnerProduct(trial[3], test[3]) * dx(definedonelements=D)
    if inertia is not None:
        f = f + inertia[1]
    if gN is None:
        sol, p_h, m_h, D = tpm.solve(mesh, order, mu, f, g, phi, D, tau, dirichlet, full=full,
                                     convect=add, factor=factor)
    else:
        sol, p_h, D = neumann.solve(mesh, order, mu, f, g, gN, phi, tau, convect=add,  # same D
                                    factor=factor)
        m_h = None
    return sol, p_h, m_h, D, S


def min_outflow(mesh, beta, D, order):
    """min of beta . n_h over the Gauss points of the faces of Gamma_h (n_h outward of D_h),
    with the rule of oseen_stress.stabilization."""
    s = oseen.edge_points(order, beta)
    gamma = boundary_facets(mesh, D)[0]
    pts, nrm = [], []
    for f in mesh.facets:
        if gamma[f.nr]:
            a, b = (np.array(mesh.vertices[v.nr].point) for v in f.vertices)
            el = next(e for e in f.elements if D[e.nr])
            c = np.mean([mesh.vertices[v.nr].point for v in mesh[el].vertices], axis=0)
            t = b - a
            n = np.array([t[1], -t[0]]) / np.linalg.norm(t)
            n = n if n @ ((a + b) / 2 - c) > 0 else -n
            pts += [a + si * t for si in s]
            nrm += [n] * len(s)
    pts, nrm = np.array(pts), np.array(nrm)
    return np.einsum("pi,pi->p", beta(mesh(pts[:, 0], pts[:, 1])), nrm).min()


def div_columns(mesh, order, mu, sol, m_h, D):
    """|div u**| and |div beta| on D_h, beta = u** + (m_h/2)(x - x_c) (divfree.corrected_beta;
    diagnostics only)."""
    us = divergence_free(mesh, order, mu, *sol[:4], D)
    beta = corrected_beta(mesh, us, m_h, D)
    dV = dx(definedonelements=D, bonus_intorder=2)
    return tuple(sqrt(Integrate(div(b) ** 2 * dV, mesh)) for b in (us, beta))


CASES = ["Example 1 (fitted disk)", "Example 2 (unfitted disk)", "traction"]
TAIL = f"  {'m_h':>10}  {'c':>10}  {'S':>7}  {'|divu**|':>9}  {'|divbeta|':>9}      R   #D_h"
TAIL_N = TAIL + f"  {'min b.n':>8}"


def study(name, order, hs, mu=1.0, stab="global", record=None):
    """EOC table: Dirichlet (Examples 1, 2 of elasticity_tpm.py, beta = (1 + y, x)) with the
    columns of stokes_stress_tpm.py, or traction on the hole of stokes_stress_tpm_neumann.py
    (beta = SINK) with the raw columns and c = mean(p_h - p); record: a study_io.Record."""
    traction = name == "traction"
    beta = SINK if traction else oseen.BETA
    phi = neumann.PHI if traction else EXAMPLES[name]
    sigma, gamma, f = oseen.manufactured(st.U, st.P, mu, beta)
    columns = neumann.COLUMNS if traction else tpm.COLUMNS
    print(f"{name}, k = {order}, mu = {mu}, stab = {stab}, not pruned")
    print(f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in columns) + (TAIL_N if traction else TAIL))
    prev, rows = None, []
    for h in hs:
        if traction:
            mesh = background_mesh(h)
            sol, p_h, m_h, D, S = solve(mesh, order, mu, f, st.U, phi, beta, stab=stab, gN=at(sigma))
            e = st.errors(mesh, sol, p_h, st.U, st.P, sigma, gamma, D)[:6]
            c = -tpm.mean_difference(mesh, st.P, p_h, D)
        else:
            mesh, D, dirichlet, _ = domain(name, h, phi)
            sol, p_h, m_h, D, S = solve(mesh, order, mu, f, at(st.U), phi, beta, D, dirichlet, stab)
            e, c = tpm.table_errors(mesh, sol, p_h, st.U, st.P, sigma, gamma, D)
        dus, dbeta = div_columns(mesh, order, mu, sol, m_h, D)
        bn = min_outflow(mesh, beta, D, order) if traction else None
        R = path_ratio(mesh, phi, D)
        print(tpm.eoc_line(h, e, prev) + f"  {'-' if m_h is None else f'{m_h:.2e}':>10}  {c:10.2e}"
              f"  {'-' if S is None else f'{S:.4f}':>7}  {dus:9.2e}  {dbeta:9.2e}  {R:5.3f}"
              f"  {D.NumSet():5d}" + (f"  {bn:8.4f}" if traction else ""), flush=True)
        if record is not None:
            record.add(h, mesh, e, sol[1].gfu, m_h=m_h, c=c, S=S, div_ustar=dus, div_beta=dbeta,
                       R=R, nD=D.NumSet(), min_bn=bn)
        prev = (h, e)
        rows.append((e, m_h, c, S, dus, dbeta))
    return rows


if __name__ == "__main__":
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, CASES)
    p.add_argument("--mu", type=float, default=1.0, help="viscosity (default 1)")
    p.add_argument("--stab", choices=("global", "pointwise"), default="global",
                   help="S = mu + max|beta.n|/2 over D_h (default) or mu + |beta.n|/2 per face")
    a = parse(p)
    if a.k is None:
        # one process per case; measured peaks (agentB-tables/goal4-stage2-*.txt): k = 1 below
        # 0.8 GB, k = 2 1.2 GB (Examples 1, 2) and 1.74 GB (traction)
        run_defaults(__file__, [["--case", c, "--k", k, "--h", 0.2, 0.1, 0.05, "--mu", mu]
                                for c in CASES for k in (1, 2) for mu in (1.0, 0.1)])
    else:
        record = Record(neumann.COLUMNS if a.case == "traction" else tpm.COLUMNS)
        study(a.case, a.k, a.h, a.mu, a.stab, record)
        if a.out:
            save(a.out, record, dict(driver="oseen_stress_tpm", case=a.case, k=a.k, mu=a.mu,
                                     stab=a.stab, tau_F=1.0, prune=False,
                                     beta="sink" if a.case == "traction" else "(1 + y, x)"))

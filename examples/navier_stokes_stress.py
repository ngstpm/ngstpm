"""Steady Navier-Stokes flow in stress-velocity-rotation form by Picard iteration: each step
an Oseen solve of oseen_stress.py (fitted) or oseen_stress_tpm.py (unfitted, Transfer
Path Method) with the convective velocity beta taken from the previous iterate
(MATLAB @NavierStokesProblem/solve.m:68-118; Solano-Vargas 2022, Section 6.4):

    A sigma - grad u + S(gamma) = 0,   (u . grad) u - div sigma = f,

    step 1: Stokes (beta = 0);   step n + 1: beta = u**_n, the divergence-free H(div)
    velocity of step n (divfree.divergence_free), and S = mu + max |beta . n|/2 from it
    (global S; pointwise S = mu + |beta . n|/2 by default in the traction case, --stab);
    on transferred boundaries with the multiplier m (Dirichlet on Gamma) beta =
    u**_n + (m_h/2)(x - x_c) (divfree.corrected_beta, exactly divergence free).

Stop when |u_h^n - u_h^(n-1)|_{L2(D_h)} / |u_h^(n-1)|_{L2(D_h)} < 1e-8, at most 50 steps
(solve.m:39-40). Iteration count n: the number of solves, the Stokes step included, as the
loop counter n of MATLAB's @NavierStokesProblem/solve.m (:79-80); MATLAB's output history
holds one increment per Oseen step (solve.m:74, :116: none for the Stokes step), so
numel(history) = n - 1. (MATLAB line numbers are those of its HEAD 23eccf76 unless marked
"working tree".)

Cases (manufactured, f = (u . grad) u - div sigma, NavierStokesProblem/
get_manufactured_solution.m with beta = u): U, P of stokes_stress.py, Dirichlet on the
unit square and on Examples 1, 2, 3 (kidney) of elasticity_tpm.py; mixed (traction on right|top) with
u = U + (1, 1), so that u . n = 1 on the traction faces (U . n = 0 there: do-nothing needs
outflow, oseen_stress.py); traction on the hole of stokes_stress_tpm_neumann.py with
u = U + SINK, u . n_hole >= 1/0.6 - 1 > 0 (outflow), Dirichlet on the box; traction kidney: the same box
minus the kidney of Example 3 (phi = -phi_kidney), traction on the kidney, Dirichlet on the box,
u = U + 2 (SINK_{c_1} + SINK_{c_2} + SINK_{c_3}), c = (0.15, 0), (-0.15, +-0.375) inside the
kidney (a single sink loses outflow on Gamma_h: agentB-tables/goal4-stage4-kidney-outflow.txt;
exact min u . n_h on Gamma_h 3.45, 2.97, 0.79 at h = 0.2, 0.1, 0.05, k = 1 points);
paper example 4: MATLAB's immersed variant of Solano-Vargas 2022 Example 4
(examples/hdg/pde_navier_stokes_unfitted_convergence.m:2-5: Example 4 geometry, Example 3
construction, Gamma_h at distance O(h)); the paper interpolates the circle (O(h^2) gap,
Section 6.4.1), so its Table 4 is not like for like. The disk of radius 0.75 immersed in
the box (-1, 1)^2 (:16-17, :38), u = (cos x cos y, sin x sin y), p = sin(x^2 + y^2) (:20-21;
the constant c is the mean over D_h, as in every Dirichlet case), Dirichlet on Gamma, mu = 1
(:14). For these data (u . grad) u = grad((cos^2 x + sin^2 y)/2) is a pure gradient: the
convection changes only the pressure, so the case is a weak test of the Picard loop.
Extra columns of this case: |div u**| = |u**|'s divergence over D_h, uncorrected (to
compare with the paper's |div u*_h|_{D_h}), and incr u** = the last relative L2(D_h)
increment of u** (the paper's stopping quantity; the stop itself stays on u_h); each step
also logs it.

Stabilisation: global S (MATLAB's) in every case but traction and traction kidney, where the
pointwise S is the default: traction, decision 23 (2) (Jaime, 2026-11-22: with the global S,
k = 2, mu = 0.1, h = 0.2 fails at the cap, agentB-tables/goal4-stage3-traction-mu0.1-k2.txt);
traction kidney, an orchestrator choice under decision 26 (autonomous mode, Jaime to
confirm); --stab overrides; the table title states S.

Differences from MATLAB: the stopping norm is the L2(D_h) norm of u_h, MATLAB the Frobenius
norm of its Dubiner dofs (solve.m:121-127; equivalent on quasi-uniform meshes, but NGSolve's
L2 basis is not orthonormal, so the dof norm would weigh the modes differently);
corrected_beta (decision 21 (A)) has a MATLAB counterpart only in MATLAB's working tree
(uncommitted, 2026-11-22): src/fem/problems/pdes/corrected_beta.m, the same formula
(beta = u* - (c/2)(x - x_c), c = the mean divergence of u* = -m_h here), called from
@NavierStokesProblem/solve.m:115-117 and solve_navier_stokes.m:89-91 (working tree); at
MATLAB's HEAD beta = u* uncorrected (solve.m:113-114); pressure mean, bubbles, the
transfer and the pointwise S as in oseen_stress.py and oseen_stress_tpm.py (the pointwise
S, default for traction, is not in MATLAB: IncompressibleElasticityMatrices.m:78-86); MATLAB
warns at the cap (solve_navier_stokes.m:70-72), here the run is reported as failed.
Paper example 4: unstructured background mesh of size h (MATLAB: StructuredSimplicialGrid
N x N, h = 2/N, :37-38), quasi-normal paths (MATLAB: radial, :18), and the "Picard" column of
MATLAB's table prints numel(history) (:56), the Oseen steps only (solve.m:116), so it reads
n - 1 for our n (which includes the Stokes step);
the paper's u = (sin x sin y, cos x cos y) (Section 6.3) is MATLAB's with the components
swapped; MATLAB is followed.

Solver: SciPy SuperLU (superlu.py) for the HDG system and the element systems of u**
(--solver umfpack for NGSolve's UMFPACK, which leaks memory at each factorization).
"""
import os
import resource
import subprocess

import numpy as np
from ngsolve import *

import oseen_stress as oseen
import oseen_stress_tpm as otpm
import stokes_stress as st
import stokes_stress_tpm as tpm
import stokes_stress_tpm_neumann as neumann
import superlu
from divfree import corrected_beta, divergence_free
from elasticity_tpm import EXAMPLES, at, domain, path_ratio
from ngstpm import unfitted_domain
from poisson_tpm import background_mesh

TOL, MAXIT = 1e-8, 50


def rss():
    """(current, peak) resident set size of this process in MB."""
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
    return int(r.stdout) / 1024, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def picard(step, mesh, order, mu, D=None, correct=False, beta=CF((0, 0)), tol=TOL, maxit=MAXIT,
           log=None, factor=None, log_us=None):
    """step(beta) -> (sol, p_h, m_h, S): one Oseen solve. The first with the given beta (zero:
    Stokes), then beta = u** of the previous iterate (corrected_beta with m_h if correct).
    log(n, increment, S, beta): called after each step with the beta of the next step.
    A NaN increment does not stop the loop (not converged). Returns sol, p_h, m_h, S of the last
    step, u** of the last step, the beta of the last step, the number of steps and the
    last increment (inf after one step). factor: the factorization of the element systems of
    u** (divfree.divergence_free; default UMFPACK). log_us(n, us, D): called after each step
    with its u** (before log)."""
    dV = dx(definedonelements=D)
    u_old, inc, n = None, float("inf"), 0
    while not inc < tol and n < maxit:
        n += 1
        last = beta
        sol, p_h, m_h, S = step(beta)
        us = divergence_free(mesh, order, mu, *sol[:4], D, factor)
        beta = corrected_beta(mesh, us, m_h, D) if correct else us
        u = GridFunction(sol[1].space)
        u.vec.data = sol[1].vec
        if u_old is not None:
            inc = sqrt(Integrate(InnerProduct(u - u_old, u - u_old) * dV, mesh)
                       / Integrate(InnerProduct(u_old, u_old) * dV, mesh))
        u_old = u
        if log_us is not None:
            log_us(n, us, D)
        if log is not None:
            log(n, inc, S, beta)
    return sol, p_h, m_h, S, us, last, n, inc


def min_outflow(mesh, beta, faces, order):
    """min of beta . n (n outward) over the Gauss points of the boundary edges of the
    regions faces (rule of oseen_stress.stabilization)."""
    s = oseen.edge_points(order, beta)
    pts, nrm = [], []
    for el in mesh.Elements(BND):
        if el.mat in faces:
            a, b = (np.array(mesh.vertices[v.nr].point) for v in el.vertices)
            K = mesh[mesh[el.edges[0]].elements[0]]
            c = np.mean([mesh.vertices[v.nr].point for v in K.vertices], axis=0)
            t = b - a
            n = np.array([t[1], -t[0]]) / np.linalg.norm(t)
            n = n if n @ ((a + b) / 2 - c) > 0 else -n
            pts += [a + si * t for si in s]
            nrm += [n] * len(s)
    pts, nrm = np.array(pts), np.array(nrm)
    return np.einsum("pi,pi->p", beta(mesh(pts[:, 0], pts[:, 1])), nrm).min()


FITTED = {"dirichlet": {}, "mixed": dict(dirichlet="left|bottom", traction=("right", "top"))}
UNFITTED = ["Example 1 (fitted disk)", "Example 2 (unfitted disk)", "traction", "Example 3 (kidney)",
            "traction kidney", "paper example 4"]
CASES = list(FITTED) + UNFITTED
PAPER4 = "paper example 4"
PHI_PAPER4 = lambda X, Y: sqrt(X * X + Y * Y) - 0.75       # pde_navier_stokes_unfitted_convergence.m:16-17
U_PAPER4 = CF((cos(x) * cos(y), sin(x) * sin(y)))           # :20 (u = curl of -cos x sin y)
P_PAPER4 = sin(x * x + y * y)                               # :21
BOX_PAPER4 = 1.0                                            # background (-1, 1)^2, :38


def paper4_domain(h):
    """mesh (background (-1, 1)^2 of size h) and D_h of paper example 4."""
    mesh = background_mesh(h, BOX_PAPER4)
    return mesh, unfitted_domain(mesh, PHI_PAPER4)[0]
U_MIXED = st.U + CF((1, 1))
U_SINK = CF((st.U[0] - x / (x * x + y * y), st.U[1] - y / (x * x + y * y)))   # U + SINK, by
# components: elasticity_tpm.at cannot Replace x, y in the derivatives of the vector quotient
SINKS, A_SINK = ((0.15, 0.0), (-0.15, 0.375), (-0.15, -0.375)), 2.0   # inside the kidney
U_KIDNEY = CF(tuple(st.U[i] - A_SINK * sum(((x, y)[i] - c[i]) / ((x - c[0]) ** 2 + (y - c[1]) ** 2)
                                             for c in SINKS) for i in (0, 1)))   # by components
PHI_KIDNEY = lambda X, Y: -EXAMPLES["Example 3 (kidney)"](X, Y)     # the box minus the kidney
TRACTION = {"traction": (neumann.PHI, U_SINK), "traction kidney": (PHI_KIDNEY, U_KIDNEY)}
COLUMNS = {"fitted": st.COLUMNS + ("|u-u**|",), "dirichlet": tpm.COLUMNS + ("|u-u**|",),
           "traction": neumann.COLUMNS + ("|u-u**|",)}
TAIL = ("m_h", "c", "S", "iter", "incr", "|divbeta|", "min b.n", "R", "#D_h", "MB")
TAIL_PAPER4 = {"div_us": "|divu**|", "incr_us": "incr u**"}   # extra columns of paper example 4


STAB = {"global": "S = mu + max|beta.n|/2", "pointwise": "S = mu + |beta.n|/2 pointwise"}


def default_stab(case):
    """The stabilisation of a case when none is given: pointwise for traction (circular hole
    or kidney), else global."""
    return "pointwise" if case in TRACTION else "global"


def kind(case):
    return "fitted" if case in FITTED else "traction" if case in TRACTION else "dirichlet"


def solve(case, order, h, mu=1.0, stab=None, tol=TOL, maxit=MAXIT, log=None, solver="superlu",
          log_us=None):
    """The Picard solution of one case on one mesh. Returns mesh, D, the exact (u, p, sigma,
    gamma), the result of picard and phi (None if fitted). solver: "superlu" or "umfpack", for
    every factorization of the loop (the HDG system and the element systems of u**); stab:
    "global", "pointwise" or None (default_stab(case)); log_us: as in picard."""
    stab = default_stab(case) if stab is None else stab
    factor = superlu.inverse if solver == "superlu" else None
    u = (U_MIXED if case == "mixed" else TRACTION[case][1] if case in TRACTION
         else U_PAPER4 if case == PAPER4 else st.U)
    p = P_PAPER4 if case == PAPER4 else st.P
    sigma, gamma, f = oseen.manufactured(u, p, mu, u)
    phi, D = None, None
    if case in FITTED:
        from netgen.geom2d import unit_square
        mesh = Mesh(unit_square.GenerateMesh(maxh=h))
        bc = FITTED[case].get("dirichlet", ".*")
        gN = sigma * specialcf.normal(2) if case == "mixed" else None
        step = lambda beta: oseen.solve(mesh, order, mu, f, u, beta, gN, bc, stab, factor)
        out = picard(step, mesh, order, mu, tol=tol, maxit=maxit, log=log, factor=factor,
                     log_us=log_us)
    else:
        if case in TRACTION:                    # D: that of stokes_stress_tpm_neumann.solve
            phi, mesh = TRACTION[case][0], background_mesh(h)
            D = unfitted_domain(mesh, phi)[0]
            args = dict(stab=stab, gN=at(sigma))
        elif case == PAPER4:
            phi = PHI_PAPER4
            mesh, D = paper4_domain(h)
            args = dict(D=D, dirichlet=".*", stab=stab)
        else:
            phi = EXAMPLES[case]
            mesh, D, bc, _ = domain(case, h, phi)
            args = dict(D=D, dirichlet=bc, stab=stab)
        g = u if case in TRACTION else at(u)

        def step(beta):
            sol, p_h, m_h, _, S = otpm.solve(mesh, order, mu, f, g, phi, beta, factor=factor, **args)
            return sol, p_h, m_h, S
        if case in TRACTION and log is not None:      # outflow of the iterates on Gamma_h
            log_ = log
            log = lambda n, inc, S, beta: log_(n, inc, S, beta,
                                               otpm.min_outflow(mesh, beta, D, order))
        out = picard(step, mesh, order, mu, D, correct=case not in TRACTION, tol=tol, maxit=maxit,
                     log=log, factor=factor, log_us=log_us)
    return mesh, D, (u, p, sigma, gamma), out, phi


def table_row(case, order, h, mu=1.0, stab=None, tol=TOL, maxit=MAXIT, verbose=True,
              solver="superlu"):
    """Errors (COLUMNS[kind(case)]) and the extra columns (TAIL) of one Picard run; with
    verbose, one line per step (increment, S, current and peak RSS; paper example 4: also the
    relative L2(D_h) increment of u**, the paper's stopping quantity)."""
    us_inc = [float("inf")]           # paper example 4: [its increment, u** of the last step]

    def log_us(n, us, D):             # u** is not zero off D_h: integrate over D_h only
        if len(us_inc) > 1:
            old, m, dV = us_inc[1], us.space.mesh, dx(definedonelements=D)
            us_inc[0] = sqrt(Integrate(InnerProduct(us - old, us - old) * dV, m)
                             / Integrate(InnerProduct(old, old) * dV, m))
        copy = GridFunction(us.space)
        copy.vec.data = us.vec
        us_inc[1:] = [copy]

    def log(n, inc, S, beta, bn=None):
        if verbose:
            print(f"    step {n:2d}  increment {inc:9.2e}  S {S if S is None else round(S, 4)}"
                  + ("" if bn is None else f"  min beta.n on Gamma_h {bn:.3g}")
                  + ("" if case != PAPER4 else f"  u** increment {us_inc[0]:9.2e}")
                  + "  RSS %.0f MB (peak %.0f MB)" % rss(), flush=True)
    mesh, D, (u, p, sigma, gamma), out, phi = solve(case, order, h, mu, stab, tol, maxit, log,
                                                    solver, log_us if case == PAPER4 else None)
    sol, p_h, m_h, S, us, beta, n, inc = out
    dV = dx(definedonelements=D, bonus_intorder=4)
    eus = sqrt(Integrate(InnerProduct(u - us, u - us) * dV, mesh))
    divb = sqrt(Integrate(div(beta) ** 2 * dx(definedonelements=D, bonus_intorder=2), mesh))
    c = R = bn = None
    if kind(case) == "fitted":
        e = st.errors(mesh, sol, p_h, u, p, sigma, gamma)
        if case == "mixed":
            bn = min_outflow(mesh, beta, FITTED[case]["traction"], order)
    elif case in TRACTION:
        e = st.errors(mesh, sol, p_h, u, p, sigma, gamma, D)[:6]
        c = -tpm.mean_difference(mesh, p, p_h, D)
        bn = otpm.min_outflow(mesh, beta, D, order)
    else:
        e, c = tpm.table_errors(mesh, sol, p_h, u, p, sigma, gamma, D)
    if phi is not None:
        R = path_ratio(mesh, phi, D)
    extra = dict(m_h=m_h, c=c, S=S, iter=n, incr=inc, div_beta=divb, min_bn=bn, R=R,
                 nD=None if D is None else D.NumSet(), MB=rss()[1])
    if case == PAPER4:                # uncorrected u** of the last step (= -m_h sqrt|D_h|)
        extra["div_us"] = sqrt(Integrate(div(us) ** 2 * dx(definedonelements=D, bonus_intorder=2),
                                         mesh))
        extra["incr_us"] = us_inc[0]
    return mesh, sol, (*e, eus), extra


def fmt(v, key):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'-':>9}"
    return {"iter": f"{int(v):9d}", "nD": f"{int(v):9d}", "MB": f"{v:9.0f}",
            "S": f"{v:9.4f}", "R": f"{v:9.3f}", "min_bn": f"{v:9.4f}"}.get(key, f"{v:9.2e}")


def failed(extra):
    """Picard did not reach the tolerance (cap, or a NaN increment)."""
    return not extra["incr"] < TOL


def print_table(title, columns, hs, E, extras, note=None):
    """EOC table: errors E (levels x columns) on the mesh sizes hs, extras a list of dicts.
    A level where Picard failed is marked FAILED and gives no EOC; note: printed below."""
    more = [k for k in TAIL_PAPER4 if k in extras[0]]
    keys = ("m_h", "c", "S", "iter", "incr", "div_beta", "min_bn", "R", "nD", "MB", *more)
    print(title)
    print(f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in columns)
          + "".join(f"  {t:>9}" for t in TAIL + tuple(TAIL_PAPER4[k] for k in more)))
    for i, h in enumerate(hs):
        line = f"  {h:7.4f}"
        rate = i and not failed(extras[i]) and not failed(extras[i - 1])
        for j, ej in enumerate(E[i]):
            r = (f"{np.log(E[i - 1][j] / ej) / np.log(hs[i - 1] / h):4.2f}"
                 if rate and ej > 0 else "  - ")
            line += f"  {ej:9.3e}  {r}"
        print(line + "".join(f"  {fmt(extras[i].get(k), k)}" for k in keys)
              + (f"  FAILED (Picard: no convergence in {int(extras[i]['iter'])} steps)"
                 if failed(extras[i]) else ""), flush=True)
    if note:
        print(f"  NOTE: {note}")


def study(case, order, hs, mu=1.0, stab=None, record=None, solver="superlu"):
    """EOC table on the mesh sizes hs (in this process; the drivers run one level per
    process, see __main__); record: a study_io.Record."""
    stab = default_stab(case) if stab is None else stab
    title = (f"{case}, k = {order}, mu = {mu}, {STAB[stab]}, Picard tol {TOL:g}, cap {MAXIT}, "
             f"{solver}")
    print(title, flush=True)
    E, extras = [], []
    for h in hs:
        mesh, sol, e, extra = table_row(case, order, h, mu, stab, solver=solver)
        if failed(extra):
            print(f"  h = {h}: Picard did not converge in {extra['iter']} steps (FAILED)")
        E.append(e)
        extras.append(extra)
        if record is not None:
            record.add(h, mesh, e, sol[1].gfu, **extra)
    print_table(title, COLUMNS[kind(case)], hs, E, extras)
    return E, extras


def merge(paths, note=None):
    """The EOC table of the levels saved by separate processes (--out of each level)."""
    data = [np.load(p) for p in paths]
    import json
    prm = json.loads(str(data[0]["params"]))
    hs = [float(d["h"][0]) for d in data]
    E = [d["errors"][0] for d in data]
    keys = ("m_h", "c", "S", "iter", "incr", "div_beta", "min_bn", "R", "nD", "MB", *TAIL_PAPER4)
    extras = [{k: float(d[k][0]) for k in keys if k in d} for d in data]
    print_table(f"{prm['case']}, k = {prm['k']}, mu = {prm['mu']}, {STAB[prm['stab']]}, "
                f"Picard tol {TOL:g}, cap {MAXIT}, {prm.get('solver', 'umfpack')}, one level per process",
                list(data[0]["columns"]), hs, E, extras, note)


# one process per mesh level (the plan); the levels that fit under the 1.85 GB guard.
# Memory: with the default SuperLU there is no per-step leak (agentB-tables/goal4-superlu.txt);
# the RSS of a run wanders but stays bounded (traction kidney k = 2, mu = 1 at h = 0.1: 418 MB
# after the Stokes step, 418-558 MB over 17 steps, goal4-stage4-logs/
# traction-kidney-mu1-k2-h0.1.log).
# Stage 4 (SuperLU, agentB-tables/goal4-stage4-*.txt): kidney, traction kidney and paper
# example 4, k = 1, 2, all fit at h = 0.05 (peaks: kidney 213-299 MB, traction kidney k = 1
# 813-830 MB, k = 2 1.72-1.73 GB, paper example 4 294 MB (k = 1), 570 MB (k = 2); paper
# example 4 was measured at mu = 1 only, the loop below also runs mu = 0.1).
# The stage-3 entries are the levels measured with UMFPACK before SuperLU became the default
# (agentB-tables/goal4-stage3-*.txt), not widened here. With UMFPACK memory grew by about one
# factorization per Picard step (not freed), so the peak was base + steps x growth:
# unfitted k = 2 at h = 0.05 (about 95 MB x 4 per step) and traction k = 1 at 0.05 did not
# fit; traction k = 2 at h = 0.1 peaked at 1.97 GB, so it is not a default (and at h = 0.2,
# mu = 0.1 Picard does not converge: the iterates lose outflow on Gamma_h, min u** . n down to
# about -100, and u_h itself blows up (|u - u_h| = 60 at one step, off Gamma_h); see
# agentB-tables/goal4-stage3-traction-mu0.1-k2.txt).
# Fitted mixed k = 2 at h = 0.05 (12 steps at mu = 0.1) peaked at 1.33 GB once and was killed
# at 1.98 GB in a rerun (macOS RSS varies between runs), so it stops at h = 0.1.
DEFAULTS = {**{(k, c): (0.2, 0.1, 0.05) for k in (1, 2) for c in FITTED}, (2, "mixed"): (0.2, 0.1),
            **{(1, c): (0.2, 0.1, 0.05) for c in UNFITTED[:2]},
            **{(2, c): (0.2, 0.1) for c in UNFITTED[:2]}, (1, "traction"): (0.2, 0.1),
            **{(k, c): (0.2, 0.1, 0.05) for c in ("Example 3 (kidney)", "traction kidney", PAPER4)
               for k in (1, 2)}}


if __name__ == "__main__":
    import sys
    import tempfile
    from study_io import Record, parse, parser, save

    p = parser(__doc__, CASES)
    p.add_argument("--mu", type=float, default=1.0, help="viscosity (default 1)")
    p.add_argument("--stab", choices=("global", "pointwise"), default=None,
                   help="S = mu + max|beta.n|/2 or mu + |beta.n|/2 pointwise (default: "
                        "pointwise for traction and traction kidney, global otherwise)")
    p.add_argument("--solver", choices=("superlu", "umfpack"), default="superlu",
                   help="sparse LU of every factorization in the loop (default SciPy SuperLU)")
    a = parse(p)
    a.stab = default_stab(a.case) if a.stab is None else a.stab
    if a.k is None:
        tmp = tempfile.mkdtemp()
        for (k, case), hs in DEFAULTS.items():
            for mu in (1.0, 0.1):
                paths = []
                for h in hs:
                    paths.append(os.path.join(tmp, f"{CASES.index(case)}-{k}-{mu}-{h}.npz"))
                    subprocess.run([sys.executable, __file__, "--case", case, "--k", str(k),
                                    "--h", str(h), "--mu", str(mu), "--out", paths[-1]], check=True)
                merge(paths)
    else:
        record = Record(COLUMNS[kind(a.case)])
        study(a.case, a.k, a.h, a.mu, a.stab, record, a.solver)
        if a.out:
            save(a.out, record, dict(driver="navier_stokes_stress", case=a.case, k=a.k, mu=a.mu,
                                     stab=a.stab, tau_F=1.0, tol=TOL, maxit=MAXIT, beta="u**",
                                     solver=a.solver,
                                     u={"mixed": "U + (1, 1)", "traction": "U + SINK",
                                        "traction kidney": "U + 2 sum_i SINK_{c_i}",
                                        PAPER4: "(cos x cos y, sin x sin y)"}.get(a.case, "U")))

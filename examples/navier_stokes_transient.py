"""Unsteady Navier-Stokes flow in stress-velocity-rotation form: the theta-scheme of MATLAB's
IncompressibleElasticityProblem (get_matrices.m:194-216, get_kinematic_state.m) with, at each
time step, the Picard loop of navier_stokes_stress.py (MATLAB @NavierStokesProblem/solve.m,
"one time step when density > 0"):

    A sigma - grad u + S(gamma) = 0,   rho u_t + rho (u . grad) u - div sigma = f.

One step t^n -> t^{n+1} = (n + 1) dt, data at t^{n+1}, written as backward Euler with
dt -> theta dt plus the history of the acceleration a^n:

    (rho/(theta dt)) (u, v) + [Oseen rows with beta] = (f, v) + rho (u^n/(theta dt)
                                                       + ((1 - theta)/theta) a^n, v),
    a^{n+1} = (u^{n+1} - u^n)/(theta dt) - ((1 - theta)/theta) a^n,

u^n, a^n in P_k (the velocity space). The convection carries rho_c = rho (get_matrices.m:80-81):
the Oseen solve gets rho beta, which scales the convective rows and S = mu + rho max|beta . n|/2
alike. Picard in each step: beta = u** (divfree.divergence_free, the postprocessed velocity) of
the previous iterate; the first iterate of step n + 1 takes u** of step n (MATLAB:
Data.convective_velocity, the caller's previous beta), that of step 1 beta = 0. Stop as in
navier_stokes_stress.picard (relative L2 increment of u_h < 1e-8, at most 50 solves per step);
a step that does not converge ends the run as failed.

Schemes: "euler" (theta = 1) and "cn" (theta = 1/2 after one Euler step, as
fsi_transient.py and MATLAB's hemodynamics.m), so a^0 is never read.

Cases (manufactured): those of navier_stokes_stress.py with u = U cos t, p = P cos t (U, P
the steady fields of each case), f = rho u_t + rho (u . grad) u - div sigma. Fitted:
"dirichlet" and "mixed" (traction on right|top, U = U + (1, 1), u . n = cos t > 0 there).
Unfitted (Transfer Path Method, oseen_stress_tpm.py): Dirichlet on Gamma, Examples 1, 2, 3 of
elasticity_tpm.py and paper example 4, beta = corrected_beta(u**) as in the steady loop;
traction on Gamma ("traction", "traction kidney"), u . n_Gamma > 0 kept for t < pi/2 (the sinks
scale with cos t), beta = u** uncorrected. Stabilisation: navier_stokes_stress.default_stab
(pointwise S for the traction cases, global otherwise).

Initial velocity (start): "elliptic" (default), u^0 = u_h of the Stokes solve of the case at
t = 0 with the data of (u, p)(0) (beta = 0, f = -div sigma(0), the boundary data at t = 0);
"l2", MATLAB's L2 projection of u(0) on the elements (initial_state.m:50-51). The L2
projection does not satisfy the discrete constraints (div, and on Gamma_h the transferred
trace), so the first step puts O(h^{k+1}/dt) into a^1; Euler damps it, CN carries it undamped
in sigma and p. Fitted it is small (dirichlet, k = 1, dt = h^2: rates about 2); unfitted it is
not (Example 2, k = 1, dt = h^2, T = 0.32, h = 0.2 -> 0.1: |s - s_h| 0.30 -> 0.19, Euler
0.33 -> 0.070; the exact a^0 instead of the Euler start: 0.28 -> 0.38; h = 0.2, dt = 0.0025,
T = 0.08: 2.07). The Stokes projection is discretely consistent: 0.33 -> 0.070 (rate 2.24) and
0.369 (Euler 0.347).

Differences from MATLAB: the default start (above; MATLAB also projects uhat^0, read by nothing
in a step); the first beta of step 1 is zero (MATLAB: what the caller sets); the stopping norm
as in navier_stokes_stress.py.

Solver: SciPy SuperLU (superlu.py), as navier_stokes_stress.py.
"""
import numpy as np
from ngsolve import *

import navier_stokes_stress as ns
import oseen_stress as oseen
import oseen_stress_tpm as otpm
import stokes_stress as st
import stokes_stress_tpm as tpm
import superlu
from divfree import corrected_beta
from elasticity_tpm import EXAMPLES, at, domain
from navier_stokes_stress import MAXIT, TOL, picard
from ngstpm import restrict, unfitted_domain
from poisson_tpm import background_mesh

t = Parameter(0.0)                                    # the time of the data
SCHEMES = ("euler", "cn")
CASES = ns.CASES
COLUMNS = ns.COLUMNS                                  # by ns.kind(case)


def steady_fields(case):
    """U, P of the case (navier_stokes_stress.solve)."""
    U = (ns.U_MIXED if case == "mixed" else ns.TRACTION[case][1] if case in ns.TRACTION
         else ns.U_PAPER4 if case == ns.PAPER4 else st.U)
    return U, ns.P_PAPER4 if case == ns.PAPER4 else st.P


def exact(case, mu, rho, U=None, P=None):
    """u = U cos t, p = P cos t, sigma, gamma, f at the Parameter t (default U, P: those of
    the case). u by components (elasticity_tpm.at, as navier_stokes_stress.U_SINK)."""
    U0, P0 = steady_fields(case)
    U = U0 if U is None else U
    P = P0 if P is None else P
    u = CF((U[0] * cos(t), U[1] * cos(t)))
    ut = CF((-U[0] * sin(t), -U[1] * sin(t)))
    p = P * cos(t)
    sigma, gamma, f = st.manufactured(u, p, mu)
    gu = CF((u[0].Diff(x), u[0].Diff(y), u[1].Diff(x), u[1].Diff(y)), dims=(2, 2))
    return u, p, sigma, gamma, f + rho * (ut + gu * u)


def scaled(beta, rho):
    """rho beta, a GridFunction if beta is one (oseen.edge_points reads the type)."""
    if rho == 1:
        return beta
    if isinstance(beta, GridFunction):
        b = GridFunction(beta.space)
        b.vec.data = rho * beta.vec
        return b
    return rho * beta


def setup(case, order, h, mu, stab, fields, mesh=None, factor=None):
    """mesh, D (None if fitted), phi (None if fitted) and solver(beta, inertia, f) -> (sol,
    p_h, m_h, S): one Oseen solve of the case, as navier_stokes_stress.solve (f: default the
    source of fields)."""
    u, p, sigma, gamma, f = fields
    if case in ns.FITTED:
        if mesh is None:
            from netgen.geom2d import unit_square
            mesh = Mesh(unit_square.GenerateMesh(maxh=h))
        bc = ns.FITTED[case].get("dirichlet", ".*")
        gN = sigma * specialcf.normal(2) if case == "mixed" else None
        return mesh, None, None, lambda beta, inertia, f=f: oseen.solve(
            mesh, order, mu, f, u, beta, gN, bc, stab, factor, inertia=inertia)
    assert mesh is None, "unfitted: the mesh is that of the case"
    if case in ns.TRACTION:
        phi, mesh = ns.TRACTION[case][0], background_mesh(h)
        D = unfitted_domain(mesh, phi)[0]
        args = dict(stab=stab, gN=at(sigma))
    elif case == ns.PAPER4:
        phi = ns.PHI_PAPER4
        mesh, D = ns.paper4_domain(h)
        args = dict(D=D, dirichlet=".*", stab=stab)
    else:
        phi = EXAMPLES[case]
        mesh, D, bc, _ = domain(case, h, phi)
        args = dict(D=D, dirichlet=bc, stab=stab)
    g = u if case in ns.TRACTION else at(u)

    def solver(beta, inertia, f=f):
        sol, p_h, m_h, _, S = otpm.solve(mesh, order, mu, f, g, phi, beta, factor=factor,
                                         inertia=inertia, **args)
        return sol, p_h, m_h, S
    return mesh, D, phi, solver


def solve(case, order, h, dt, T, mu=1.0, rho=1.0, scheme="cn", mesh=None, log=None, U=None,
          P=None, stab=None, start="elliptic"):
    """The run to t = T. Returns mesh, D (None if fitted), the exact (u, p, sigma, gamma) at T
    (t is left at T), sol, p_h, m_h, S, u** of the last step, and the Picard counts per step;
    failed is True if a step did not converge (the run stops there). mesh: fitted cases only,
    default the unit square of size h; U, P: as in exact; stab: default
    navier_stokes_stress.default_stab(case); start: the initial velocity, "elliptic" (the
    Stokes projection of u(0)) or "l2" (MATLAB's L2 projection)."""
    nsteps = round(T / dt)
    assert abs(nsteps * dt - T) < 1e-12 * max(T, 1), "T must be a multiple of dt"
    stab = ns.default_stab(case) if stab is None else stab
    fields = exact(case, mu, rho, U, P)
    u, p, sigma, gamma, _ = fields
    factor = superlu.inverse
    mesh, D, phi, solver = setup(case, order, h, mu, stab, fields, mesh, factor)
    correct = D is not None and case not in ns.TRACTION
    V = VectorL2(mesh, order=order)
    V = V if D is None else restrict(V, D)            # the velocity space of the solves
    un, an = GridFunction(V), GridFunction(V)        # a^0 = 0: not read (Euler first step)
    t.Set(0.0)
    un.Set(u, definedonelements=D)
    if start == "elliptic":        # the Stokes solve with the data of (u, p)(0): beta = 0, f = -div sigma(0)
        un.vec.data = solver(CF((0, 0)), None, st.manufactured(u, p, mu)[2])[0][1].vec
    beta, iters, failed = CF((0, 0)), [], False
    for n in range(nsteps):
        theta = 1.0 if scheme == "euler" or n == 0 else 0.5
        t.Set((n + 1) * dt)
        c = rho / (theta * dt)
        w = c * un + (rho * (1 - theta) / theta) * an

        def step(b):
            return solver(scaled(b, rho), (c, w))
        sol, p_h, m_h, S, us, _, it, inc = picard(step, mesh, order, mu, D, correct, beta,
                                                  factor=factor)
        iters.append(it)
        if log is not None:
            log(n + 1, it, inc)
        if not inc < TOL:
            failed = True
            break
        u1 = sol[1].vec
        assert len(u1) == len(un.vec)
        a_old = an.vec.CreateVector()        # not aliased: NGSolve evaluates a sum term by term
        a_old.data = an.vec
        an.vec.data = (1 / (theta * dt)) * (u1 - un.vec) - ((1 - theta) / theta) * a_old
        un.vec.data = u1
        beta = corrected_beta(mesh, us, m_h, D) if correct else us
    return mesh, D, (u, p, sigma, gamma), sol, p_h, m_h, S, us, iters, failed


def row(case, order, h, dt, T, mu=1.0, rho=1.0, scheme="cn", stab=None, verbose=False,
        start="elliptic"):
    """Errors at T (COLUMNS[ns.kind(case)]) and the extras of one run."""
    log = (lambda n, it, inc: print(f"    step {n:4d}  Picard {it:2d}  increment {inc:9.2e}",
                                    flush=True)) if verbose else None
    mesh, D, (u, p, sigma, gamma), sol, p_h, m_h, S, us, iters, failed = solve(
        case, order, h, dt, T, mu, rho, scheme, log=log, stab=stab, start=start)
    c = None
    if D is None:
        e = st.errors(mesh, sol, p_h, u, p, sigma, gamma)
    elif case in ns.TRACTION:
        e = st.errors(mesh, sol, p_h, u, p, sigma, gamma, D)[:6]
        c = -tpm.mean_difference(mesh, p, p_h, D)
    else:
        e, c = tpm.table_errors(mesh, sol, p_h, u, p, sigma, gamma, D)
    eus = sqrt(Integrate(InnerProduct(u - us, u - us) * dx(definedonelements=D, bonus_intorder=4),
                         mesh))
    return mesh, sol, (*e, eus), dict(m_h=m_h, c=c, S=S, max_iter=max(iters),
                                      mean_iter=float(np.mean(iters)), steps=len(iters),
                                      nD=None if D is None else D.NumSet(), failed=failed)


def fmt(v, spec, width):
    return f"{'-':>{width}}" if v is None else f"{v:{width}{spec}}"


def study(case, order, hs, dts, T, mu=1.0, rho=1.0, scheme="cn", stab=None, record=None,
          start="elliptic"):
    """EOC table over the pairs (h, dt): rates in h, or in dt when h is fixed."""
    stab = ns.default_stab(case) if stab is None else stab
    var = "dt" if len(set(hs)) == 1 else "h"
    title = (f"{case}, k = {order}, mu = {mu}, rho = {rho}, {scheme}"
             f"{' (Euler start)' if scheme == 'cn' else ''}, u^0 {start}, T = {T}, {ns.STAB[stab]}, "
             f"Picard tol {TOL:g}, cap {MAXIT}, rates in {var}")
    print(title, flush=True)
    print(f"  {'h':>7}  {'dt':>9}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS[ns.kind(case)])
          + f"  {'m_h':>9}  {'c':>9}  {'S':>8}  {'max it':>6}  {'mean it':>7}  {'steps':>5}"
          f"  {'#D_h':>5}")
    prev = None
    for h, dt in zip(hs, dts):
        mesh, sol, e, ex = row(case, order, h, dt, T, mu, rho, scheme, stab, start=start)
        s = h if var == "h" else dt
        line = f"  {h:7.4f}  {dt:9.3e}"
        for i, ei in enumerate(e):
            r = (f"{np.log(prev[1][i] / ei) / np.log(prev[0] / s):4.2f}"
                 if prev and ei > 0 and not ex["failed"] else "  - ")
            line += f"  {ei:9.3e}  {r}"
        print(line + f"  {fmt(ex['m_h'], '.2e', 9)}  {fmt(ex['c'], '.2e', 9)}"
              f"  {fmt(ex['S'], '.4f', 8)}  {ex['max_iter']:6d}  {ex['mean_iter']:7.2f}"
              f"  {ex['steps']:5d}  {fmt(ex['nD'], 'd', 5)}" + ("  FAILED" if ex["failed"] else ""),
              flush=True)
        if record is not None:
            record.add(h, mesh, e, sol[1].gfu, dt=dt, **{k: v for k, v in ex.items()
                                                          if k != "failed"})
        prev = (s, e)


if __name__ == "__main__":
    from study_io import Record, parse, parser, save

    p = parser(__doc__, CASES)
    p.add_argument("--dt", type=float, nargs="+", required=True,
                   help="time steps, one per --h (or one for all)")
    p.add_argument("--T", type=float, default=1.0, help="final time (default 1)")
    p.add_argument("--scheme", choices=SCHEMES, default="cn", help="default cn (Euler start)")
    p.add_argument("--mu", type=float, default=1.0, help="viscosity (default 1)")
    p.add_argument("--rho", type=float, default=1.0, help="density (default 1)")
    p.add_argument("--start", choices=("elliptic", "l2"), default="elliptic",
                   help="initial velocity: Stokes projection (default) or L2 projection (MATLAB)")
    p.add_argument("--stab", choices=("global", "pointwise"), default=None,
                   help="default: navier_stokes_stress.default_stab(case)")
    a = parse(p)
    if a.k is None:
        p.error("--k and --h are required")
    dts = a.dt * len(a.h) if len(a.dt) == 1 else a.dt
    hs = a.h * len(dts) if len(a.h) == 1 else a.h
    if len(hs) != len(dts):
        p.error("--h and --dt: equal lengths, or one of them a single value")
    stab = ns.default_stab(a.case) if a.stab is None else a.stab
    record = Record(COLUMNS[ns.kind(a.case)])
    study(a.case, a.k, hs, dts, a.T, a.mu, a.rho, a.scheme, stab, record, a.start)
    if a.out:
        save(a.out, record, dict(driver="navier_stokes_transient", case=a.case, k=a.k, mu=a.mu,
                                 rho=a.rho, scheme=a.scheme, start=a.start, T=a.T, stab=stab, tol=TOL,
                                 maxit=MAXIT, beta="u**", solver="superlu", u="U cos t"))

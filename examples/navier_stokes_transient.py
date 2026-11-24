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

Case (manufactured): u = U cos t, p = P cos t (U, P of stokes_stress.py), f = rho u_t +
rho (u . grad) u - div sigma; "dirichlet" on the unit square, "mixed" with traction on
right|top (U + (1, 1) there, as navier_stokes_stress.py: do-nothing needs outflow, so
u = (U + (1, 1)) cos t keeps u . n = cos t > 0 for t < pi/2).

Differences from MATLAB: initial velocity u^0 the L2 projection of u(0) on the elements only
(MATLAB also projects uhat^0, read by nothing in a step); the first beta of step 1 is zero
(MATLAB: what the caller sets); the stopping norm as in navier_stokes_stress.py.

Solver: SciPy SuperLU (superlu.py), as navier_stokes_stress.py.
"""
import numpy as np
from ngsolve import *

import oseen_stress as oseen
import stokes_stress as st
import superlu
from navier_stokes_stress import FITTED, MAXIT, TOL, picard

t = Parameter(0.0)                                    # the time of the data
SCHEMES = ("euler", "cn")
CASES = list(FITTED)
SHIFT = {"dirichlet": CF((0, 0)), "mixed": CF((1, 1))}


def exact(case, mu, rho, U=None, P=None):
    """u = U cos t, p = P cos t, sigma, gamma, f at the Parameter t (default U, P: those of
    the case)."""
    U = st.U + SHIFT[case] if U is None else U
    P = st.P if P is None else P
    u, p, ut = U * cos(t), P * cos(t), -U * sin(t)
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


def solve(case, order, h, dt, T, mu=1.0, rho=1.0, scheme="cn", mesh=None, log=None, U=None,
          P=None):
    """The run to t = T. Returns mesh, the exact (u, p, sigma, gamma) at T (t is left at T),
    sol, p_h, m_h, S, u** of the last step, and the Picard counts per step; failed is True if
    a step did not converge (the run stops there). mesh: given, or the unit square of size h;
    U, P: as in exact."""
    if mesh is None:
        from netgen.geom2d import unit_square
        mesh = Mesh(unit_square.GenerateMesh(maxh=h))
    nsteps = round(T / dt)
    assert abs(nsteps * dt - T) < 1e-12 * max(T, 1), "T must be a multiple of dt"
    u, p, sigma, gamma, f = exact(case, mu, rho, U, P)
    bc =FITTED[case].get("dirichlet", ".*")
    gN = sigma * specialcf.normal(2) if case == "mixed" else None
    factor = superlu.inverse
    V = VectorL2(mesh, order=order)
    un, an = GridFunction(V), GridFunction(V)        # a^0 = 0: not read (Euler first step)
    t.Set(0.0)
    un.Set(u)
    beta, iters, failed = CF((0, 0)), [], False
    for n in range(nsteps):
        theta = 1.0 if scheme == "euler" or n == 0 else 0.5
        t.Set((n + 1) * dt)
        c = rho / (theta * dt)
        w = c * un + (rho * (1 - theta) / theta) * an

        def step(b):
            return oseen.solve(mesh, order, mu, f, u, scaled(b, rho), gN, bc, "global", factor,
                               inertia=(c, w))
        sol, p_h, m_h, S, us, _, it, inc = picard(step, mesh, order, mu, beta=beta, factor=factor)
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
        beta = us
    return mesh, (u, p, sigma, gamma), sol, p_h, m_h, S, us, iters, failed


COLUMNS = st.COLUMNS + ("|u-u**|",)


def row(case, order, h, dt, T, mu=1.0, rho=1.0, scheme="cn", verbose=False):
    """Errors at T (COLUMNS) and the extras of one run."""
    log = (lambda n, it, inc: print(f"    step {n:4d}  Picard {it:2d}  increment {inc:9.2e}",
                                    flush=True)) if verbose else None
    mesh, (u, p, sigma, gamma), sol, p_h, m_h, S, us, iters, failed = solve(
        case, order, h, dt, T, mu, rho, scheme, log=log)
    e = st.errors(mesh, sol, p_h, u, p, sigma, gamma)
    eus = sqrt(Integrate(InnerProduct(u - us, u - us) * dx(bonus_intorder=4), mesh))
    return mesh, sol, (*e, eus), dict(m_h=m_h, S=S, max_iter=max(iters),
                                      mean_iter=float(np.mean(iters)), steps=len(iters),
                                      failed=failed)


def study(case, order, hs, dts, T, mu=1.0, rho=1.0, scheme="cn", record=None):
    """EOC table over the pairs (h, dt): rates in h, or in dt when h is fixed."""
    var = "dt" if len(set(hs)) == 1 else "h"
    title = (f"{case}, k = {order}, mu = {mu}, rho = {rho}, {scheme}"
             f"{' (Euler start)' if scheme == 'cn' else ''}, T = {T}, Picard tol {TOL:g}, "
             f"cap {MAXIT}, rates in {var}")
    print(title, flush=True)
    print(f"  {'h':>7}  {'dt':>9}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS)
          + f"  {'m_h':>9}  {'S':>8}  {'max it':>6}  {'mean it':>7}  {'steps':>5}")
    prev = None
    for h, dt in zip(hs, dts):
        mesh, sol, e, ex = row(case, order, h, dt, T, mu, rho, scheme)
        s = h if var == "h" else dt
        line = f"  {h:7.4f}  {dt:9.3e}"
        for i, ei in enumerate(e):
            r = (f"{np.log(prev[1][i] / ei) / np.log(prev[0] / s):4.2f}"
                 if prev and ei > 0 and not ex["failed"] else "  - ")
            line += f"  {ei:9.3e}  {r}"
        m = "-" if ex["m_h"] is None else f"{ex['m_h']:.2e}"
        print(line + f"  {m:>9}  {ex['S']:8.4f}  {ex['max_iter']:6d}  {ex['mean_iter']:7.2f}"
              f"  {ex['steps']:5d}" + ("  FAILED" if ex["failed"] else ""), flush=True)
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
    a = parse(p)
    if a.k is None:
        p.error("--k and --h are required")
    dts = a.dt * len(a.h) if len(a.dt) == 1 else a.dt
    hs = a.h * len(dts) if len(a.h) == 1 else a.h
    if len(hs) != len(dts):
        p.error("--h and --dt: equal lengths, or one of them a single value")
    record = Record(COLUMNS)
    study(a.case, a.k, hs, dts, a.T, a.mu, a.rho, a.scheme, record)
    if a.out:
        save(a.out, record, dict(driver="navier_stokes_transient", case=a.case, k=a.k, mu=a.mu,
                                 rho=a.rho, scheme=a.scheme, T=a.T, tol=TOL, maxit=MAXIT,
                                 beta="u**", solver="superlu", u="U cos t"))

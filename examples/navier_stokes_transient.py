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

Schemes: "euler" (theta = 1); "cn" (theta = 1/2 after one Euler step of size EPS dt: the
steps end at EPS dt, dt, 2 dt, ..., T, the first CN step of size (1 - EPS) dt; the theta
update of a holds for any step size); "cn-euler" (CN after one Euler step of size dt, as
fsi_transient.py and MATLAB's hemodynamics.m: a control). a^0 is never read.
Why EPS: the Euler step gives a^1 = (u^1 - u^0)/tau satisfying the discrete momentum row of
its own step exactly, so CN starts from a consistent (u, a) (CN from an a^0 that does not
satisfy it carries the defect undamped in sigma and p), with an O(tau) error in the pressure
of that step: tau = dt (cn-euler) makes sigma and p first order in dt, tau = EPS dt shrinks
that to O(EPS dt) at the cost of round-off of order eps_machine/(EPS dt) in a^1. Pure time
error (u = (x + 2 y, 3 x - y) cos t, p = 0, k = 1, exact in space; T = 0.4, dt = 0.1 -> 0.0125):
|s - s_h| cn-euler 4.0e-2 -> 4.9e-3 (rate 1), EPS = 1e-3: 2.2e-4 -> 9.0e-7 (rates 2.3-2.8),
EPS = 1e-5: round-off takes over at dt = 0.0125 (3.3e-4). EPS = 1e-3 is the default. This
replaces the acceleration problem of MATLAB's initial_state (decision of Jaime, 2026-11-24:
its a^0 is another discretization, not consistent with the step, and on TPM converges at rate
k only).

Cases (manufactured): those of navier_stokes_stress.py with u = U cos t, p = P cos t (U, P
the steady fields of each case), f = rho u_t + rho (u . grad) u - div sigma. Fitted:
"dirichlet" and "mixed" (traction on right|top, U = U + (1, 1), u . n = cos t > 0 there).
Unfitted (Transfer Path Method, oseen_stress_tpm.py): Dirichlet on Gamma, Examples 1, 2, 3 of
elasticity_tpm.py and paper example 4, beta = corrected_beta(u**) as in the steady loop;
traction on Gamma ("traction", "traction kidney"), u . n_Gamma > 0 kept for t < pi/2 (the sinks
scale with cos t), beta = u** uncorrected. Stabilisation: navier_stokes_stress.default_stab
(pointwise S for the traction cases, global otherwise).

Initial velocity (start): "elliptic" (default), u^0 = u_h of the steady Navier-Stokes solve
of the case at t = 0 (the Picard loop with f = f(0) - rho u_t(0) = -div sigma(0) + rho (u . grad) u(0),
the boundary data at t = 0), and the first beta of step 1 its u** (corrected on the Dirichlet
TPM cases); "l2", MATLAB's L2 projection of u(0) on the elements (initial_state.m:50-51), the
first beta zero. The first step of size tau divides by tau any defect of u^0 against the
discrete constraints of the step (div, the transferred trace on Gamma_h, and S, beta in the
local solver), and CN carries it in sigma and p. The L2 projection has such a defect
(Example 2, k = 1, h = 0.2, dt = 0.0025, T = 0.08: |s - s_h| 2.07 with cn-euler, 2.2e+3 with
cn; dt = h^2, T = 0.32, h = 0.2 -> 0.1, cn-euler: 0.30 -> 0.19, Euler 0.33 -> 0.070); so has
the Stokes projection (beta = 0, S = mu) against the Oseen step (cn-euler 0.369, cn 103). The
steady Navier-Stokes solve shares the operator of the first step up to O(tau): cn 0.347,
cn-euler 0.347, Euler 0.347; dt = h^2, h = 0.2 -> 0.1: 0.33 -> 0.071 (rate 2.23, as the steady
loop); traction: rates 1.68, 1.54, 2.02 for sigma, p, u (steady: the same).

Stability on the Dirichlet TPM cases with paths O(h) (Examples 2, 3, paper example 4): the
steady loop does not see it, but the semi-discrete operator (the step without the mass) can
have eigenvalues of positive real part, so the error grows from round-off. Measured with Euler
at dt = 0.0005 (backward Euler damps a real eigenvalue lambda > 0 only when dt lambda > 2):
- whole stress transferred (bubbles included, Cardenas-Solano (2.2g), MATLAB): paper example 4,
  k = 1, mu = 1, h = 0.1 grows about 16 times per step (lambda about 2e3); Examples 2, 3 at
  mu = 0.1 likewise (lambda about 1e3, proportional to mu/h^2). Not the multiplier m (pinned
  pressure: the same), not tau (0.05 to 20: the same); the traction cases do not grow.
- P_k part transferred: all of Examples 2, 3 and paper example 4 at k = 1, 2, mu = 1, 0.1 are
  stable but paper example 4, k = 2, mu = 1 (about 1.6 per step, lambda about 750).
CN amplifies every eigenvalue of positive real part at any dt; bdf2 (one Euler step, then
(3 u^{n+1} - 4 u^n + u^{n-1})/(2 dt), second order: rates 1.85-2.5 on the pure time error
above) is stable outside a bounded region of real part in (0, 4/dt). Hence the defaults there:
P_k part and bdf2 (default_full, default_scheme). The tables of dt = h^2, T = 0.2, h = 0.2 ->
0.1 then run in all twelve configurations (k = 1, 2, mu = 1, 0.1) but one: paper example 4,
k = 2, mu = 0.1 (h = 0.1 and 0.05; Picard at its cap after about 8 steps, with Picard
converged before: a growth of about 6.5 per step at dt = 0.01, absent from the Stokes
problem, so it comes with the convection). Pruning D_h (--prune, elasticity_tpm.prune)
removes it there (h = 0.2 -> 0.1 -> 0.05: |s - s_h - cI| 1.8e-4 -> 1.9e-5 -> 2.2e-6, rates
3.26, 3.12), so it is the default there (default_prune), but it is not a remedy in general: pruned, the kidney at h = 0.2 has paths that miss Gamma (transfer_paths
raises), and Example 2, k = 2, mu = 1 (whole stress, cn) fails at h = 0.1.

Differences from MATLAB: the default start and the EPS step of "cn" (above; MATLAB also projects
uhat^0, read by nothing in a step, and solves the acceleration problem for a^0); on the Dirichlet
TPM cases with paths O(h), the P_k part of the stress transferred and bdf2 (above; MATLAB: the
whole stress, the theta-scheme only); the stopping norm as in navier_stokes_stress.py.

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
from elasticity_tpm import EXAMPLES, at, domain, prune as prune_domain
from navier_stokes_stress import MAXIT, TOL, picard
from ngstpm import restrict, unfitted_domain
from poisson_tpm import background_mesh

t = Parameter(0.0)                                    # the time of the data
SCHEMES = ("euler", "cn", "cn-euler", "bdf2")
EPS = 1e-3                                            # the first step of "cn", in units of dt
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
    return u, p, sigma, gamma, f + rho * (ut + gradient(u) * u)


def gradient(u):
    return CF((u[0].Diff(x), u[0].Diff(y), u[1].Diff(x), u[1].Diff(y)), dims=(2, 2))


def scaled(beta, rho):
    """rho beta, a GridFunction if beta is one (oseen.edge_points reads the type)."""
    if rho == 1:
        return beta
    if isinstance(beta, GridFunction):
        b = GridFunction(beta.space)
        b.vec.data = rho * beta.vec
        return b
    return rho * beta


def tpm_dirichlet_oh(case):
    """A Dirichlet TPM case with paths O(h) (Examples 2, 3, paper example 4)."""
    return ns.kind(case) == "dirichlet" and case != ns.UNFITTED[0]


def default_prune(case):
    """D_h pruned (elasticity_tpm.prune) on paper example 4 only (docstring: stability)."""
    return case == ns.PAPER4


def default_full(case):
    """The stress transferred on Gamma_h: the whole stress (P_k part and bubbles,
    Cardenas-Solano 2024 (2.2g)), except on the Dirichlet TPM cases with paths O(h), where its
    P_k part (docstring: stability)."""
    return not tpm_dirichlet_oh(case)


def default_scheme(case):
    """bdf2 on the Dirichlet TPM cases with paths O(h), cn otherwise (docstring: stability)."""
    return "bdf2" if tpm_dirichlet_oh(case) else "cn"


def setup(case, order, h, mu, stab, fields, mesh=None, factor=None, prune=False, full=True):
    """mesh, D (None if fitted), phi (None if fitted) and solver(beta, inertia, f) -> (sol,
    p_h, m_h, S): one Oseen solve of the case, as navier_stokes_stress.solve (f: default the
    source of fields); prune: D_h pruned (elasticity_tpm.prune); full: the stress transferred
    on Gamma_h (stokes_stress_tpm.solve); both for the Dirichlet TPM cases only."""
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
        D = prune_domain(mesh, phi, D) if prune else D
        args = dict(D=D, dirichlet=".*", stab=stab, full=full)
    else:
        phi = EXAMPLES[case]
        mesh, D, bc, _ = domain(case, h, phi, prune)
        args = dict(D=D, dirichlet=bc, stab=stab, full=full)
    g = u if case in ns.TRACTION else at(u)

    def solver(beta, inertia, f=f):
        sol, p_h, m_h, _, S = otpm.solve(mesh, order, mu, f, g, phi, beta, factor=factor,
                                         inertia=inertia, **args)
        return sol, p_h, m_h, S
    return mesh, D, phi, solver


def solve(case, order, h, dt, T, mu=1.0, rho=1.0, scheme=None, mesh=None, log=None, U=None,
          P=None, stab=None, start="elliptic", prune=None, full=None):
    """The run to t = T. Returns mesh, D (None if fitted), the exact (u, p, sigma, gamma) at T
    (t is left at T), sol, p_h, m_h, S, u** of the last step, and the Picard counts per step;
    failed is True if a step did not converge (the run stops there). mesh: fitted cases only,
    default the unit square of size h; U, P: as in exact; stab: default
    navier_stokes_stress.default_stab(case); start: the initial velocity, "elliptic" (the
    steady Navier-Stokes solve at t = 0) or "l2" (MATLAB's L2 projection); prune, full: as in
    setup (defaults default_prune(case), default_full(case)); scheme: default
    default_scheme(case)."""
    nsteps = round(T / dt)
    assert abs(nsteps * dt - T) < 1e-12 * max(T, 1), "T must be a multiple of dt"
    stab = ns.default_stab(case) if stab is None else stab
    fields = exact(case, mu, rho, U, P)
    u, p, sigma, gamma, _ = fields
    factor = superlu.inverse
    full = default_full(case) if full is None else full
    prune = default_prune(case) if prune is None else prune
    scheme = default_scheme(case) if scheme is None else scheme
    mesh, D, phi, solver = setup(case, order, h, mu, stab, fields, mesh, factor, prune, full)
    correct = D is not None and case not in ns.TRACTION
    V = VectorL2(mesh, order=order)
    V = V if D is None else restrict(V, D)            # the velocity space of the solves
    un, an, uold = GridFunction(V), GridFunction(V), GridFunction(V)   # a^0 = 0: not read
    t.Set(0.0)
    un.Set(u, definedonelements=D)
    beta, iters, failed = CF((0, 0)), [], False
    if start == "elliptic":   # steady Navier-Stokes at t = 0 with f = f(0) - rho u_t(0): the operator of step 1
        f0 = st.manufactured(u, p, mu)[2] + rho * (gradient(u) * u)
        out = picard(lambda b: solver(scaled(b, rho), None, f0), mesh, order, mu, D, correct,
                     factor=factor)
        assert out[-1] < TOL, "the steady solve of the initial velocity did not converge"
        un.vec.data = out[0][1].vec
        beta = corrected_beta(mesh, out[4], out[2], D) if correct else out[4]
    times = [(n + 1) * dt for n in range(nsteps)]
    if scheme == "cn":
        times = [EPS * dt] + times
    for n, tn in enumerate(times):
        theta = 1.0 if scheme == "euler" or n == 0 else 0.5
        tau = tn - (times[n - 1] if n else 0.0)
        t.Set(tn)
        if scheme == "bdf2" and n > 0:        # (3 u^{n+1} - 4 u^n + u^{n-1})/(2 dt)
            c, w = 3 * rho / (2 * tau), (rho / tau) * (2 * un - 0.5 * uold)
        else:
            c = rho / (theta * tau)
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
        an.vec.data = (1 / (theta * tau)) * (u1 - un.vec) - ((1 - theta) / theta) * a_old
        uold.vec.data = un.vec                 # u^{n-1} of bdf2
        un.vec.data = u1
        beta = corrected_beta(mesh, us, m_h, D) if correct else us
    return mesh, D, (u, p, sigma, gamma), sol, p_h, m_h, S, us, iters, failed


def row(case, order, h, dt, T, mu=1.0, rho=1.0, scheme=None, stab=None, verbose=False,
        start="elliptic", prune=None, full=None):
    """Errors at T (COLUMNS[ns.kind(case)]) and the extras of one run."""
    log = (lambda n, it, inc: print(f"    step {n:4d}  Picard {it:2d}  increment {inc:9.2e}",
                                    flush=True)) if verbose else None
    mesh, D, (u, p, sigma, gamma), sol, p_h, m_h, S, us, iters, failed = solve(
        case, order, h, dt, T, mu, rho, scheme, log=log, stab=stab, start=start, prune=prune,
        full=full)
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


def study(case, order, hs, dts, T, mu=1.0, rho=1.0, scheme=None, stab=None, record=None,
          start="elliptic", prune=None, full=None):
    """EOC table over the pairs (h, dt): rates in h, or in dt when h is fixed."""
    stab = ns.default_stab(case) if stab is None else stab
    full = default_full(case) if full is None else full
    prune = default_prune(case) if prune is None else prune
    scheme = default_scheme(case) if scheme is None else scheme
    var = "dt" if len(set(hs)) == 1 else "h"
    title = (f"{case}, k = {order}, mu = {mu}, rho = {rho}, {scheme}"
             f"{f' (Euler start {EPS:g} dt)' if scheme == 'cn' else ''}, u^0 {start}, T = {T}, {ns.STAB[stab]}, "
             f"Picard tol {TOL:g}, cap {MAXIT}, rates in {var}"
             + (", D_h pruned" if prune else "")
             + ("" if ns.kind(case) != "dirichlet" or full else ", P_k part of sigma transferred"))
    print(title, flush=True)
    print(f"  {'h':>7}  {'dt':>9}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS[ns.kind(case)])
          + f"  {'m_h':>9}  {'c':>9}  {'S':>8}  {'max it':>6}  {'mean it':>7}  {'steps':>5}"
          f"  {'#D_h':>5}")
    prev = None
    for h, dt in zip(hs, dts):
        mesh, sol, e, ex = row(case, order, h, dt, T, mu, rho, scheme, stab, start=start, prune=prune,
                               full=full)
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


# the default tables (one process per case, k and mu; dt = h^2, T = 0.2): h down to 0.05 on
# the fitted meshes and Example 1 (gap O(h^2)), to 0.1 on the cases with gap O(h)
DEFAULTS = {**{(k, c): (0.2, 0.1, 0.05) for k in (1, 2) for c in (*ns.FITTED, ns.UNFITTED[0])},
            **{(k, c): (0.2, 0.1) for k in (1, 2) for c in ns.UNFITTED[1:]}}


if __name__ == "__main__":
    import argparse
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, CASES)
    p.add_argument("--dt", type=float, nargs="+",
                   help="time steps, one per --h (or one for all); default h^2")
    p.add_argument("--T", type=float, default=1.0, help="final time (default 1)")
    p.add_argument("--scheme", choices=SCHEMES, default=None,
                   help="default: bdf2 on the Dirichlet TPM cases with paths O(h), cn (Euler start of "
                        "EPS dt) otherwise")
    p.add_argument("--mu", type=float, default=1.0, help="viscosity (default 1)")
    p.add_argument("--rho", type=float, default=1.0, help="density (default 1)")
    p.add_argument("--start", choices=("elliptic", "l2"), default="elliptic",
                   help="initial velocity: steady Navier-Stokes solve at t = 0 (default) or L2 projection (MATLAB)")
    p.add_argument("--prune", action=argparse.BooleanOptionalAction, default=None,
                   help="prune D_h (Dirichlet TPM cases; default: on paper example 4 only)")
    p.add_argument("--transfer", choices=("full", "pk"), default=None,
                   help="stress transferred on Gamma_h (Dirichlet TPM): whole, or P_k part "
                        "(default: pk on the cases with paths O(h), full otherwise)")
    p.add_argument("--stab", choices=("global", "pointwise"), default=None,
                   help="default: navier_stokes_stress.default_stab(case)")
    a = parse(p)
    if a.k is None:
        run_defaults(__file__, [["--case", c, "--k", k, "--h", *hs, "--T", 0.2, "--mu", mu]
                                for (k, c), hs in DEFAULTS.items() for mu in (1.0, 0.1)])
        raise SystemExit
    dts = [h * h for h in a.h] if a.dt is None else a.dt
    dts = dts * len(a.h) if len(dts) == 1 else dts
    hs = a.h * len(dts) if len(a.h) == 1 else a.h
    if len(hs) != len(dts):
        p.error("--h and --dt: equal lengths, or one of them a single value")
    stab = ns.default_stab(a.case) if a.stab is None else a.stab
    record = Record(COLUMNS[ns.kind(a.case)])
    full = default_full(a.case) if a.transfer is None else a.transfer == "full"
    prune = default_prune(a.case) if a.prune is None else a.prune
    scheme = default_scheme(a.case) if a.scheme is None else a.scheme
    study(a.case, a.k, hs, dts, a.T, a.mu, a.rho, scheme, stab, record, a.start, prune, full)
    if a.out:
        save(a.out, record, dict(driver="navier_stokes_transient", case=a.case, k=a.k, mu=a.mu,
                                 rho=a.rho, scheme=scheme, start=a.start, prune=prune, full=full, T=a.T, stab=stab, tol=TOL,
                                 maxit=MAXIT, beta="u**", solver="superlu", u="U cos t"))

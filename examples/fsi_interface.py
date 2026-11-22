"""Fluid-structure interface problem (one frozen step, fixed interface, steady, no inertia
terms), HDG with transfer paths: Stokes in stress form (stokes_stress.py) in the fluid,
linear elasticity (elasticity.py) in the solid,

    A_F sigma_F - grad u_F + S(gamma_F) = 0,   -div sigma_F = f_F   in Omega_F,
    A_S sigma_S - grad eta + S(gamma_S) = 0,   -div sigma_S = f_S   in Omega_S,
    u_F - alpha_S eta = g,   sigma_F n_F + sigma_S n_S = g_nu        on Gamma = {phi = 0},
    u_F = g_F on the boundary of the box,

Omega_S = {phi < 0} (unit disk), Omega_F the rest of the box (-1.5, 1.5)^2, n_F = -n_S,
A_F sigma = dev(sigma)/(2 mu_F), p_F = -tr(sigma_F)/2, A_S the plane-strain compliance of
(lam_S, mu_S). alpha_S = 1/dt with dt = 1e-3 (alpha_S = 1000): the steady kinematic row
< dt u_F - u_S, mu_2 > of MATLAB's InterfaceProblem.fsi (scheme "steady") divided by dt,
with dt = 1e-3 as in examples/tpm/FSI_closed_convergence.m.

Each side has the unknowns and rows of its example on its element set D_h^F, D_h^S
(families fitted / snapped of elasticity_interface.py; the fluid with the pressure split,
rho_F the element means of tr sigma_F), tau_F = tau mu_F, tau_S = tau mu_S. Side 1 of
pair is the FLUID (pair(p_F, p_S)); nubar = p_F.target_normal is the normal of Gamma from
{phi < 0} to {phi > 0}, i.e. nubar = n_S = -n_F. Rows at the ends xbar of the paths (MATLAB
sides: T1 on the fluid faces, T2 on the solid faces):

  Gamma_h^F (equation of uhat_F), traction, P_k parts of the stresses:
      [sP_F(xbar) n_F - tau mu_F (u_F - uhat_F)] + [sP_S(xbar) n_S - tau mu_S (eta - etahat)(x')]
          = g_nu(xbar);
  Gamma_h^S (equation of etahat), kinematic, transferred along both paths with the whole
  stresses (full=False: P_k parts):
      uhat_F.Transfer(p_F, gradient=A_F sigma_F + S(gamma_F))
          - alpha_S etahat.Transfer(p_S, gradient=A_S sigma_S + S(gamma_S)) = g(xbar).

Pressure level (rule for the pressure mean of stokes_stress.py): the fluid velocity is
prescribed on the box but not on Gamma, so the mode p_F + c is fixed by the coupling: the
fluid mass balance fixes the flux of u_F through Gamma, the kinematic row turns it into the
volume change of the solid, its bulk modulus into its mean stress, and the traction row
passes that to the fluid. Hence no zero-mean constraint, no multiplier, no extra scalar;
sigma_F and p_F are compared raw. The level carries the error of the discrete volume change
of the solid times its bulk modulus; c = mean(p_F - p_h) does not depend on alpha_S. The
fluid volume budget is closed (Dirichlet on the whole box fixes the net flux of u_F through
Gamma_h^F), so the net-flux defect eps of the transfer can only go into a mean volume change of
the solid, which costs the pressure (lam_S + mu_S) eps/|Omega_S,h| (reports/tpm-flux-balancing.typ,
Remark 3; unfitted-hdg/experiments/analysis-paper/LAMBDA_CONDITIONING.md). For nu_S = 0.3, c is
O(h^(k+2)), below the other errors; for the nearly incompressible solid (lam_S ~ 1.7e7) it is
O(lam_S h^(k+3)) and dominates sigma_F, p_F and sigma_S on all meshes that can be run, while the
velocity and displacement errors are unaffected. This is accepted as a limitation, like the
constant c I of the transferred Dirichlet problem of elasticity_tpm.py: with modulo=True the
tables also report sigma_F, p_F and sigma_S modulo the common shift c I (the same c on both
sides: the traction row passes the shift unchanged). Flux balancing (tpm-flux-balancing.typ,
Section 4) is an untried constraint-free remedy; the solid pressure split (spaces(split=True),
MATLAB's factored condensation) is an exact reformulation that only fixes the conditioning. The rule fixes
the level only with strength 1/lam_S and degenerates as nu_S -> 1/2: at nu_S = 0.49999999 the
condensed matrix has condition number about 4.5e14 already at h = 0.3, k = 1 (sigma_max ~ lam_S
from the unsplit solid, sigma_min ~ 1/lam_S for the level mode), against about 9e4 at nu_S = 0.3.
MATLAB's FSI_closed_convergence.m does the same: it uses the default pressure_mean = "pinned" of
FluidStructureCoupling.load_problems (load_problems.m:101-102), which eliminates the
pressure-level scalar alpha (@InterfaceProblem/solve.m:205-208) and makes the zero-mean
constraint a no-op (ZeroMeanPressure.m:4), so the level is fixed by the coupling there too.
"constrained" with the scalar alpha (a sliver-weighted mean row, T1) is only the default of
InterfaceProblem.fsi itself (InterfaceProblem.m:239-240).

The unfitted family is not implemented: the unfitted stress-form Stokes interface does not
converge regularly (stokes_stress_interface.py), a documented limitation.
"""
from math import log2

from ngsolve import *

import elasticity as el
import stokes_stress as st
from elasticity import compliance, free_dofs, hdg_rows, lame, postprocess, skew, solve_linear_system, spaces, stress
from elasticity_interface import domains
from elasticity_tpm import at
from ngstpm import interior_ds, pair, transfer_paths

INF = float("inf")
ALPHA = 1e3                                                           # alpha_S = 1/dt, dt = 1e-3


def solve(mesh, order, phi, DF, DS, muF, lamS, muS, fF, fS, g, gS, gF, alpha=ALPHA, tau=1.0,
          kind=None, full=True):
    """g(X, Y): kinematic data at xbar (vector CF); gS(X, Y): the stress jump
    sigma_S - sigma_F at xbar (2x2 CF), g_nu = gS(xbar) nubar; gF: u_F on the
    box (CF). Returns the fluid (sigma_h, u_h, gamma_h, uhat_h, ustar_h), p_h and the solid
    (sigma_h, eta_h, gamma_h, etahat_h, etastar_h)."""
    pF, pS = (transfer_paths(mesh, phi, D, kind=kind, order=order + 4) for D in (DF, DS))
    I = pair(pF, pS)                                                  # side 1: fluid, side 2: solid
    dF, dS = (I.ds(side=i, bonus_intorder=max(6, 2 * order)) for i in (1, 2))

    spF, spS = spaces(mesh, order, "outer", DF, split=True), spaces(mesh, order, "outer", DS)
    fes = spF[0] * spS[0]
    trial, test = fes.TnT()
    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    for tr, te, lam, mu, f, D in ((trial[:8], test[:8], INF, muF, fF, DF),
                                  (trial[8:], test[8:], lamS, muS, fS, DS)):
        sigmahat_n = hdg_rows(lhs, rhs, tr, te, lam, mu, f, tau, D)
        lhs += sigmahat_n * te[5] * interior_ds(mesh, D)
    (sPF, wF1, wF2, uF, gamF, uhatF), (sPS, wS1, wS2, eta, gamS, etahat) = trial[:6], trial[8:]

    # Gamma_h^F, traction:  sigma_F n_F + sigma_S n_S = g_nu  at xbar, n_S = nubar = -n_F
    xbar, nubar = pF.target, pF.target_normal
    tF = -sPF.Extrapolate(xbar) * nubar - tau * muF * (uF - uhatF)
    tS = sPS.Extrapolate(xbar) * nubar - tau * muS * (eta.Extrapolate() - etahat.Extrapolate())
    lhs += (tF + tS) * test[5] * dF
    rhs += gS(*xbar) * nubar * test[5] * dF

    # Gamma_h^S, kinematic:  u_F - alpha_S eta = g  at xbar
    xbar = pS.target
    sF = stress(sPF, wF1, wF2) if full else sPF
    sS = stress(sPS, wS1, wS2) if full else sPS
    ubF = uhatF.Transfer(pF, gradient=compliance(sF, INF, muF) + skew(gamF))
    ubS = etahat.Transfer(pS, gradient=compliance(sS, lamS, muS) + skew(gamS))
    lhs += (ubF - alpha * ubS) * test[13] * dS
    rhs += g(*xbar) * test[13] * dS

    gfu = GridFunction(fes)
    gfu.components[5].Set(gF, BND, definedon=mesh.Boundaries("outer"))
    free = free_dofs(fes, spF[1], True)
    free = free_dofs(fes, spS[1], True, free=free, first=8)
    solve_linear_system(lhs, rhs, gfu, free)
    c = gfu.components
    c[3].gfu = gfu                                      # the whole solution, for study_io
    sigF, sigS = stress(*c[:3]), stress(*c[8:11])
    fluid = (sigF, c[3], c[4], c[5], postprocess(mesh, order, INF, muF, sigF, c[3], c[4], DF))
    solid = (sigS, c[11], c[12], c[13], postprocess(mesh, order, lamS, muS, sigS, c[11], c[12], DS))
    return fluid, -Trace(sigF) / 2, solid


PHI = lambda X, Y: sqrt(X * X + Y * Y) - 1                            # solid: unit disk in (-1.5, 1.5)^2
MU_F, E_S, NU_S, NU_INC = 1.0, 1.0, 0.3, 0.49999999
PSI_F = sin(x) * exp(y / 2) + x * x * y / 3
U_F = CF((PSI_F.Diff(y), -PSI_F.Diff(x)))                             # curl psi_F
P_F = x * y + cos(x)                                                  # nonzero mean on D_h^F
ETA = CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y)))


def problem(uF=U_F, pF=P_F, eta=None, muF=MU_F, E=E_S, nu=NU_S, alpha=ALPHA):
    """lam_S, mu_S, the exact (sigma, gamma, f) of both sides, g and the stress jump
    sigma_S - sigma_F (callables at xbar; g_nu = (sigma_S - sigma_F) n_S), u_F on the box
    and eta. eta = None: ETA, or the bounded data elasticity.bounded_data(lam_S)
    if nu = NU_INC."""
    lamS, muS = lame(E, nu)
    if eta is None:
        eta = el.bounded_data(lamS) if nu == NU_INC else ETA
    exF, exS = st.manufactured(uF, pF, muF), el.manufactured(eta, lamS, muS)
    return lamS, muS, exF, exS, at(uF - alpha * eta), at(exS[0] - exF[0]), uF, eta


def side_errors(mesh, fluid, p_h, solid, DF, DS, uF, pF, exF, eta, exS):
    """Fluid: |sigma - sigma_h|, |p - p_h|, |u - u_h|, |gamma - gamma_h|, |P_M u - uhat|_h,
    |u - u*| (raw: no shift of the pressure); solid: |sigma - sigma_h|, |eta - eta_h|,
    |gamma - gamma_h|, |P_M eta - etahat|_h, |eta - eta*|; and the mean over D_h^F of p - p_h."""
    eF = st.errors(mesh, fluid, p_h, uF, pF, exF[0], exF[1], DF)[:6]
    eS = el.errors(mesh, solid, eta, exS[0], exS[1], DS)
    dV = dx(definedonelements=DF, bonus_intorder=4)
    mean = Integrate((pF - p_h) * dV, mesh) / Integrate(CF(1) * dV, mesh)
    return list(eF) + [eS[j] for j in (0, 1, 2, 4, 5)], mean


def modulo_errors(mesh, fluid, p_h, solid, DF, DS, pF, exF, exS, c):
    """|sigma_F - sigma_h - c I|, |p - p_h - c|, |sigma_S - sigma_h - c I|, c = mean over D_h^F
    of p - p_h (p_h = p - c means sigma_h = sigma + c I)."""
    l2 = lambda a, D: sqrt(Integrate(InnerProduct(a, a) * dx(definedonelements=D, bonus_intorder=4), mesh))
    return [l2(fluid[0] - exF[0] - c * Id(2), DF), l2(pF - p_h - c, DF), l2(solid[0] - exS[0] - c * Id(2), DS)]


COLUMNS = ("F|s-sh|", "F|p-ph|", "F|u-uh|", "F|g-gh|", "F|Pu-uh^|", "F|u-u*|",
           "S|s-sh|", "S|e-eh|", "S|g-gh|", "S|Pe-eh^|", "S|e-e*|")
MODULO = ("F|s-sh-cI|", "F|p-ph-c|", "S|s-sh-cI|")


def study(family, order, hs, nu=NU_S, alpha=ALPHA, full=True, modulo=False, record=None):
    """EOC table, the two sides in separate columns (with modulo: also the three columns of
    modulo_errors); record: a study_io.Record with the columns COLUMNS (+ MODULO)."""
    lamS, muS, exF, exS, g, gS, uF, eta = problem(nu=nu, alpha=alpha)
    print(f"{family}, k = {order}, mu_F = {MU_F}, E_S = {E_S}, nu_S = {nu}, alpha_S = {alpha:g}"
          + ("" if full else ", P_k part of sigma in the kinematic row"))
    columns = COLUMNS + (MODULO if modulo else ())
    print(f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in columns) + f"  {'mean p-ph':>10}  #DF/#DS")
    prev, rows = None, []
    for h in hs:
        mesh, DS, DF, _ = domains(family, h, PHI)
        fluid, p_h, solid = solve(mesh, order, PHI, DF, DS, MU_F, lamS, muS, exF[2], exS[2], g, gS, uF,
                                  alpha=alpha, full=full)
        e, mean = side_errors(mesh, fluid, p_h, solid, DF, DS, uF, P_F, exF, eta, exS)
        if modulo:
            e += modulo_errors(mesh, fluid, p_h, solid, DF, DS, P_F, exF, exS, mean)
        line = f"  {h:7.4f}"
        for j, ej in enumerate(e):
            r = f"{log2(prev[1][j] / ej) / log2(prev[0] / h):4.2f}" if prev else "  - "
            line += f"  {ej:9.3e}  {r}"
        print(line + f"  {mean:10.2e}  {DF.NumSet()}/{DS.NumSet()}")
        if record is not None:
            record.add(h, mesh, e, fluid[1].gfu, mean=mean, nDF=DF.NumSet(), nDS=DS.NumSet())
        prev = (h, e)
        rows.append((e, mean))
    return rows


FAMILIES = ("fitted", "snapped")


if __name__ == "__main__":
    import argparse
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(FAMILIES))
    p.add_argument("--alpha", type=float, default=ALPHA, help=f"alpha_S (default {ALPHA:g} = 1/dt)")
    p.add_argument("--nearly-incompressible", action="store_true",
                   help=f"nu_S = {NU_INC} with the bounded data of elasticity.bounded_data")
    p.add_argument("--modulo", action=argparse.BooleanOptionalAction, default=None,
                   help="also the errors of sigma_F, p_F, sigma_S modulo c I, c = mean of p - p_h "
                        "(default: on with --nearly-incompressible)")
    p.add_argument("--pk", dest="full", action="store_false",
                   help="P_k part of sigma in the kinematic row (default: the whole stress)")
    a = parse(p)
    if a.k is None:
        # one process per case, within 2 GB each. Measured peaks (fitted): k = 1 on 0.2, 0.1, 0.05:
        # 0.82 GB (nu_S = 0.3), 1.09 GB (nearly incompressible); k = 2 on 0.2, 0.1, 0.07:
        # 1.05 GB, 1.35 GB. k = 2 at h = 0.05 is estimated at 2.3 GB (0.19, 0.53 GB at h = 0.2, 0.1).
        # Snapped (nu_S = 0.3): 0.92 GB (k = 1), 1.21 GB (k = 2).
        levels = ((1, (0.2, 0.1, 0.05)), (2, (0.2, 0.1, 0.07)))
        run_defaults(__file__, [["--case", "fitted", "--k", k, "--h", *hs, *flag]
                                for flag in ([], ["--nearly-incompressible"]) for k, hs in levels]
                     + [["--case", "snapped", "--k", k, "--h", *hs] for k, hs in levels])
    else:
        nu = NU_INC if a.nearly_incompressible else NU_S
        modulo = a.nearly_incompressible if a.modulo is None else a.modulo
        record = Record(COLUMNS + (MODULO if modulo else ()))
        study(a.case, a.k, a.h, nu=nu, alpha=a.alpha, full=a.full, modulo=modulo, record=record)
        if a.out:
            save(a.out, record, dict(driver="fsi_interface", case=a.case, k=a.k, mu_F=MU_F, E_S=E_S,
                                     nu_S=nu, alpha_S=a.alpha, tau=1.0, full=a.full, modulo=modulo))

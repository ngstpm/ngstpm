"""Stokes interface problem in stress form, two viscosities, HDG with transfer paths.

The HDG of stokes_stress.py on each side, with the Transfer Path Method:

    A_i sigma_i - grad u_i + S(gamma_i) = 0,   -div sigma_i = f_i   in Omega_i,
    u_1 - u_2 = gD,   (sigma_1 - sigma_2) nu = gN nu   on Gamma = {phi = 0},
    u_2 = g on the boundary of the box,

Omega_1 = {phi < 0} (unit disk), Omega_2 the rest of the box (-1.5, 1.5)^2,
A_i sigma = dev(sigma)/(2 mu_i), p_i = -tr(sigma_i)/2, nu from Omega_1 to Omega_2.

Each side has the unknowns and rows of stokes_stress.py on its element set D_h^i
(families fitted / snapped / unfitted as in elasticity_interface.py), tau_F,i = tau mu_i,
and the transmission conditions are imposed at the ends xbar of the paths:

  Gamma_h^1 (equation of uhat_1), traction, P_k parts of the stresses:
      sP_1(xbar) nubar - tau mu_1 (u_1 - uhat_1)
        - [sP_2(xbar) nubar + tau mu_2 (u_2 - uhat_2)(x')] = gN(xbar) nubar;
  Gamma_h^2 (equation of uhat_2), velocity, transferred along both paths with
  grad u_i = A_i sigma_i + S(gamma_i), sigma_i the whole stress (full=False: P_k part):
      uhat_1.Transfer(p_1) - uhat_2.Transfer(p_2) = gD(xbar).

Pressure mean (rule of stokes_stress.py): the outer boundary is all Dirichlet and the
traction jump fixes only the relative level of the two pressures, so the common mode
p + c is undetermined: one zero-mean constraint for the whole fluid,
(rho_1, 1)_{D_h^1} + (rho_2, 1)_{D_h^2} = 0, with one global multiplier m in the
mass-balance rows of both sides, <uhat_i . n, psi>_dK + (m, psi)_K = 0.
"""
from math import log2

from ngsolve import *

from elasticity import compliance, free_dofs, hdg_rows, postprocess, skew, solve_linear_system, spaces, stress
from elasticity_interface import domains
from elasticity_tpm import at
from ngstpm import interior_ds, pair, transfer_paths
import stokes_stress as st

INF = float("inf")


def solve(mesh, order, phi, D1, D2, mu, f, gD, gN, g, tau=1.0, kind=None, full=True):
    """mu, f: pairs (side 1, side 2); gD(X, Y), gN(X, Y): the jumps of u and sigma at xbar
    (vector, 2x2 CF); g: u_2 on the box (CF). Returns, per side, (sigma_h, u_h, gamma_h,
    uhat_h, ustar_h) and p_h, and m_h."""
    D = (D1, D2)
    p = [transfer_paths(mesh, phi, Di, kind=kind, order=order + 4) for Di in D]
    I = pair(*p)                                                      # common subdivision of Gamma
    dI = [I.ds(side=i, bonus_intorder=max(6, 2 * order)) for i in (1, 2)]

    sp = [spaces(mesh, order, "outer", Di, split=True) for Di in D]
    fes = sp[0][0] * sp[1][0] * NumberSpace(mesh)
    trial, test = fes.TnT()
    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    m, q = trial[16], test[16]
    for i in range(2):                                                # Stokes on each side
        tr, te = trial[8 * i:8 * i + 8], test[8 * i:8 * i + 8]
        sigmahat_n = hdg_rows(lhs, rhs, tr, te, INF, mu[i], f[i], tau, D[i])
        lhs += sigmahat_n * te[5] * interior_ds(mesh, D[i])
        lhs += (m * te[7] + tr[7] * q) * dx(definedonelements=D[i])  # one constraint, one m
    (sP1, w11, w12, u1, gam1, uhat1), (sP2, w21, w22, u2, gam2, uhat2) = trial[:6], trial[8:14]
    muh1, muh2 = test[5], test[13]

    # Gamma_h^1, traction:  sigma_1 nubar - sigma_2 nubar = gN nubar  at xbar
    xbar, nubar = p[0].target, p[0].target_normal
    t1 = sP1.Extrapolate(xbar) * nubar - tau * mu[0] * (u1 - uhat1)
    t2 = sP2.Extrapolate(xbar) * nubar + tau * mu[1] * (u2.Extrapolate() - uhat2.Extrapolate())
    lhs += (t1 - t2) * muh1 * dI[0]
    rhs += gN(*xbar) * nubar * muh1 * dI[0]

    # Gamma_h^2, velocity:  u_1 - u_2 = gD  at xbar
    xbar = p[1].target
    s1 = stress(sP1, w11, w12) if full else sP1
    s2 = stress(sP2, w21, w22) if full else sP2
    ub1 = uhat1.Transfer(p[0], gradient=compliance(s1, INF, mu[0]) + skew(gam1))
    ub2 = uhat2.Transfer(p[1], gradient=compliance(s2, INF, mu[1]) + skew(gam2))
    lhs += (ub1 - ub2) * muh2 * dI[1]
    rhs += gD(*xbar) * muh2 * dI[1]

    gfu = GridFunction(fes)
    for c in (5, 13):
        gfu.components[c].Set(g, BND, definedon=mesh.Boundaries("outer"))
    free = free_dofs(fes, sp[0][1], True)
    free = free_dofs(fes, sp[1][1], True, free=free, first=8)
    solve_linear_system(lhs, rhs, gfu, free, st.multiplier_inverse(fes, 7, 16))
    c = gfu.components
    c[3].gfu = gfu                                      # the whole solution, for study_io
    out = []
    for i in range(2):
        sigma_h = stress(c[8 * i], c[8 * i + 1], c[8 * i + 2])
        u_h, gam_h = c[8 * i + 3], c[8 * i + 4]
        sol = (sigma_h, u_h, gam_h, c[8 * i + 5], postprocess(mesh, order, INF, mu[i], sigma_h, u_h, gam_h, D[i]))
        out.append((sol, -Trace(sigma_h) / 2))
    return out, c[16].vec[0]


PHI = lambda X, Y: sqrt(X * X + Y * Y) - 1                            # unit disk in (-1.5, 1.5)^2
MU = (1.0, 10.0)
PSI2 = sin(x) * exp(y / 2) + x * x * y / 3
U = (st.U, CF((PSI2.Diff(y), -PSI2.Diff(x))))                         # curl psi_1, curl psi_2
P = (st.P, x * y + cos(x))


def problem(u=U, p=P, mu=MU):
    """f pair, exact (sigma_i, gamma_i), gD, gN and g."""
    ex = [st.manufactured(u[i], p[i], mu[i]) for i in range(2)]
    return (ex[0][2], ex[1][2]), ex, at(u[0] - u[1]), at(ex[0][0] - ex[1][0]), u[1]


def errors(mesh, out, D, u=U, p=P, mu=MU):
    """Both sides together (square root of the sum of the squares): |sigma - sigma_h|,
    |sigma - sigma_h - c I|, |p - p_h - c|, |u - u_h|, |gamma - gamma_h|, |P_M u - uhat|_h,
    |u - u*|, with c the mean over D_h^1 and D_h^2 of p - p_h; and c and the per-side
    means of p - p_h."""
    f, ex, *_ = problem(u, p, mu)
    sq = [0.0] * 7
    num = vol = 0.0
    means = []
    for i in range(2):
        dV = dx(definedonelements=D[i], bonus_intorder=4)
        ai, vi = Integrate((p[i] - out[i][1]) * dV, mesh), Integrate(CF(1) * dV, mesh)
        num, vol = num + ai, vol + vi
        means.append(ai / vi)
    c = num / vol
    for i in range(2):
        (sol, p_h), (sigma, gamma, _) = out[i], ex[i]
        e = st.errors(mesh, sol, p_h, u[i], p[i], sigma, gamma, D[i])
        dV = dx(definedonelements=D[i], bonus_intorder=4)
        E = sol[0] - sigma - c * Id(2)
        es = Integrate(InnerProduct(E, E) * dV, mesh)
        ep = Integrate((p[i] - p_h - c) ** 2 * dV, mesh)
        for j, v in enumerate((e[0] ** 2, es, ep, e[2] ** 2, e[3] ** 2, e[4] ** 2, e[5] ** 2)):
            sq[j] += v
    return [sqrt(v) for v in sq], c, means


COLUMNS = ("|s-sh|", "|s-sh-cI|", "|p-ph-c|", "|u-uh|", "|g-gh|", "|Pu-uh^|h", "|u-u*|")
HEADER = (f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS)
          + f"  {'m_h':>10}  {'c':>10}  {'c_1':>10}  {'c_2':>10}  #D1/#D2")


def study(family, order, hs, prune=False, full=True, record=None):
    """EOC table, both sides together; c_i: mean over D_h^i of p - p_h. record: a
    study_io.Record that collects the levels."""
    f, ex, gD, gN, g = problem()
    print(f"{family}, k = {order}, mu = {MU}" + (", pruned" if family == "unfitted" and prune else "")
          + ("" if full else ", P_k part of sigma in the velocity row"))
    print(HEADER)
    prev, rows = None, []
    for h in hs:
        mesh, D1, D2, removed = domains(family, h, PHI, prune)
        out, m_h = solve(mesh, order, PHI, D1, D2, MU, f, gD, gN, g, full=full)
        e, c, means = errors(mesh, out, (D1, D2))
        line = f"  {h:7.4f}"
        for j, ej in enumerate(e):
            r = f"{log2(prev[1][j] / ej) / log2(prev[0] / h):5.2f}" if prev else "   - "
            line += f"  {ej:9.3e} {r}"
        print(line + f"  {m_h:10.2e}  {c:10.2e}  {means[0]:10.2e}  {means[1]:10.2e}  {D1.NumSet()}/{D2.NumSet()}")
        if record is not None:
            record.add(h, mesh, e, out[0][0][1].gfu, m_h=m_h, c=c, c1=means[0], c2=means[1],
                       nD1=D1.NumSet(), nD2=D2.NumSet(), removed1=removed[0], removed2=removed[1])
        prev = (h, e)
        rows.append((e, m_h, c, means))
    return rows


FAMILIES = ("fitted", "snapped", "unfitted")


if __name__ == "__main__":
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(FAMILIES))
    p.add_argument("--prune", action="store_true", help="prune D_h^1, D_h^2 (unfitted family)")
    p.add_argument("--pk", dest="full", action="store_false",
                   help="P_k part of sigma in the velocity row (default: the whole stress)")
    a = parse(p)
    if a.k is None:
        # one process per case, within 2 GB each: h = 0.2, 0.1. Measured peaks: k = 1 at
        # h = 0.1: 0.6 GB; k = 2 at h = 0.1: 0.9 to 1.2 GB; k = 1 at h = 0.05: 3.1 (snapped)
        # and 4.1 GB (unfitted), so h = 0.05 is not a default (see cosmos-pending.md).
        run_defaults(__file__, [["--case", fam, "--k", k, "--h", 0.2, 0.1] for fam in FAMILIES for k in (1, 2)])
    else:
        record = Record(COLUMNS)
        study(a.case, a.k, a.h, prune=a.prune, full=a.full, record=record)
        if a.out:
            save(a.out, record, dict(driver="stokes_stress_interface", case=a.case, k=a.k, mu=list(MU),
                                     tau=1.0, full=a.full, prune=a.prune))

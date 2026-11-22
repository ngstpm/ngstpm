"""Interface problem of linear elasticity, two materials, by the HDG of elasticity.py with
the Transfer Path Method:

    A_i sigma_i - grad u_i + S(gamma_i) = 0,   -div sigma_i = f_i   in Omega_i,
    u_1 - u_2 = gD,   (sigma_1 - sigma_2) nu = gN nu   on Gamma = {phi = 0},
    u_2 = g on the boundary of the box,

with Omega_1 = {phi < 0}, Omega_2 the rest of the box (-1.5, 1.5)^2, A_i the plane-strain
compliance of (lam_i, mu_i) and nu the normal of Gamma from Omega_1 to Omega_2.

Each side has the HDG unknowns of elasticity.py on its element set D_h^i (families
fitted / snapped / unfitted as in poisson_interface.py), tau_S,i = tau mu_i, and the
two transmission conditions are imposed at the ends xbar of the paths, as in
poisson_interface.py:

  Gamma_h^1 (equation of uhat_1), traction, P_k parts of the stresses:
      sP_1(xbar) nubar - tau mu_1 (u_1 - uhat_1)
        - [sP_2(xbar) nubar + tau mu_2 (u_2 - uhat_2)(x')] = gN(xbar) nubar;
  Gamma_h^2 (equation of uhat_2), displacement, transferred along both paths with
  grad u_i = A_i sigma_i + S(gamma_i), sigma_i the whole stress (full=False: P_k part):
      uhat_1.Transfer(p_1) - uhat_2.Transfer(p_2) = gD(xbar).

On the unfitted family each D_h^i can be pruned as in elasticity_tpm.prune (prune=True;
by default the interior subdomains of the background mesh are kept).
"""

from math import log2

from ngsolve import *

from elasticity import (compliance, errors, free_dofs, hdg_rows, lame, manufactured, postprocess,
                        skew, solve_linear_system, spaces, stress)
from elasticity_tpm import at, prune as prune_domain
from ngstpm import interior_ds, pair, transfer_paths
from poisson_interface import domains as poisson_domains


def solve(mesh, order, phi, D1, D2, lam, mu, f, gD, gN, g, tau=1.0, kind=None, full=True):
    """lam, mu, f: pairs (side 1, side 2); gD(X, Y), gN(X, Y): the jumps of u and sigma
    at xbar (vector, 2x2 CF); g: u_2 on the box (CF). Returns, per side,
    (sigma_h, u_h, gamma_h, uhat_h, ustar_h)."""
    D = (D1, D2)
    p = [transfer_paths(mesh, phi, Di, kind=kind, order=order + 4) for Di in D]
    I = pair(*p)                                                      # common subdivision of Gamma
    dI = [I.ds(side=i, bonus_intorder=max(6, 2 * order)) for i in (1, 2)]

    sp = [spaces(mesh, order, "outer", Di) for Di in D]
    fes = sp[0][0] * sp[1][0]
    trial, test = fes.TnT()
    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    for i in range(2):                                                # HDG on each side
        sigmahat_n = hdg_rows(lhs, rhs, trial[6 * i:6 * i + 6], test[6 * i:6 * i + 6],
                              lam[i], mu[i], f[i], tau, D[i])
        lhs += sigmahat_n * test[6 * i + 5] * interior_ds(mesh, D[i])
    (sP1, w11, w12, u1, gam1, uhat1), (sP2, w21, w22, u2, gam2, uhat2) = trial[:6], trial[6:]
    muh1, muh2 = test[5], test[11]

    # Gamma_h^1, traction:  sigma_1 nubar - sigma_2 nubar = gN nubar  at xbar
    xbar, nubar = p[0].target, p[0].target_normal
    t1 = sP1.Extrapolate(xbar) * nubar - tau * mu[0] * (u1 - uhat1)
    t2 = sP2.Extrapolate(xbar) * nubar + tau * mu[1] * (u2.Extrapolate() - uhat2.Extrapolate())
    lhs += (t1 - t2) * muh1 * dI[0]
    rhs += gN(*xbar) * nubar * muh1 * dI[0]

    # Gamma_h^2, displacement:  u_1 - u_2 = gD  at xbar
    xbar = p[1].target
    s1 = stress(sP1, w11, w12) if full else sP1
    s2 = stress(sP2, w21, w22) if full else sP2
    ub1 = uhat1.Transfer(p[0], gradient=compliance(s1, lam[0], mu[0]) + skew(gam1))
    ub2 = uhat2.Transfer(p[1], gradient=compliance(s2, lam[1], mu[1]) + skew(gam2))
    lhs += (ub1 - ub2) * muh2 * dI[1]
    rhs += gD(*xbar) * muh2 * dI[1]

    gfu = GridFunction(fes)
    for c in (5, 11):
        gfu.components[c].Set(g, BND, definedon=mesh.Boundaries("outer"))
    free = free_dofs(fes, sp[0][1], True)
    free = free_dofs(fes, sp[1][1], True, free=free, first=6)
    solve_linear_system(lhs, rhs, gfu, free)
    c = gfu.components
    c[3].gfu = gfu                                      # the whole solution, for study_io
    out = []
    for i in range(2):
        sigma_h = stress(c[6 * i], c[6 * i + 1], c[6 * i + 2])
        u_h, gam_h = c[6 * i + 3], c[6 * i + 4]
        out.append((sigma_h, u_h, gam_h, c[6 * i + 5],
                    postprocess(mesh, order, lam[i], mu[i], sigma_h, u_h, gam_h, D[i])))
    return out


def domains(family, maxh, phi, prune=False):
    """mesh, D1, D2 of the family (poisson_interface.domains); on the unfitted family
    each side is pruned if prune. Also returns the numbers removed."""
    mesh, D1, D2 = poisson_domains(family, maxh, phi)
    if family != "unfitted" or not prune:
        return mesh, D1, D2, (0, 0)
    P1 = prune_domain(mesh, phi, D1)
    P2 = prune_domain(mesh, lambda X, Y: -phi(X, Y), D2)
    return mesh, P1, P2, (D1.NumSet() - P1.NumSet(), D2.NumSet() - P2.NumSet())


PHI = lambda X, Y: sqrt(X * X + Y * Y) - 1                            # unit disk in (-1.5, 1.5)^2
MATERIALS = (lame(1.0, 0.3), lame(10.0, 0.25))                        # (lam_i, mu_i)
U = (CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y))),
     CF((exp(x / 2) * cos(y / 2) + x * y, sin(x) * y + x * x / 2)))


def problem(u=U, materials=MATERIALS):
    """lam, mu, f pairs, exact (sigma_i, gamma_i), and gD, gN, g for displacements u."""
    lam, mu = zip(*materials)
    ex = [manufactured(u[i], lam[i], mu[i]) for i in range(2)]
    f = (ex[0][2], ex[1][2])
    return lam, mu, f, ex, at(u[0] - u[1]), at(ex[0][0] - ex[1][0]), u[1]


def side_errors(mesh, sol, D, u=U, materials=MATERIALS):
    """errors() of elasticity.py on each side (8 columns each)."""
    lam, mu, f, ex, *_ = problem(u, materials)
    return [errors(mesh, sol[i], u[i], ex[i][0], ex[i][1], D[i]) for i in range(2)]


COLUMNS = ("|s-sh|", "|u-uh|", "|g-gh|", "|Pu-uh^|h", "|u-u*|")
PICK = (0, 1, 2, 4, 5)                                                # of the 8 columns of errors


def study(family, order, hs, prune=False, full=True, record=None):
    """EOC table, both sides together (square root of the sum of the squares); record: a
    study_io.Record that collects the levels."""
    lam, mu, f, ex, gD, gN, g = problem()
    print(f"{family}, k = {order}" + (", pruned" if family == "unfitted" and prune else "")
          + (", not pruned" if family == "unfitted" and not prune else "")
          + ("" if full else ", P_k part of sigma in the displacement row"))
    print(f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS) + "  #D1/#D2  removed")
    prev, rows = None, []
    for h in hs:
        mesh, D1, D2, removed = domains(family, h, PHI, prune)
        sol = solve(mesh, order, PHI, D1, D2, lam, mu, f, gD, gN, g, full=full)
        e1, e2 = side_errors(mesh, sol, (D1, D2))
        e = [sqrt(e1[j] ** 2 + e2[j] ** 2) for j in PICK]
        line = f"  {h:7.4f}"
        for j, ej in enumerate(e):
            r = f"{log2(prev[1][j] / ej) / log2(prev[0] / h):4.2f}" if prev else "  - "
            line += f"  {ej:9.3e}  {r}"
        print(line + f"  {D1.NumSet()}/{D2.NumSet()}  {removed[0]}/{removed[1]}")
        if record is not None:
            record.add(h, mesh, e, sol[0][1].gfu, nD1=D1.NumSet(), nD2=D2.NumSet(),
                       removed1=removed[0], removed2=removed[1])
        prev = (h, e)
        rows.append((e1, e2))
    return rows


FAMILIES = ("fitted", "snapped", "unfitted")


if __name__ == "__main__":
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(FAMILIES))
    p.add_argument("--prune", action="store_true", help="prune D_h^1, D_h^2 (unfitted family)")
    p.add_argument("--pk", dest="full", action="store_false",
                   help="P_k part of sigma in the displacement row (default: the whole stress)")
    a = parse(p)
    if a.k is None:
        # one process per case, within 2 GB each (h = 0.025: 2.1 GB for k = 1, about 5 GB for k = 2)
        run_defaults(__file__, [["--case", fam, "--k", k, "--h", 0.2, 0.1, 0.05]
                                for fam in FAMILIES for k in (1, 2)])
    else:
        record = Record(COLUMNS)
        study(a.case, a.k, a.h, prune=a.prune, full=a.full, record=record)
        if a.out:
            (l1, m1), (l2, m2) = MATERIALS
            save(a.out, record, dict(driver="elasticity_interface", case=a.case, k=a.k, tau=1.0,
                                     lam=[l1, l2], mu=[m1, m2], E=[1.0, 10.0], nu=[0.3, 0.25],
                                     full=a.full, prune=a.prune))

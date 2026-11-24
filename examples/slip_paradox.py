"""The polygonal free-slip paradox for Stokes flow in stress form (stokes_stress.py), and
its cure by the Transfer Path Method (port of Stage 4 of
ngsolve-tpm/experiments/slip_paradox).

Fluid between the circle r = R1 (Dirichlet, a rotating cylinder u = Omega x^perp) and the
circle Gamma: r = R2 with free slip, u . nu = 0, tau . sigma nu = 0. Exact solution: the
rigid rotation u = Omega x^perp, sigma = 0 (torque M = 0, vorticity 2 Omega). The mesh
replaces Gamma by the inscribed regular N-gon Gamma_h (edges split in m), all elements
straight. On Gamma_h the conservativity row is replaced by two rows, tested with
muh . n and muh . t:

  S    (no transfer)  uhat . n_h = 0,  t_h . sigma_hat n_h = 0  (n_h the polygon normal),
       sigma_hat n = sigma n - tau mu (u - uhat): free slip on the polygon. As N grows
       this tends to the Hodge-type limit, not to the exact solution (Babuska paradox);
  TPM  the trace and the traction transferred to the end xbar of the radial path,
       nubar = the normal of Gamma at xbar, tbar = nubar^perp:
           nubar . uhat.Transfer(p, gradient=A sigma + S(gamma)) = g_n(xbar),
           tbar . sP(xbar) nubar - tau mu tbar . (u - uhat)(x)  = g_t(xbar),
       with the P_k part sP of the stress extrapolated (as stokes_stress_tpm_neumann.py;
       MATLAB InterfaceProblem/solve.m, interface_law "full": E_P1_N and -tau mu D1).

Pressure mean: the normal velocity is prescribed on the whole boundary, so the
zero-mean constraint with the multiplier m is used (rule of stokes_stress.py).

Diagnostics (paradox case): relative L2 errors of u_h against the exact, Hodge
(Omega R1^2 x^perp / r^2, M = pi) and no-slip (M = 4 pi / 3) rotations; the torques
M_s = -int_{Gamma_h} x^perp . sigma_hat n and M_in = -int_{r=R1} x^perp . sigma_hat n
(M_s + M_in = 0 by the conservativity of the scheme); the mean vorticity on Gamma_h.

Manufactured case (TPM only): u = stokes_stress.U, p = stokes_stress.P with the
inhomogeneous slip data g_n = u . nu, g_t = tau . sigma nu at xbar, refined by doubling N
at fixed m, to measure the EOC (the rigid rotation is reproduced exactly by TPM).

Differences from ngsolve-tpm (Stage 4):
- ngsolve-tpm extrapolated the whole stress with its enrichment in the tangential row;
  here the P_k part, as ngstpm's transferred traction and MATLAB.
- Fluid-only mesh: the one-way solid solve and the projected traction are not ported.
- Zero-mean pressure by the multiplier (stokes_stress.finish), not pinning + shift.
"""
import argparse
import math
from math import log2

from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *

from elasticity import compliance, skew, stress
from ngstpm import boundary_ds, interior_ds, transfer_paths
import stokes_stress as st
from stokes_stress_tpm import table_errors

INF = float("inf")
R1, R2 = 0.5, 1.0
MU, OMEGA = 1.0, 1.0
PHI = lambda X, Y: sqrt(X * X + Y * Y) - R2                           # Gamma = {phi = 0}
XPERP = CF((-y, x))
_A = -OMEGA * R1**2 / (R2**2 - R1**2)
_B = OMEGA * R1**2 * R2**2 / (R2**2 - R1**2)
REF = {"exact": OMEGA * XPERP,
       "hodge": OMEGA * R1**2 / (x * x + y * y) * XPERP,
       "noslip": (_A + _B / (x * x + y * y)) * XPERP}
REF_TORQUE = {"exact": 0.0, "hodge": math.pi, "noslip": 4 * math.pi / 3}


def polygon_mesh(N, m=1, maxh=None):
    """The fluid R1 < r, inside the regular N-gon inscribed in r = R2, edges of the N-gon
    of length L/m (L = 2 R2 sin(pi/N)); global maxh min(0.1, 4 L/m) unless given (the
    sizing of ngsolve-tpm; Netgen may split the N-gon edges further). Boundaries "inner"
    and "interface"; straight elements."""
    L = 2 * R2 * math.sin(math.pi / N)
    wp = WorkPlane().MoveTo(R2, 0.0)
    for j in range(1, N):
        wp.LineTo(R2 * math.cos(2 * math.pi * j / N), R2 * math.sin(2 * math.pi * j / N))
    poly = wp.Close().Face()
    poly.edges.name = "interface"
    poly.edges.maxh = L / m
    hole = WorkPlane().Circle(0, 0, R1).Face()
    hole.edges.name = "inner"
    fluid = poly - hole
    fluid.faces.name = "fluid"
    maxh = min(0.1, 4 * L / m) if maxh is None else maxh
    return Mesh(OCCGeometry(fluid, dim=2).GenerateMesh(maxh=maxh, grading=0.3))


def solve(mesh, order, mu, f, g, transfer=True, gn=None, gt=None, tau=1.0, compile=True):
    """Slip on "interface" (S if not transfer, TPM if transfer), uhat = g on "inner".
    gn(X, Y), gt(X, Y): the slip data at xbar (default 0; TPM only). compile: compile the
    slip rows (same matrix, far faster assembly of nbar = normal(phi, *xbar)).
    Returns (sigma_h, u_h, gamma_h, uhat_h, ustar_h), p_h and m_h."""
    iface = mesh.Boundaries("interface")
    dB = boundary_ds(mesh, facets=iface, bonus_intorder=max(6, 2 * order))
    fes, top, lhs, rhs, trial, test, sigmahat_n = st.forms(mesh, order, mu, f, "inner", tau)
    sP, w1, w2, u, gam, uhat = trial[:6]
    muh = test[5]
    lhs += sigmahat_n * muh * interior_ds(mesh)
    if transfer:
        p = transfer_paths(mesh, PHI, facets=iface, order=order + 4)    # radial, x -> p.target
        n = p.target_normal
        trace = uhat.Transfer(p, gradient=compliance(stress(sP, w1, w2), INF, mu) + skew(gam))
        traction = sP.Extrapolate(p.target) * n - tau * mu * (u - uhat)
    else:
        if gn is not None or gt is not None:
            raise ValueError("slip data only with transfer")
        n = specialcf.normal(2)                                         # polygon normal
        trace, traction = uhat, sigmahat_n
    t = CF((-n[1], n[0]))
    row = (InnerProduct(trace, n) * InnerProduct(muh, n)
           + InnerProduct(traction, t) * InnerProduct(muh, t))
    lhs += (row.Compile() if compile else row) * dB          # Compile: ~40x faster with nbar
    if gn is not None:
        rhs += gn(*p.target) * InnerProduct(muh, n) * dB
    if gt is not None:
        rhs += gt(*p.target) * InnerProduct(muh, t) * dB
    st.add_constraint(lhs, trial, test)
    return st.finish(lhs, rhs, fes, top, (g, BND, mesh.Boundaries("inner")), mesh, order, mu)


def diagnostics(mesh, sol, order, tau=1.0, mu=MU):
    """Relative L2 errors against REF, M_s, M_in, |M_s + M_in| and the mean vorticity on
    Gamma_h."""
    sigma_h, u_h, _, uhat_h, _ = sol
    dV = dx(bonus_intorder=2 * order + 4)
    out = {}
    for name, ref in REF.items():
        e = Integrate(InnerProduct(u_h - ref, u_h - ref) * dV, mesh)
        out["err_" + name] = math.sqrt(abs(e) / Integrate(InnerProduct(ref, ref) * dV, mesh))
    flux = sigma_h * specialcf.normal(2) - tau * mu * (u_h - uhat_h)
    dS = {b: boundary_ds(mesh, facets=mesh.Boundaries(b), bonus_intorder=2 * order + 2)
          for b in ("interface", "inner")}
    out["M_s"] = -Integrate(InnerProduct(XPERP, flux) * dS["interface"], mesh)
    out["M_in"] = -Integrate(InnerProduct(XPERP, flux) * dS["inner"], mesh)
    out["torque_balance"] = abs(out["M_s"] + out["M_in"])
    G = grad(u_h)                                                       # G[i, j] = d_j u_i
    out["vort_iface"] = (Integrate((G[1, 0] - G[0, 1]) * dS["interface"], mesh)
                         / Integrate(CF(1) * dS["interface"], mesh))
    return out


def paradox(order, Ns, m=1, forms=("S", "TPM")):
    """The rotating-cylinder table: one row per (form, N)."""
    print(f"rotating cylinder, k = {order}, m = {m}; references M: exact 0, hodge pi, "
          f"noslip 4 pi/3")
    cols = ("err_exact", "err_hodge", "M_s", "torque_balance", "vort_iface")
    print(f"  {'form':>4}  {'N':>4}" + "".join(f"  {c:>14}" for c in cols))
    rows = []
    for form in forms:
        for N in Ns:
            mesh = polygon_mesh(N, m)
            sol, _, _ = solve(mesh, order, MU, CF((0, 0)), OMEGA * XPERP, transfer=form == "TPM")
            d = diagnostics(mesh, sol, order)
            print(f"  {form:>4}  {N:4d}" + "".join(f"  {d[c]:14.6e}" for c in cols))
            rows.append(dict(form=form, N=N, m=m, k=order, **d))
    return rows


def manufactured(order, Ns, m=1, mu=MU):
    """EOC of TPM with the inhomogeneous slip data of st.U, st.P (h ~ 1/(N m))."""
    sigma, gamma, f = st.manufactured(st.U, st.P, mu)
    nu = lambda X, Y: CF((X, Y)) / sqrt(X * X + Y * Y)
    at = lambda cf, X, Y: cf.Replace({x: X, y: Y})
    gn = lambda X, Y: InnerProduct(at(st.U, X, Y), nu(X, Y))
    gt = lambda X, Y: InnerProduct(CF((-nu(X, Y)[1], nu(X, Y)[0])), at(sigma, X, Y) * nu(X, Y))
    print(f"manufactured slip (TPM), k = {order}, m = {m}, mu = {mu}")
    cols = ("|s-sh|", "|s-sh-cI|", "|p-ph-c|", "|u-uh|", "|g-gh|", "|Pu-uh^|h", "|u-u*|")
    print(f"  {'N':>4}  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in cols) + f"  {'m_h':>10}")
    prev, rows = None, []
    for N in Ns:
        mesh = polygon_mesh(N, m, maxh=2 * R2 * math.sin(math.pi / N) / m)
        h = 2 * R2 * math.sin(math.pi / N) / m
        sol, p_h, m_h = solve(mesh, order, mu, f, st.U, gn=gn, gt=gt)
        e, _ = table_errors(mesh, sol, p_h, st.U, st.P, sigma, gamma, None)
        line = f"  {N:4d}  {h:7.4f}"
        for i, ei in enumerate(e):
            r = f"{log2(prev[1][i] / ei) / log2(prev[0] / h):4.2f}" if prev else "  - "
            line += f"  {ei:9.3e}  {r}"
        print(line + f"  {m_h:10.2e}")
        prev = (h, e)
        rows.append((N, h, e, m_h))
    return rows


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--case", choices=("paradox", "manufactured"), default="paradox")
    a.add_argument("--k", type=int, nargs="+", default=[1, 2], help="polynomial degrees")
    a.add_argument("--N", type=int, nargs="+", default=[8, 16, 32, 64], help="polygon sizes")
    a.add_argument("--m", type=int, default=1, help="mesh edges per polygon edge")
    args = a.parse_args()
    for k in args.k:
        (paradox if args.case == "paradox" else manufactured)(k, args.N, args.m)

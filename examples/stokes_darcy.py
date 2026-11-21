"""Stokes-Darcy coupling by HDG with the Transfer Path Method on two meshes
(Bermudez-Manriquez-Solano 2025, BMS):

    L_s = grad u_s,  -div(nu L_s - P_s I) = f_s,  div u_s = 0  in Omega_s,  u_s = g on Gamma_s,
    u_d + kappa grad p_d = 0,  div u_d = f_d  in Omega_d,  u_d . n_d = gN on Gamma_d (or p_d = gD on a part),
    u_s . n_s + u_d . n_d = T_mass,
    (nu L_s - P_s I) n_s + (1/omega) (u_s . m) m - p_d n_d = T_force  on the interface,

with n_s = -n_d the normal of the interface from Omega_s to Omega_d and m its
tangent (the data T_mass, T_force are zero in the model; they let a
manufactured solution violate the transmission conditions).

Stokes (side 1) has the HDG of stokes.py on mesh 1, Darcy (side 2) that of
darcy.py on mesh 2. The transmission conditions are imposed at xbar on the
interface (BMS (3.3), (3.6)), the mass balance as the equation of the Darcy
trace on Gamma_h^d, the force balance as that of the Stokes trace on
Gamma_h^s, with the traces moved to xbar along the paths:

    utilde_s . n_s + (E u_d)(xbar) . n_d + tau (p_d - phat_d) = T_mass(xbar),
        utilde_s = uhat_s + int (E L_s) dl,
    (nu E L_s - (E ptilde_s + alpha) I)(xbar) n_s - nu tau (u_s - uhat_s)
        + (1/omega) (utilde_s . m) m - ptilde_d n_d = T_force(xbar),
        ptilde_d = phat_d - int (kappa^{-1} E u_d) . dl.

The pressure level: if the Darcy side has no Dirichlet part, (P_s + c, p_d + c)
solves the homogeneous problem; then ptilde_s has zero mean on mesh 1 (the
multiplier rho) and the level alpha, an unknown of the system, makes the mean
of P_s = ptilde_s + alpha over Omega_s zero:

    (alpha, 1)_{Omega_h^s} + int_{Omega_s \\ Omega_h^s} E(ptilde_s + alpha) = 0.

With a Darcy Dirichlet part there is neither rho nor alpha.

Meshes (BMS Section 6): structured, mesh 1 = (0,1) x (1/2 + delta, 1),
mesh 2 = (0,1) x (0, 1/2 - delta), the gap 2 delta around the interface y = 1/2
unmeshed; h = 1/n is the Stokes mesh size (the paper's meshes are unstructured,
and its Table 6 meshes are not semi-aligned while ours are). With delta = 0 and
delta = h^{7/4} all of L, u, p converge with order k+1 (BMS Tables 1-4; with the
gap the rates of p are above k+1 on these meshes before they settle). With
delta = h the rates are regular and approach k+1 from below (k = 1: 1.90, 1.90,
1.98 for L, u, p at n = 64), with errors several times larger than without
gap, as in BMS Tables 5, 6; delta = h/2, h/4 behave the same with smaller
constants.
"""

from concurrent.futures import ProcessPoolExecutor
from math import log

import numpy as np
from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh

from darcy import darcy_data
from darcy_tpm import inverse
from poisson import solve_linear_system
from stokes import multiplier_inverse, stokes_data
from ngstpm import boundary_ds, interior_ds, normal, pair, transfer_paths


def meshes(n, hanging=False, delta=0.0):
    """Structured meshes of (0,1) x (1/2 + delta, 1) (Stokes, n x n/2 squares, two
    triangles each) and (0,1) x (0, 1/2 - delta) (Darcy, the same, or n/2 x n/4 with
    hanging=True: each Darcy interface edge then faces two Stokes edges). Boundaries:
    "interface"; "outer" (and "bottom", the bottom side of the Darcy mesh)."""
    m = n // 2 if hanging else n
    mesh1 = MakeStructured2DMesh(quads=False, nx=n, ny=n // 2,
                                 mapping=lambda s, t: (s, 0.5 + delta + (0.5 - delta) * t))
    mesh2 = MakeStructured2DMesh(quads=False, nx=m, ny=m // 2, mapping=lambda s, t: (s, (0.5 - delta) * t))
    for mesh, names in ((mesh1, ("interface", "outer", "outer", "outer")),
                        (mesh2, ("bottom", "outer", "interface", "outer"))):  # bottom, right, top, left
        for i, name in enumerate(names):
            mesh.ngmesh.SetBCName(i, name)
    return Mesh(mesh1.ngmesh), Mesh(mesh2.ngmesh)


def at(cf, X):
    """cf (a CF of x, y) at the point X, e.g. xbar."""
    return cf.Replace({x: X[0], y: X[1]})


def solve(mesh1, mesh2, order, phi, data, nu=1.0, kappa=CF((1, 0, 0, 1), dims=(2, 2)), omega=1.0, tau=1.0,
          darcy_dirichlet=None, condense=True):
    """data: dict of CFs f_s, g (u_s on Gamma_s), f_d, u_d (its normal component is gN),
    p_d (the Dirichlet datum), T_mass, T_force. darcy_dirichlet: boundary names of mesh 2
    with p_d = data["p_d"] (None: Darcy all Neumann; then rho and alpha are active).
    Returns (L_s, u_s, P_s, uhat_s, u_d, p_d, phat_d), rho, alpha (0, 0 without the level)."""
    level = darcy_dirichlet is None
    p1 = transfer_paths(mesh1, phi, facets=mesh1.Boundaries("interface"))   # Stokes: all of mesh 1
    p2 = transfer_paths(mesh2, phi, facets=mesh2.Boundaries("interface"))   # Darcy: all of mesh 2
    I = pair(p1, p2)                                                  # common subdivision of the interface
    bonus = max(6, 2 * order)
    dI = [I.ds(side=i, bonus_intorder=bonus) for i in (1, 2)]

    fes1 = (MatrixValued(L2(mesh1, order=order)) * VectorL2(mesh1, order=order)
            * L2(mesh1, order=order, lowest_order_wb=True)                  # element means of p stay global
            * FacetFESpace(mesh1, order=order, dirichlet="outer") ** 2)
    if level:
        fes1 = fes1 * NumberSpace(mesh1) * NumberSpace(mesh1)               # rho, alpha
    fes2 = (VectorL2(mesh2, order=order) * L2(mesh2, order=order)
            * FacetFESpace(mesh2, order=order, **({"dirichlet": darcy_dirichlet} if darcy_dirichlet else {})))
    fes = I.Space(fes1, fes2)
    trial, test = fes.TnT()
    (L, u, ph, uhat), (G, v, q, mu) = trial[:4], test[:4]
    ud, pd, pdhat = trial[-3:]
    vd, qd, mud = test[-3:]

    lhs = I.BilinearForm(fes, condense=condense)
    rhs = I.LinearForm(fes)
    n = specialcf.normal(2)
    kappa_inv = inverse(kappa)

    # Stokes: the HDG of stokes.py
    div_G = fem.Einsum("ijj->i", grad(G))                             # row-wise: (div G)_i = d_j G_ij
    sigmahat_n = nu * L * n - ph * n - tau * nu * (u - uhat)
    dS1 = dx(element_boundary=True)
    lhs += InnerProduct(L, G) * dx + u * div_G * dx - uhat * (G * n) * dS1
    lhs += nu * InnerProduct(L, grad(v)) * dx - ph * Trace(grad(v)) * dx - sigmahat_n * v * dS1
    rhs += data["f_s"] * v * dx
    lhs += -u * grad(q) * dx + (uhat * n) * q * dS1
    lhs += sigmahat_n * mu * interior_ds(mesh1)

    # Darcy: the HDG of darcy.py
    uhat_n = ud * n + tau * (pd - pdhat)
    lhs += kappa_inv * ud * vd * dx - pd * div(vd) * dx + pdhat * (vd * n) * dS1
    lhs += -ud * grad(qd) * dx + uhat_n * qd * dS1
    rhs += data["f_d"] * qd * dx
    lhs += uhat_n * mud * interior_ds(mesh2)
    neumann = "bottom|outer" if darcy_dirichlet is None else \
        "|".join(b for b in ("bottom", "outer") if b not in darcy_dirichlet.split("|"))
    if neumann:
        dN = boundary_ds(mesh2, facets=mesh2.Boundaries(neumann))
        lhs += uhat_n * mud * dN
        rhs += (data["u_d"] * n) * mud * dN

    # Darcy interface faces, mass:  utilde_s . n_s + (E u_d)(xbar) . n_d + tau (p_d - phat_d) = T_mass(xbar)
    xbar, n_s = p2.target, p2.target_normal                           # n_s: from Stokes to Darcy
    utilde_s = uhat.Transfer(p1, gradient=L)
    lhs += (utilde_s * n_s - ud.Extrapolate(xbar) * n_s + tau * (pd - pdhat)) * mud * dI[1]
    rhs += at(data["T_mass"], xbar) * mud * dI[1]

    # Stokes interface faces, force:
    #   (nu E L_s - (E ptilde_s + alpha) I)(xbar) n_s - nu tau (u_s - uhat_s) + (1/omega)(utilde_s . m) m - ptilde_d n_d = T_force(xbar)
    xbar, n_s = p1.target, p1.target_normal
    m = CF((-n_s[1], n_s[0]))
    P = ph.Extrapolate(xbar) + (trial[5] if level else 0)
    ptilde_d = pdhat.Transfer(p2, gradient=-kappa_inv * ud)
    force = ((nu * L.Extrapolate(xbar) - P * Id(2)) * n_s - nu * tau * (u - uhat)
             + (1 / omega) * (utilde_s * m) * m + ptilde_d * n_s)
    lhs += force * mu * dI[0]
    rhs += at(data["T_force"], xbar) * mu * dI[0]

    if level:
        rho, alpha = trial[4], trial[5]
        rho_, alpha_ = test[4], test[5]
        lhs += ph * rho_ * dx + rho * q * dx                          # (ptilde_s, 1)_{Omega_h^s} = 0
        lhs += alpha * alpha_ * dx + (ph + alpha).RegionIntegral(p1) * alpha_ * dI[0]

    gfu = I.GridFunction(fes)
    gfu.components[3].Set(data["g"], definedon=mesh1.Boundaries("outer"))
    if darcy_dirichlet:
        gfu.components[-1].Set(data["p_d"], definedon=mesh2.Boundaries(darcy_dirichlet))
    solve_linear_system(lhs, rhs, gfu, inverse=multiplier_inverse(fes1) if level else "umfpack")

    c = gfu.components
    rho, alpha = (c[4].vec[0], c[5].vec[0]) if level else (0.0, 0.0)
    return (c[0], c[1], c[2] + alpha, c[3], *c[-3:]), rho, alpha


def stokes_darcy_data(u_s, P_s, p_d, phi, nu=1.0, kappa=CF((1, 0, 0, 1), dims=(2, 2)), omega=1.0):
    """The fields and data (CFs of x, y) of an exact solution: L_s, f_s, u_d, f_d and the interface
    data T_mass = u_s . n_s + u_d . n_d, T_force = (nu L_s - P_s I) n_s + (1/omega)(u_s . m) m - p_d n_d
    (n_s = -n_d from the normal of the level sets of phi, m = (-n_s[1], n_s[0]))."""
    L_s, f_s = stokes_data(u_s, P_s, nu)
    u_d, f_d = darcy_data(p_d, kappa)
    n_s = normal(phi, x, y)
    m = CF((-n_s[1], n_s[0]))
    T_mass = u_s * n_s - u_d * n_s
    T_force = (nu * L_s - P_s * Id(2)) * n_s + (1 / omega) * (u_s * m) * m + p_d * n_s
    return dict(u_s=u_s, L_s=L_s, P_s=P_s, f_s=f_s, g=u_s, u_d=u_d, p_d=p_d, f_d=f_d, T_mass=T_mass, T_force=T_force)


def phi(X, Y):
    return 0.5 - Y                                                    # Omega_1 = Stokes = {y > 1/2}


def paper_data():
    """BMS Section 6: nu = 1, kappa = I, omega = 1."""
    e = exp(1)
    P_s = exp(-x - y) - 2 * exp(-2) * (sqrt(e) - 1) ** 2 * (sqrt(e) + 1)  # zero mean on (0,1) x (1/2,1)
    u_s = CF((pi * cos(pi * x) * sin(pi * y), -pi * cos(pi * y) * sin(pi * x)))
    p_d = cos(pi * x) * cos(pi * y)
    return stokes_darcy_data(u_s, P_s, p_d, phi)


def errors(mesh1, mesh2, sol, data):
    """|L_s - L_s,h|, |u_s - u_s,h|, |P_s - P_s,h|, |u_d - u_d,h|, |p_d - p_d,h|."""
    pairs = ((mesh1, sol[0], data["L_s"]), (mesh1, sol[1], data["u_s"]), (mesh1, sol[2], data["P_s"]),
             (mesh2, sol[4], data["u_d"]), (mesh2, sol[5], data["p_d"]))
    return [sqrt(Integrate(InnerProduct(a - b, a - b), mesh, order=14))
            for mesh, a, b in pairs]


DELTA = {"0": lambda h: 0.0, "h^7/4": lambda h: h ** 1.75, "h": lambda h: h}   # h = 1/n, the Stokes mesh size
NS = {"0": {1: (8, 16, 32, 64), 2: (8, 16, 32, 64), 3: (4, 8, 16, 32)},   # k = 3, n = 64: 2.0 GB
      "h^7/4": {1: (8, 16, 32, 64), 2: (8, 16, 32, 64), 3: (4, 8, 16, 32)},
      "h": {1: (8, 16, 32, 64), 2: (8, 16, 32, 64), 3: (8, 16, 32, 48)}}  # n = 4: delta = 1/4


def row(hanging, k, n, gap="0"):
    """The errors (and that of ptilde_s alone), |rho|, alpha and N for the paper's solution."""
    data = paper_data()
    mesh1, mesh2 = meshes(n, hanging, DELTA[gap](1 / n))
    sol, rho, alpha = solve(mesh1, mesh2, k, phi, data)
    eptilde = sqrt(Integrate((sol[2] - alpha - data["P_s"]) ** 2, mesh1, order=14))
    return errors(mesh1, mesh2, sol, data) + [eptilde], abs(rho), alpha, mesh1.ne + mesh2.ne


if __name__ == "__main__":
    # each row in a new process: NGSolve's umfpack inverse does not give its memory back
    with ProcessPoolExecutor(1, max_tasks_per_child=1) as pool:
        for gap in DELTA:
            for hanging in (False, True):
                print(f"delta = {gap}, " + ("hanging nodes" if hanging else "matching vertices"))
                for k in (1, 2, 3):
                    print(f"  k = {k}")
                    prev = None
                    for n in NS[gap][k]:
                        e, rho, alpha, N = pool.submit(row, hanging, k, n, gap).result()
                        tot = [e[0], np.hypot(e[1], e[3]), np.hypot(e[2], e[4])] + e[1:]
                        r = ([f"({log(a / b) / log(n / prev[-1]):.2f})" for a, b in zip(prev, tot)] if prev
                             else [" " * 6] * len(tot))
                        names = ("e(L)", "e(u)", "e(p)", "u_s", "P_s", "u_d", "p_d", "ptilde_s")
                        print(f"    n={n:3d}  N={N:6d}" + "".join(f"  {c}={v:.3e} {ri}" for c, v, ri in zip(names, tot, r))
                              + f"  |rho|={rho:.1e}  alpha={alpha:.1e}", flush=True)
                        prev = tot + [n]

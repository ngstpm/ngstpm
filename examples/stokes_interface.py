"""Stokes interface problem by HDG with the Transfer Path Method:

    L_i = grad u_i,  -div(nu L_i - p_i I) = f,  div u_i = 0  in Omega_1 = {phi < 0}, Omega_2 = {phi > 0},
    u_1 = u_2,  (nu L_1 - p_1 I) nu = (nu L_2 - p_2 I) nu  on Gamma = {phi = 0},
    u = g on the outer boundary,  (p_1, 1)_{Omega_1} + (p_2, 1)_{Omega_2} = 0,

with nu (the vector) the normal of Gamma from Omega_1 to Omega_2
(Manriquez-Nguyen-Solano 2022, (4a)-(4k) with the assignment of Remark 1).

Each side has the HDG of stokes.py on D_h^i; the transmission conditions are
imposed at xbar on Gamma as in poisson_interface.py: the stress condition is
the equation of uhat_1 on Gamma_h^1, the trace condition that of uhat_2 on
Gamma_h^2. The pressure is p_i = ptilde_i + alpha: only the sum of the means
of ptilde_1, ptilde_2 over D_h^1, D_h^2 is constrained, by one multiplier
bordering the matrix (stokes.multiplier_inverse), and the level alpha, an
unknown of the system, makes the mean of p over Omega zero:

    (alpha, 1)_{D_1} + (alpha, 1)_{D_2} + int_{sliver_1} (P + alpha) + int_{sliver_2} (P + alpha) = 0,

sliver_i the region swept by the paths of side i (from Gamma_h^i to Gamma) and
P = (E ptilde_1 + E ptilde_2)/2 the average of the two extrapolations (M22,
Section 5).

Meshes: the unit square fitted to the flat interface y = 1/2 (straight
elements; the paths have length zero), and the unit disk in (-1.5, 1.5)^2 on
the families of poisson_interface.py (fitted, snapped, unfitted).

With gaps of width O(h^2) (fitted, snapped) L, u and p converge with order
k+1 (snapped, k = 1: L and p dip to about 1.9). With the O(h) gap of the
unfitted family the rates are irregular, as for the flux of
poisson_interface.py on the same meshes (the open issue D21 of TODO.md; the
stress condition is a flux transferred over O(h) paths). Jaime's note: "for
O(h) gaps it is known that extrapolation may not be the best, which makes the
flux interface condition suboptimal".
"""

from concurrent.futures import ProcessPoolExecutor
from math import log
import sys

import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import *

from poisson import solve_linear_system
from poisson_interface import domains
from stokes import multiplier_inverse, stokes_data
from ngstpm import interior_ds, pair, restrict, transfer_paths

def sliver(path, f, npoints):
    """On Gamma_h of the path's side, sum_j w_j J(s_j) f(x + s_j d): the formula of
    RegionIntegral, with f a callable of the point, so that under I.ds(side=own) f
    may extrapolate a trial function of the other side (from the partner element)."""
    X, d, n = CF((x, y)), path.ell * path.sigma, specialcf.normal(2)
    s, w = np.polynomial.legendre.leggauss(npoints)
    return sum(0.5 * wj * InnerProduct(d, Cof(Id(2) + 0.5 * (sj + 1) * path.grad_d) * n) * f(X + 0.5 * (sj + 1) * d)
               for sj, wj in zip(s, w))


def gap_pressure(i, paths, ph, order):
    """int over sliver_i (swept by the paths of side i) of (E ptilde_1 + E ptilde_2)/2,
    an integrand on Gamma_h^i, for I.ds(side=i)."""
    return 0.5 * (ph[i].RegionIntegral(paths[i]) + sliver(paths[i], ph[1 - i].Extrapolate, order + 2))


def gap_integral(I, fes, gfu, dI, paths, ph, mu, order):
    """int over both slivers of (E ptilde_1 + E ptilde_2)/2 for the solution gfu: a row
    assembled with the facet test functions mu_i, applied to gfu.vec and summed with the
    facet function e = (1, 0) (constants are in the facet space)."""
    e = I.GridFunction(fes)
    for c in (3, 7):
        e.components[c].Set(CF((1, 0)), dual=True)
    B = I.BilinearForm(fes)
    for i in (0, 1):
        B += gap_pressure(i, paths, ph, order) * mu[i][0] * dI[i]
    B.Assemble()
    y = B.mat.CreateColVector()
    y.data = B.mat * gfu.vec
    return InnerProduct(e.vec, y)


def solve(mesh, order, phi, D1, D2, f, g, nu=1.0, tau=1.0):
    """Returns (L_1, u1, p1, uhat1, L_2, u2, p2, uhat2), rho, alpha, with
    p_i = ptilde_i + alpha (alpha an unknown of the system)."""
    p1 = transfer_paths(mesh, phi, D1)                                # side 1: x -> p1.target on Gamma
    p2 = transfer_paths(mesh, phi, D2)                                # side 2
    I = pair(p1, p2)                                                  # common subdivision of Gamma
    bonus = max(6, 2 * order)
    dI = [I.ds(side=i, bonus_intorder=bonus) for i in (1, 2)]         # Gamma_h^1, Gamma_h^2

    spaces = []
    for D in (D1, D2):
        spaces += [restrict(MatrixValued(L2(mesh, order=order)), D), restrict(VectorL2(mesh, order=order), D),
                   restrict(L2(mesh, order=order, lowest_order_wb=True), D),  # element means of p stay global
                   restrict(FacetFESpace(mesh, order=order, dirichlet="outer"), D) ** 2]  # restrict(F**2) drops dirichlet
    fes = FESpace(spaces + [NumberSpace(mesh)])                       # the pressure level alpha last
    (L_1, u1, ph1, uhat1, L_2, u2, ph2, uhat2, alpha), (G1, v1, q1, mu1, G2, v2, q2, mu2, alpha_) = fes.TnT()

    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    border = LinearForm(fes)                                          # (1, q_1) + (1, q_2)
    n = specialcf.normal(2)

    # HDG of stokes.py on each side
    for L, u, ph, uhat, G, v, q, mu, D in ((L_1, u1, ph1, uhat1, G1, v1, q1, mu1, D1),
                                           (L_2, u2, ph2, uhat2, G2, v2, q2, mu2, D2)):
        dV = dx(definedonelements=D)
        dS = dx(element_boundary=True, definedonelements=D)
        div_G = fem.Einsum("ijj->i", grad(G))                         # row-wise: (div G)_i = d_j G_ij
        sigmahat_n = nu * L * n - ph * n - tau * nu * (u - uhat)
        lhs += InnerProduct(L, G) * dV + u * div_G * dV - uhat * (G * n) * dS
        lhs += nu * InnerProduct(L, grad(v)) * dV - ph * Trace(grad(v)) * dV - sigmahat_n * v * dS
        rhs += f * v * dV
        lhs += -u * grad(q) * dV + (uhat * n) * q * dS
        lhs += sigmahat_n * mu * interior_ds(mesh, D)                 # conservativity: interior faces
        border += q * dV

    # Gamma_h^1, stress:  sigmahat_1 nubar - sigmahat_2 nubar = 0  at xbar
    xbar, nubar = p1.target, p1.target_normal                         # side 1's paths end here
    s1 = (nu * L_1.Extrapolate(xbar) - ph1.Extrapolate(xbar) * Id(2)) * nubar - tau * nu * (u1 - uhat1)
    s2 = ((nu * L_2.Extrapolate(xbar) - ph2.Extrapolate(xbar) * Id(2)) * nubar
          + tau * nu * (u2.Extrapolate() - uhat2.Extrapolate()))
    lhs += (s1 - s2) * mu1 * dI[0]

    # Gamma_h^2, trace:  u_1 - u_2 = 0  at xbar
    lhs += (uhat1.Transfer(p1, gradient=L_1) - uhat2.Transfer(p2, gradient=L_2)) * mu2 * dI[1]

    # zero mean of p = ptilde + alpha over Omega = D_1 + D_2 + sliver_1 + sliver_2
    paths, ph = (p1, p2), (ph1, ph2)
    for i, D in enumerate((D1, D2)):
        lhs += alpha * alpha_ * dx(definedonelements=D)
        lhs += (gap_pressure(i, paths, ph, order) + alpha.RegionIntegral(paths[i])) * alpha_ * dI[i]

    gfu = GridFunction(fes)
    for uhat in (gfu.components[3], gfu.components[7]):
        uhat.Set(g, definedon=mesh.Boundaries("outer"))
    border.Assemble()
    inverse = multiplier_inverse(fes, border=border.vec)
    solve_linear_system(lhs, rhs, gfu, inverse=inverse)

    sol = list(gfu.components[:8])
    alpha = gfu.components[8].vec[0]
    for c in (2, 6):
        sol[c] = sol[c] + alpha
    return sol, inverse.op.rho, alpha


def square_mesh(maxh):
    """The unit square with the interface y = 1/2 as an internal line: material
    "upper" (Omega_1) above, "lower" (Omega_2) below."""
    geo = SplineGeometry()
    pts = [geo.AppendPoint(*p) for p in ((0, 0), (1, 0), (1, 0.5), (1, 1), (0, 1), (0, 0.5))]
    for a, b, left in ((0, 1, 2), (1, 2, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 0, 2)):
        geo.Append(["line", pts[a], pts[b]], leftdomain=left, rightdomain=0, bc="outer")
    geo.Append(["line", pts[5], pts[2]], leftdomain=1, rightdomain=2, bc="interface")
    geo.SetMaterial(1, "upper")
    geo.SetMaterial(2, "lower")
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    D1 = BitArray([el.mat == "upper" for el in mesh.Elements(VOL)])
    return mesh, D1, ~D1


def phi(X, Y):
    return 0.5 - Y                                                    # Omega_1 = {y > 1/2}


def disk(X, Y):
    return sqrt(X * X + Y * Y) - 1                                    # unit disk in (-1.5, 1.5)^2


def errors(pieces, sol, alpha, Lex, uex, pex):
    """Errors summed over both sides (pieces: (mesh, D) of each): L, u, p = ptilde + alpha, ptilde."""
    e = [0] * 4
    for (mesh, D), Lh, uh, ph in zip(pieces, (sol[0], sol[4]), (sol[1], sol[5]), (sol[2], sol[6])):
        dV = dx(definedonelements=D, bonus_intorder=4)
        for i, (a, b) in enumerate(((Lh, Lex), (uh, uex), (ph, pex), (ph - alpha, pex))):
            e[i] += Integrate(InnerProduct(a - b, a - b) * dV, mesh)
    return [sqrt(ei) for ei in e]


def paper_solution(nu=1.0):
    uex = CF((sin(pi * x) * sin(pi * y), cos(pi * x) * cos(pi * y)))
    pex = sin(2 * pi * x) * sin(2 * pi * y)
    return (uex, pex, *stokes_data(uex, pex, nu))


def row(family, k, maxh):
    """The errors, |rho| and alpha for the paper's solution, nu = 1."""
    uex, pex, Lex, f = paper_solution()
    if family == "flat":
        (mesh, D1, D2), geo = square_mesh(maxh), phi
    else:
        (mesh, D1, D2), geo = domains(family, maxh, disk), disk
    sol, rho, alpha = solve(mesh, k, geo, D1, D2, f, uex)
    return errors(((mesh, D1), (mesh, D2)), sol, alpha, Lex, uex, pex), abs(rho), alpha


def print_row(maxh, e, prev, rho, alpha, N=None):
    """One table row: L, u, p with EOC (log ratio of errors / log ratio of h), ptilde, |rho|, alpha."""
    r = [f"({log(a / b) / log(prev[-1] / maxh):.2f})" for a, b in zip(prev, e)] if prev else [" " * 6] * len(e)
    print(f"    h={maxh:6.4f}" + (f"  N={N:6d}" if N else "")
          + "".join(f"  {c}={ei:.3e} {ri}" for c, ei, ri in zip(("|L-Lh|", "|u-uh|", "|p-ph|", "ptilde"), e, r))
          + f"  |rho|={rho:.1e}  alpha={alpha:.2e}", flush=True)


MAXH = {"flat": {1: (0.2, 0.1, 0.05, 0.025), 2: (0.2, 0.1, 0.05, 0.025)},
        "disk": {1: (0.2, 0.141, 0.1, 0.0707, 0.05, 0.035), 2: (0.2, 0.141, 0.1, 0.0707)}}   # < 2 GB per solve


if __name__ == "__main__":
    families = sys.argv[1:] or ["flat", "fitted", "snapped", "unfitted"]
    # each row in a new process: NGSolve's umfpack inverse does not give its memory back
    with ProcessPoolExecutor(1, max_tasks_per_child=1) as pool:
        for family in families:
            print(family)
            for k in (1, 2):
                print(f"  k = {k}")
                prev = None
                for maxh in MAXH["flat" if family == "flat" else "disk"][k]:
                    e, rho, alpha = pool.submit(row, family, k, maxh).result()
                    print_row(maxh, e, prev, rho, alpha)
                    prev = e + [maxh]

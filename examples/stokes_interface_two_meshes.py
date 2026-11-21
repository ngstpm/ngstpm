"""The Stokes interface problem of stokes_interface.py on two independent meshes
(Manriquez-Nguyen-Solano 2022, Sections 5.1, 5.2, Tables 1-3):
Omega_1 = (0,1) x (1/2,1) and Omega_2 = (0,1) x (0,1/2), meshed on their own
(h2 = 1.23 h1) with a gap of width delta around y = 1/2: mesh 1 covers
(0,1) x (1/2 + delta/2, 1), mesh 2 (0,1) x (0, 1/2 - delta/2), and the
vertices of Gamma_h^1 and Gamma_h^2 do not match.

The weak form is line for line that of stokes_interface.py; only the
construction differs: each side's spaces live on its own mesh, D_i is the
whole mesh i and Gamma_h^i its boundary "interface", and the product space
and the forms come from the Interface (I.Space, I.BilinearForm, ...). The
multiplier of (ptilde_1, 1) + (ptilde_2, 1) = 0 borders the assembled matrix
(stokes.multiplier_inverse), and the pressure level alpha, whose row needs the
slivers of both sides, is an extra row added algebraically: a NumberSpace of
one mesh cannot take the other side's sliver (its test function is not of the
form's space under I.ds of the other side). alpha has no other entries in its
column, so the system is block triangular and alpha = -int_gap P / |Omega_h|,
P = (E ptilde_1 + E ptilde_2)/2 (M22, Section 5),
after the solve (stokes_interface.gap_integral).

L, u and p converge with order k+1 for delta = 0, h^2 and h/2. The flat O(h)
gap is regular here, unlike the unfitted (staircase) family of
stokes_interface.py, where the rates are irregular (D21 of TODO.md). Jaime's
note: "for O(h) gaps it is known that extrapolation may not be the best, which
makes the flux interface condition suboptimal".
"""

from concurrent.futures import ProcessPoolExecutor
import sys

from netgen.geom2d import SplineGeometry
from ngsolve import *

from poisson import solve_linear_system
from stokes import multiplier_inverse
from stokes_interface import errors, gap_integral, paper_solution, print_row
from ngstpm import interior_ds, pair, transfer_paths


def meshes(maxh1, maxh2, delta=0):
    g1 = SplineGeometry()
    g1.AddRectangle((0, 0.5 + delta / 2), (1, 1), bcs=["interface", "outer", "outer", "outer"])
    g2 = SplineGeometry()
    g2.AddRectangle((0, 0), (1, 0.5 - delta / 2), bcs=["outer", "outer", "interface", "outer"])
    return Mesh(g1.GenerateMesh(maxh=maxh1)), Mesh(g2.GenerateMesh(maxh=maxh2))


def solve(mesh1, mesh2, order, phi, f, g, nu=1.0, tau=1.0):
    """Returns (L1, u1, p1, uhat1, L2, u2, p2, uhat2), rho, alpha, with p_i = ptilde_i + alpha."""
    p1 = transfer_paths(mesh1, phi, facets=mesh1.Boundaries("interface"))   # D_1: all of mesh1
    p2 = transfer_paths(mesh2, phi, facets=mesh2.Boundaries("interface"))   # D_2: all of mesh2
    D1, D2 = p1.elements, p2.elements
    I = pair(p1, p2)                                                  # common subdivision of Gamma
    bonus = max(6, 2 * order)
    dI = [I.ds(side=i, bonus_intorder=bonus) for i in (1, 2)]

    fes1, fes2 = (MatrixValued(L2(mesh, order=order)) * VectorL2(mesh, order=order)
                  * L2(mesh, order=order, lowest_order_wb=True)           # element means of p stay global
                  * FacetFESpace(mesh, order=order, dirichlet="outer") ** 2 for mesh in (mesh1, mesh2))
    fes = I.Space(fes1, fes2)                                         # one space on two meshes
    (L_1, u1, ph1, uhat1, L_2, u2, ph2, uhat2), (G1, v1, q1, mu1, G2, v2, q2, mu2) = fes.TnT()

    lhs = I.BilinearForm(fes, condense=True)
    rhs = I.LinearForm(fes)
    border = I.LinearForm(fes)                                        # (1, q_1) + (1, q_2)
    n = specialcf.normal(2)

    # HDG of stokes.py on each side
    for L, u, ph, uhat, G, v, q, mu, D, mesh in ((L_1, u1, ph1, uhat1, G1, v1, q1, mu1, D1, mesh1),
                                                 (L_2, u2, ph2, uhat2, G2, v2, q2, mu2, D2, mesh2)):
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

    gfu = I.GridFunction(fes)
    gfu.components[3].Set(g, definedon=mesh1.Boundaries("outer"))
    gfu.components[7].Set(g, definedon=mesh2.Boundaries("outer"))
    border.Assemble()
    inverse = multiplier_inverse(fes1, border=border.vec)            # pins a pressure mean of side 1
    solve_linear_system(lhs, rhs, gfu, inverse=inverse)

    # the alpha row: zero mean of p = ptilde + alpha over Omega = D_1 + D_2 + sliver_1 + sliver_2
    measure = sum(Integrate(CF(1) * dx, mesh) + Integrate(CF(1).RegionIntegral(p) * d, mesh)
                  for mesh, p, d in ((mesh1, p1, dI[0]), (mesh2, p2, dI[1])))
    alpha = -gap_integral(I, fes, gfu, dI, (p1, p2), (ph1, ph2), (mu1, mu2), order) / measure
    sol = list(gfu.components)
    for c in (2, 6):
        sol[c] = sol[c] + alpha
    return sol, inverse.op.rho, alpha


def phi(X, Y):
    return 0.5 - Y                                                    # Omega_1 = {y > 1/2}


DELTA = {"0": lambda h: 0, "h^2": lambda h: h * h, "h/2": lambda h: h / 2}


def row(gap, k, maxh):
    """The errors, |rho|, alpha and the number of elements, for the
    paper's solution, nu = 1, with the gap DELTA[gap](maxh)."""
    uex, pex, Lex, f = paper_solution()
    mesh1, mesh2 = meshes(maxh, 1.23 * maxh, DELTA[gap](maxh))      # non-matching on Gamma
    sol, rho, alpha = solve(mesh1, mesh2, k, phi, f, uex)
    e = errors(((mesh1, None), (mesh2, None)), sol, alpha, Lex, uex, pex)
    return e, abs(rho), alpha, mesh1.ne + mesh2.ne


if __name__ == "__main__":
    # each row in a new process: NGSolve's umfpack inverse does not give its memory back
    with ProcessPoolExecutor(1, max_tasks_per_child=1) as pool:
        for gap in sys.argv[1:] or DELTA:
            print(f"delta = {gap}")
            for k in (1, 2):
                print(f"  k = {k}")
                prev = None
                for maxh in (0.2, 0.1, 0.05, 0.025):
                    e, rho, alpha, N = pool.submit(row, gap, k, maxh).result()
                    print_row(maxh, e, prev, rho, alpha, N)
                    prev = e + [maxh]

"""Stokes in Omega = {phi < 0} (L = grad u, -div(nu L - p I) = f, div u = 0,
u = g on Gamma) by HDG on an unfitted background mesh with the Transfer
Path Method. 2D.

Same HDG as stokes.py, on the interior subdomain D_h. On Gamma_h the facet
equation is the transferred Dirichlet condition

    uhat(x) + int_x^xbar (E L_h) dl = g(xbar),    xbar = x + ell sigma on Gamma.

The pressure is p_h = ptilde_h + alpha: ptilde_h has zero mean on D_h (the
multiplier rho), and the level alpha, one more unknown, makes the mean of p_h
over Omega zero (cf. Manriquez-Nguyen-Solano 2022, Section 5):

    (alpha, 1)_{D_h} + int_{Omega \\ D_h} E(ptilde_h + alpha) = 0,

the integral over the region between Gamma_h and Gamma by RegionIntegral.
"""

from ngsolve import *

from poisson import solve_linear_system
from poisson_tpm import GEOMETRIES, background_mesh
from convergence import coefficients, main
from stokes import multiplier_inverse, shifted, stokes_data
from ngstpm import boundary_ds, interior_ds, restrict, transfer_paths, unfitted_domain


def solve(mesh, order, f, g, phi, nu=1.0, tau=1.0):
    inner = unfitted_domain(mesh, phi)[0]
    p = transfer_paths(mesh, phi, inner, order=order + 4)             # x -> p.target on Gamma
    X = (x, y)

    n = specialcf.normal(2)
    dV = dx(definedonelements=inner)
    dS = dx(element_boundary=True, definedonelements=inner)
    dI = interior_ds(mesh, inner)                                     # interior facets
    dB = boundary_ds(mesh, inner, bonus_intorder=max(6, 2 * order))   # Gamma_h; g, ell not polynomial

    S = restrict(MatrixValued(L2(mesh, order=order)), inner)
    V = restrict(VectorL2(mesh, order=order), inner)
    W = restrict(L2(mesh, order=order, lowest_order_wb=True), inner)  # element means of p stay global
    M = restrict(FacetFESpace(mesh, order=order, dirichlet=".*") ** 2, inner)
    R = NumberSpace(mesh)
    A = NumberSpace(mesh)                                             # the pressure level alpha
    fes = S * V * W * M * R * A

    (L, u, ph, uhat, rho, alpha), (G, v, q, mu, rho_, alpha_) = fes.TnT()
    div_G = fem.Einsum("ijj->i", grad(G))  # row-wise: (div G)_i = d_j G_ij
    sigmahat_n = nu * L * n - ph * n - tau * nu * (u - uhat)

    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)

    lhs += InnerProduct(L, G) * dV + u * div_G * dV - uhat * (G * n) * dS

    lhs += nu * InnerProduct(L, grad(v)) * dV - ph * Trace(grad(v)) * dV - sigmahat_n * v * dS
    rhs += f * v * dV

    lhs += -u * grad(q) * dV + (uhat * n) * q * dS

    lhs += sigmahat_n * mu * dI
    lhs += uhat.Transfer(p, gradient=L) * mu * dB
    rhs += g(*p.target) * mu * dB

    lhs += ph * rho_ * dV + rho * q * dV  # (ptilde, 1)_{D_h} = 0

    # p = ptilde + alpha has zero mean on Omega: (alpha, 1)_{D_h} + int_{Omega \ D_h} E(ptilde + alpha) = 0
    lhs += alpha * alpha_ * dV + (ph + alpha).RegionIntegral(p) * alpha_ * dB

    gfu = GridFunction(fes)
    gfu.components[3].Set(g(*X), definedon=mesh.Boundaries(".*"))  # fitted part of the boundary
    solve_linear_system(lhs, rhs, gfu, inverse=multiplier_inverse(fes))

    Lh, uh, pt, uhath, rhoh, alphah = gfu.components
    alpha = alphah.vec[0]
    return (Lh, uh, shifted(pt, alpha), uhath, rhoh), inner, alpha


CASES = {name: {k: (0.2, 0.141, 0.1, 0.0707, 0.05, 0.0354) for k in (1, 2)} for name in ("disk", "flower")}


def row(case, k, maxh, solution=False):
    """Errors on D_h of L, u, p and ptilde ("unshifted"), |rho|, alpha (and the solution)
    for the paper's solution, nu = 1, on the geometry case."""
    g = lambda X, Y: CF((sin(pi * X) * sin(pi * Y), cos(pi * X) * cos(pi * Y)))
    uex = g(x, y)
    pex = sin(2 * pi * x) * sin(2 * pi * y)
    Lex, f = stokes_data(uex, pex, 1.0)
    mesh = background_mesh(maxh)
    sol, inner, alpha = solve(mesh, k, f, g, GEOMETRIES[case])
    Lh, uh, ph, _, rho = sol
    dV = dx(definedonelements=inner, bonus_intorder=4)
    e = [sqrt(Integrate(InnerProduct(a - b, a - b) * dV, mesh))
         for a, b in ((Lh, Lex), (uh, uex), (ph, pex), (ph - alpha, pex))]
    r = dict(zip(("|L-Lh|", "|u-uh|", "|p-ph|", "unshifted"), e), **{"|rho|": abs(rho.vec[0]), "alpha": alpha})
    if solution:
        r["solution"] = coefficients(sol)
    return r


if __name__ == "__main__":
    main(row, CASES, ("|L-Lh|", "|u-uh|", "|p-ph|", "unshifted"), {"|rho|": ".1e", "alpha": ".1e"}, hfmt="6.4f")

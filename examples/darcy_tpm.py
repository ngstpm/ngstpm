"""Darcy flow (darcy.py) in Omega = {phi < 0} on an unfitted background mesh with
the Transfer Path Method; as poisson_tpm.py with (q, u, uhat) -> (u, p, phat).

solve: p = g on Gamma. On Gamma_h the trace is transferred to xbar on Gamma
with grad p = -kappa^{-1} u,

    phat(x) - int_x^xbar (kappa^{-1} E u_h) . dl = g(xbar)

(Bermudez-Manriquez-Solano 2025, (3.6a)); kappa may depend on x, y, it is then
moved along the path.

solve_neumann: u . nu = gN on Gamma (as poisson_tpm_neumann.py), p = g on the
fitted boundary of the box; the flux is extrapolated to xbar,

    (E u_h)(xbar) . nu(xbar) + tau (p_h - phat)(x) = gN(xbar) . nu(xbar).

With paths of length O(h) its rates are irregular, as for poisson_tpm_neumann.py:
"for O(h) gaps it is known that extrapolation may not be the best, which makes
the flux interface condition suboptimal" (J. Manriquez).
"""

from ngsolve import *

from convergence import coefficients, main
from darcy import darcy_data
from poisson import solve_linear_system
from poisson_tpm import GEOMETRIES, background_mesh
from ngstpm import boundary_ds, interior_ds, restrict, transfer_paths, unfitted_domain


def inverse(kappa):
    """kappa^{-1} of a 2x2 matrix CF by the adjugate: PathIntegral moves kappa along the
    path with CF.Replace, which NGSolve's Inv(...) node does not support."""
    return CF((kappa[1, 1], -kappa[0, 1], -kappa[1, 0], kappa[0, 0]), dims=(2, 2)) / Det(kappa)


def solve(mesh, order, kappa, f, g, phi, tau=1.0, gN=None):
    """g, gN: callables of the coordinates (gN vector valued, its normal component at
    xbar is imposed); with gN the flux condition of solve_neumann replaces the trace."""
    inner = unfitted_domain(mesh, phi)[0]
    path = transfer_paths(mesh, phi, inner, order=order + 4)          # x -> path.target on Gamma
    xbar, nubar = path.target, path.target_normal
    X = (x, y)

    n = specialcf.normal(2)
    dV = dx(definedonelements=inner)
    dS = dx(element_boundary=True, definedonelements=inner)
    dI = interior_ds(mesh, inner)                                     # interior facets
    dB = boundary_ds(mesh, inner, bonus_intorder=max(6, 2 * order))   # Gamma_h; g, ell not polynomial

    V = restrict(VectorL2(mesh, order=order), inner)
    W = restrict(L2(mesh, order=order), inner)
    M = restrict(FacetFESpace(mesh, order=order, dirichlet=".*"), inner)
    fes = V * W * M

    (u, p, phat), (v, q, mu) = fes.TnT()
    uhat_n = u*n + tau*(p - phat)

    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)

    lhs += inverse(kappa)*u*v*dV - p*div(v)*dV + phat*(v*n)*dS

    lhs += -u*grad(q)*dV + uhat_n*q*dS
    rhs += f * q * dV

    lhs += uhat_n * mu * dI
    if gN is None:
        lhs += phat.Transfer(path, gradient=-inverse(kappa)*u) * mu * dB
        rhs += g(*xbar) * mu * dB
    else:
        lhs += (u.Extrapolate(xbar)*nubar + tau*(p - phat)) * mu * dB
        rhs += gN(*xbar) * nubar * mu * dB

    gfu = GridFunction(fes)
    gfu.components[2].Set(g(*X), definedon=mesh.Boundaries(".*"))  # fitted part of the boundary
    solve_linear_system(lhs, rhs, gfu)
    return gfu.components, inner


def solve_neumann(mesh, order, kappa, f, g, gN, phi, tau=1.0):
    return solve(mesh, order, kappa, f, g, phi, tau=tau, gN=gN)


KAPPA = {"I": CF((1, 0, 0, 1), dims=(2, 2)), "tensor": CF((2, 0.5, 0.5, 1), dims=(2, 2))}
SETUPS = {"disk": ("disk", "I"), "flower": ("flower", "I"), "disk-tensor": ("disk", "tensor"), "neumann": (None, "I")}
CASES = {case: {k: (0.2, 0.1, 0.05, 0.025) for k in (1, 2, 3)} for case in SETUPS}


def row(case, k, maxh, solution=False):
    """Errors of u, p on D_h (and the solution) for p = cos(pi x) cos(pi y)."""
    name, kappa = SETUPS[case][0], KAPPA[SETUPS[case][1]]
    pex = lambda X, Y: cos(pi * X) * cos(pi * Y)
    uex, f = darcy_data(pex(x, y), kappa)
    mesh = background_mesh(maxh)
    if name is None:                                                  # the geometry of poisson_tpm_neumann.py
        uN = lambda X, Y: -kappa * CF((-pi * sin(pi * X) * cos(pi * Y), -pi * cos(pi * X) * sin(pi * Y)))
        sol, inner = solve_neumann(mesh, k, kappa, f, pex, uN, lambda X, Y: 0.6 - sqrt(X*X + Y*Y))
    else:
        sol, inner = solve(mesh, k, kappa, f, pex, GEOMETRIES[name])
    uh, ph, _ = sol
    dV = dx(definedonelements=inner, bonus_intorder=4)
    r = {"|u-uh|": sqrt(Integrate((uh - uex) ** 2 * dV, mesh)), "|p-ph|": sqrt(Integrate((ph - pex(x, y)) ** 2 * dV, mesh))}
    if solution:
        r["solution"] = coefficients(sol)
    return r


if __name__ == "__main__":
    main(row, CASES, ("|u-uh|", "|p-ph|"))

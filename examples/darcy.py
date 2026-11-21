"""Darcy flow, u + kappa grad p = 0, div u = f in Omega, p = g on the Dirichlet
part of the boundary and u . n = gN on the rest, by HDG (Bermudez-Manriquez-
Solano 2025, (3.1g)-(3.1i), (3.2a)): the HDG of poisson.py with
(q, u, uhat) -> (u, p, phat) and the tensor kappa (symmetric positive definite,
constant or a matrix CF) in the first equation:

    (kappa^{-1} u, v) - (p, div v) + <phat, v . n> = 0,
    -(u, grad q) + <uhat . n, q> = (f, q),    uhat . n = u . n + tau (p - phat),
    <uhat . n, mu> = 0 on interior faces, <gN, mu> on Neumann faces, phat = g on Dirichlet faces.
"""

from math import log2

from netgen.geom2d import SplineGeometry
from ngsolve import *

from poisson import solve_linear_system


def solve(mesh, order, kappa, f, g, gN=None, dirichlet=".*", tau=1.0):
    """kappa: 2x2 matrix CF; g: CF on the Dirichlet faces; gN: CF (u . n) on the others."""
    n = specialcf.normal(mesh.dim)
    dS = dx(element_boundary=True)

    V = VectorL2(mesh, order=order)
    W = L2(mesh, order=order)
    M = FacetFESpace(mesh, order=order, dirichlet=dirichlet)
    fes = V * W * M

    (u, p, phat), (v, q, mu) = fes.TnT()
    uhat_n = u*n + tau*(p - phat)

    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)

    lhs += Inv(kappa)*u*v*dx - p*div(v)*dx + phat*(v*n)*dS

    lhs += -u*grad(q)*dx + uhat_n*q*dS
    rhs += f * q * dx

    lhs += uhat_n * mu * dS
    if gN is not None:
        rhs += gN * mu * ds(skeleton=True, definedon=~mesh.Boundaries(dirichlet))

    gfu = GridFunction(fes)
    gfu.components[2].Set(g, definedon=mesh.Boundaries(dirichlet))
    solve_linear_system(lhs, rhs, gfu)
    return gfu.components


def darcy_data(p, kappa):
    """u = -kappa grad p and f = div u for a pressure p and a tensor kappa (2D)."""
    u = -kappa * CF((p.Diff(x), p.Diff(y)))
    return u, u[0].Diff(x) + u[1].Diff(y)


def half_square(maxh):
    """(0,1) x (0,1/2), the Darcy domain of the paper's test case; the top side is "top"."""
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 0.5), bcs=["bottom", "right", "top", "left"])
    return Mesh(geo.GenerateMesh(maxh=maxh))


if __name__ == "__main__":
    pex = cos(pi * x) * cos(pi * y)
    n = specialcf.normal(2)
    cases = (("kappa = I, u.n = 0 on bottom/left/right, p on top", CF((1, 0, 0, 1), dims=(2, 2)), "top"),
             ("kappa = [[2, 0.5], [0.5, 1]], all Dirichlet", CF((2, 0.5, 0.5, 1), dims=(2, 2)), ".*"))
    for name, kappa, dirichlet in cases:
        print(name)
        uex, f = darcy_data(pex, kappa)
        for k in (1, 2, 3):
            print(f"  k = {k}")
            prev = None
            for maxh in (0.1, 0.05, 0.025, 0.0125):
                mesh = half_square(maxh)
                uh, ph, _ = solve(mesh, k, kappa, f, pex, uex * n, dirichlet)
                eu = sqrt(Integrate((uh - uex) ** 2, mesh, order=2 * k + 6))
                ep = sqrt(Integrate((ph - pex) ** 2, mesh, order=2 * k + 6))
                if prev:
                    ru, rp = log2(prev[0] / eu), log2(prev[1] / ep)
                    print(f"    h={maxh:6.4f}  |u-uh|={eu:.3e} ({ru:.2f})  |p-ph|={ep:.3e} ({rp:.2f})")
                else:
                    print(f"    h={maxh:6.4f}  |u-uh|={eu:.3e}         |p-ph|={ep:.3e}")
                prev = (eu, ep)

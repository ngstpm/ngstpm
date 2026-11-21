"""-Delta u = f in Omega = {phi < 0} with a Neumann condition on the unfitted
part Gamma = {phi = 0} of its boundary, q . nu = gN (q = -grad u, nu the
outward normal), and u = g on the fitted part (the boundary of the box).

As poisson_tpm.py, but the equation of uhat on Gamma_h is the flux
transferred to the end xbar of the path,

    (E q_h)(xbar) . nu(xbar) + tau (u_h - uhat)(x) = gN(xbar).

Polynomial solutions of degree k are reproduced to roundoff. With paths of
length O(h) the transferred flux is less robust than the transferred trace
of poisson_tpm.py: the errors decrease, but below order k+1.
"""

from math import log2

from ngsolve import *

from poisson import solve_linear_system
from poisson_tpm import background_mesh
from ngstpm import boundary_ds, interior_ds, restrict, transfer_paths, unfitted_domain


def solve(mesh, order, f, g, gN, phi, tau=1.0):
    inner = unfitted_domain(mesh, phi)[0]
    p = transfer_paths(mesh, phi, inner, order=order + 4)             # x -> p.target on Gamma
    xbar, nubar = p.target, p.target_normal                           # nubar: normal of Gamma at xbar

    n = specialcf.normal(mesh.dim)
    dV = dx(definedonelements=inner)
    dS = dx(element_boundary=True, definedonelements=inner)
    dI = interior_ds(mesh, inner)                                     # interior facets
    dB = boundary_ds(mesh, inner, bonus_intorder=max(6, 2 * order))   # Gamma_h

    V = restrict(VectorL2(mesh, order=order), inner)
    W = restrict(L2(mesh, order=order), inner)
    M = restrict(FacetFESpace(mesh, order=order, dirichlet=".*"), inner)
    fes = V * W * M

    (q, u, uhat), (r, v, mu) = fes.TnT()
    qhat_n = q*n + tau*(u - uhat)

    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)

    lhs += q*r*dV - u*div(r)*dV + uhat*(r*n)*dS

    lhs += -q*grad(v)*dV + qhat_n*v*dS
    rhs += f * v * dV

    lhs += qhat_n * mu * dI
    lhs += (q.Extrapolate(xbar)*nubar + tau*(u - uhat)) * mu * dB
    rhs += gN(*xbar) * nubar * mu * dB

    gfu = GridFunction(fes)
    gfu.components[2].Set(g, definedon=mesh.Boundaries(".*"))
    solve_linear_system(lhs, rhs, gfu)
    return gfu.components, inner


if __name__ == "__main__":
    phi = lambda X, Y: 0.6 - sqrt(X*X + Y*Y)                          # (-1.3, 1.3)^2 minus a disk
    uex = sin(pi * x) * cos(pi * y)
    qex = lambda X, Y: CF((-pi * cos(pi * X) * cos(pi * Y), pi * sin(pi * X) * sin(pi * Y)))
    f = 2 * pi**2 * uex

    for k in (1, 2, 3):
        print(f"k = {k}")
        prev = None
        for maxh in (0.2, 0.1, 0.05, 0.025):
            mesh = background_mesh(maxh)
            (qh, uh, _), inner = solve(mesh, k, f, uex, qex, phi)
            dV = dx(definedonelements=inner, bonus_intorder=6)
            eu = sqrt(Integrate((uh - uex) ** 2 * dV, mesh))
            eq = sqrt(Integrate((qh - qex(x, y)) ** 2 * dV, mesh))
            if prev:
                ru, rq = log2(prev[0] / eu), log2(prev[1] / eq)
                print(f"  h={maxh:5.3f}  |u-uh|={eu:.3e} ({ru:.2f})  |q-qh|={eq:.3e} ({rq:.2f})")
            else:
                print(f"  h={maxh:5.3f}  |u-uh|={eu:.3e}         |q-qh|={eq:.3e}")
            prev = (eu, eq)

"""-Delta u = f in Omega = {phi < 0}, u = g on its boundary, by the mixed
Raviart-Thomas method on an unfitted background mesh with the Transfer Path
Method (Oyarzua, Solano, Zuniga, J. Sci. Comput. 79 (2019) 49-78).

(q, u) in RT_k x P_k on the interior subdomain D_h, q = -grad u. The
Dirichlet condition is natural, <u, r.n> on Gamma_h, with the trace of u on
Gamma_h transferred from Gamma along the paths p:

    u(x) = g(xbar) + int_x^xbar (E q_h) . dl,    xbar = x + ell sigma on Gamma,

E q_h the extrapolation of q_h from the element owning x (q.PathIntegral(p)).
Where Omega reaches the boundary of the background mesh, that part of the
boundary is fitted and <g, r.n> is imposed there directly.

Same geometries and exact solution as poisson_tpm.py.
"""

from math import log2

from ngsolve import *

from poisson_tpm import GEOMETRIES, background_mesh
from ngstpm import boundary_ds, restrict, transfer_paths, unfitted_domain


def solve(mesh, order, f, g, phi):
    inner = unfitted_domain(mesh, phi)[0]
    p = transfer_paths(mesh, phi, inner, order=order + 4)             # x -> p.target on Gamma
    X = (x, y, z)[:mesh.dim]

    n = specialcf.normal(mesh.dim)
    dV = dx(definedonelements=inner)
    dB = boundary_ds(mesh, inner, bonus_intorder=max(6, 2 * order))   # Gamma_h; g, ell not polynomial

    V = restrict(HDiv(mesh, order=order, RT=True), inner)             # RT_k (NGSolve's default is BDM)
    W = restrict(L2(mesh, order=order), inner)
    fes = V * W

    (q, u), (r, v) = fes.TnT()

    lhs = BilinearForm(fes)
    rhs = LinearForm(fes)

    lhs += q*r*dV - u*div(r)*dV + div(q)*v*dV
    rhs += f * v * dV

    lhs += q.PathIntegral(p) * (r*n) * dB                             # u on Gamma_h: g(xbar) + int q . dl
    rhs += -g(*p.target) * (r*n) * dB
    rhs += -g(*X) * (r.Trace()*n) * ds                                # background-mesh boundary (not part of Gamma)

    lhs.Assemble()
    rhs.Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = lhs.mat.Inverse(fes.FreeDofs(), inverse="umfpack") * rhs.vec
    return gfu.components, inner


if __name__ == "__main__":
    g = lambda X, Y: sin(pi * X) * cos(pi * Y)
    uex = g(x, y)
    qex = CF((-pi * cos(pi * x) * cos(pi * y), pi * sin(pi * x) * sin(pi * y)))
    f = 2 * pi**2 * uex

    for name, phi in GEOMETRIES.items():
        print(name)
        for k in (0, 1, 2, 3):
            print(f"  k = {k}")
            prev = None
            for maxh in (0.2, 0.1, 0.05, 0.025):
                mesh = background_mesh(maxh)
                (qh, uh), inner = solve(mesh, k, f, g, phi)
                dV = dx(definedonelements=inner, bonus_intorder=6)
                eu = sqrt(Integrate((uh - uex) ** 2 * dV, mesh))
                eq = sqrt(Integrate((qh - qex) ** 2 * dV, mesh))
                if prev:
                    ru, rq = log2(prev[0] / eu), log2(prev[1] / eq)
                    print(f"    h={maxh:5.3f}  |u-uh|={eu:.3e} ({ru:.2f})  |q-qh|={eq:.3e} ({rq:.2f})")
                else:
                    print(f"    h={maxh:5.3f}  |u-uh|={eu:.3e}         |q-qh|={eq:.3e}")
                prev = (eu, eq)

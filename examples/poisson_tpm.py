"""-Delta u = f in Omega = {phi < 0}, u = g on its boundary, by mixed HDG on an
unfitted background mesh with the Transfer Path Method.

Same HDG as poisson.py, on the interior subdomain D_h (elements entirely
inside Omega). On its boundary Gamma_h the facet equation becomes

    uhat(x) - int_x^xbar (E q_h) . dl = g(xbar),    xbar = x + ell sigma on Gamma,

with E q_h the extrapolation of q_h from the element owning x along its
transfer path p (q = -grad u, so uhat.Transfer(p, gradient=-q)).
Where Omega reaches the boundary of the background mesh, that part of the
boundary is fitted and u = g is imposed there as in poisson.py.

Works in 2D and 3D; the geometries below are simply connected (disk, flower),
multiply connected (annulus) and partly fitted (square with a hole).
"""

from math import log2

from netgen.geom2d import SplineGeometry
from ngsolve import *

from poisson import solve_linear_system
from ngstpm import boundary_ds, interior_ds, restrict, transfer_paths, unfitted_domain


def solve(mesh, order, f, g, phi, tau=1.0):
    inner = unfitted_domain(mesh, phi)[0]
    p = transfer_paths(mesh, phi, inner, order=order + 4)             # x -> p.target on Gamma
    X = (x, y, z)[:mesh.dim]

    n = specialcf.normal(mesh.dim)
    dV = dx(definedonelements=inner)
    dS = dx(element_boundary=True, definedonelements=inner)
    dI = interior_ds(mesh, inner)                                     # interior facets
    dB = boundary_ds(mesh, inner, bonus_intorder=max(6, 2 * order))   # Gamma_h; g, ell not polynomial

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
    lhs += uhat.Transfer(p, gradient=-q) * mu * dB
    rhs += g(*p.target) * mu * dB

    gfu = GridFunction(fes)
    gfu.components[2].Set(g(*X), definedon=mesh.Boundaries(".*"))  # background-mesh boundary (not part of Gamma)
    solve_linear_system(lhs, rhs, gfu)
    return gfu.components, inner


def background_mesh(maxh, L=1.3):
    geo = SplineGeometry()
    geo.AddRectangle((-L, -L), (L, L))
    return Mesh(geo.GenerateMesh(maxh=maxh))


GEOMETRIES = {
    "disk": lambda X, Y: sqrt(X*X + Y*Y) - 1,
    "flower": lambda X, Y: sqrt(X*X + Y*Y) - 1 - 0.12 * cos(5 * atan2(Y, X)),
    "annulus": lambda X, Y: (sqrt(X*X + Y*Y) - 0.75)**2 - 0.25**2,       # 0.5 < r < 1
    "square with a hole": lambda X, Y: 0.6 - sqrt(X*X + Y*Y),            # (-1.3, 1.3)^2, r > 0.6
}


if __name__ == "__main__":
    g = lambda X, Y: sin(pi * X) * cos(pi * Y)
    uex = g(x, y)
    qex = CF((-pi * cos(pi * x) * cos(pi * y), pi * sin(pi * x) * sin(pi * y)))
    f = 2 * pi**2 * uex

    for name, phi in GEOMETRIES.items():
        print(name)
        for k in (1, 2, 3):
            print(f"  k = {k}")
            prev = None
            for maxh in (0.2, 0.1, 0.05, 0.025):
                mesh = background_mesh(maxh)
                (qh, uh, _), inner = solve(mesh, k, f, g, phi)
                dV = dx(definedonelements=inner, bonus_intorder=6)
                eu = sqrt(Integrate((uh - uex) ** 2 * dV, mesh))
                eq = sqrt(Integrate((qh - qex) ** 2 * dV, mesh))
                if prev:
                    ru, rq = log2(prev[0] / eu), log2(prev[1] / eq)
                    print(f"    h={maxh:5.3f}  |u-uh|={eu:.3e} ({ru:.2f})  |q-qh|={eq:.3e} ({rq:.2f})")
                else:
                    print(f"    h={maxh:5.3f}  |u-uh|={eu:.3e}         |q-qh|={eq:.3e}")
                prev = (eu, eq)

"""poisson_tpm.py (HDG) and poisson_tpm_RT.py (mixed Raviart-Thomas) on a
background mesh of quadrilaterals: the structured n x n grid of the same
square (-1.3, 1.3)^2, with the same geometries and exact solution.

The solvers are unchanged: unfitted_domain, transfer_paths and the facet
measures work on any element shape, and on quads the spaces are the tensor
ones (L2: Q_k; HDiv(RT=True): RT_[k]).

On squares, the HDG spaces Q_k^2 x Q_k with traces P_k do not admit an
M-decomposition and the flux loses accuracy (Cockburn-Fu, M2AN 51 (2017),
Thm. 2.1). HDG-Q enriches the flux space as in their Table 4,

    V(K) = Q_k^2 + curl span{x^(k+1) y, x y^(k+1)},   W(K) = Q_k,

(HDG-Q[k]), with x, y local coordinates of the (axis-parallel) element. The
two enrichment fields are divergence free, so only the terms with q itself
change; the transfer integrates the enriched q along the paths.
"""

from math import log2

import numpy as np
from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh

import poisson_tpm
import poisson_tpm_RT
from ngstpm import boundary_ds, interior_ds, restrict, transfer_paths, unfitted_domain
from poisson import solve_linear_system
from poisson_tpm import GEOMETRIES


def quad_mesh(maxh, L=1.3):
    n = round(2 * L / maxh)
    return MakeStructured2DMesh(quads=True, nx=n, ny=n, mapping=lambda s, t: (2 * L * s - L, 2 * L * t - L))


def enrichment(mesh, k):
    """curl(x^(k+1) y), curl(x y^(k+1)) in the local coordinates (x - x_K) / h_K of each
    element K (centre x_K, edge length h_K; axis-parallel squares)."""
    V0 = L2(mesh, order=0)
    cx, cy, h = (GridFunction(V0) for _ in range(3))
    for el in mesh.Elements(VOL):
        P = np.array([mesh.vertices[v.nr].point for v in el.vertices])
        d = V0.GetDofNrs(el)[0]
        cx.vec[d], cy.vec[d] = P.mean(axis=0)
        h.vec[d] = np.ptp(P[:, 0])
    X, Y = (x - cx) / h, (y - cy) / h
    return CF((-X**(k + 1), (k + 1) * X**k * Y)), CF((-(k + 1) * X * Y**k, Y**(k + 1)))


def solve_hdgq(mesh, order, f, g, phi, tau=1.0):
    """poisson_tpm.solve with the flux space enriched to HDG-Q[k]."""
    inner = unfitted_domain(mesh, phi)[0]
    p = transfer_paths(mesh, phi, inner, order=order + 4)
    X = (x, y, z)[:mesh.dim]

    n = specialcf.normal(mesh.dim)
    dV = dx(definedonelements=inner)
    dS = dx(element_boundary=True, definedonelements=inner)
    dI = interior_ds(mesh, inner)
    dB = boundary_ds(mesh, inner, bonus_intorder=max(6, 2 * order))

    V = restrict(VectorL2(mesh, order=order), inner)
    E = restrict(L2(mesh, order=0), inner)                            # the coefficients of the enrichment
    W = restrict(L2(mesh, order=order), inner)
    M = restrict(FacetFESpace(mesh, order=order, dirichlet=".*"), inner)
    fes = V * E * E * W * M

    (q0, a1, a2, u, uhat), (r0, b1, b2, v, mu) = fes.TnT()
    c1, c2 = enrichment(mesh, order)
    q, r = q0 + a1 * c1 + a2 * c2, r0 + b1 * c1 + b2 * c2              # div c1 = div c2 = 0
    qhat_n = q*n + tau*(u - uhat)

    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)

    lhs += q*r*dV - u*div(r0)*dV + uhat*(r*n)*dS

    lhs += -q*grad(v)*dV + qhat_n*v*dS
    rhs += f * v * dV

    lhs += qhat_n * mu * dI
    lhs += uhat.Transfer(p, gradient=-q) * mu * dB
    rhs += g(*p.target) * mu * dB

    gfu = GridFunction(fes)
    gfu.components[4].Set(g(*X), definedon=mesh.Boundaries(".*"))  # background-mesh boundary (not part of Gamma)
    solve_linear_system(lhs, rhs, gfu)
    q0h, a1h, a2h, uh, uhath = gfu.components
    return (q0h + a1h * c1 + a2h * c2, uh, uhath), inner


METHODS = {
    "HDG": poisson_tpm.solve,          # ((q_h, u_h, uhat_h), D_h)
    "HDG-Q": solve_hdgq,               # the same, flux space enriched
    "RT": poisson_tpm_RT.solve,        # ((q_h, u_h), D_h)
}


if __name__ == "__main__":
    g = lambda X, Y: sin(pi * X) * cos(pi * Y)
    uex = g(x, y)
    qex = CF((-pi * cos(pi * x) * cos(pi * y), pi * sin(pi * x) * sin(pi * y)))
    f = 2 * pi**2 * uex

    for method, solve in METHODS.items():
        for name, phi in GEOMETRIES.items():
            print(f"{method}, {name}")
            for k in (0, 1, 2, 3) if method == "RT" else (1, 2, 3):
                print(f"  k = {k}")
                prev = None
                for maxh in (0.2, 0.1, 0.05, 0.025):
                    mesh = quad_mesh(maxh)
                    (qh, uh, *_), inner = solve(mesh, k, f, g, phi)
                    dV = dx(definedonelements=inner, bonus_intorder=6)
                    eu = sqrt(Integrate((uh - uex) ** 2 * dV, mesh))
                    eq = sqrt(Integrate((qh - qex) ** 2 * dV, mesh))
                    if prev:
                        ru, rq = log2(prev[0] / eu), log2(prev[1] / eq)
                        print(f"    h={maxh:5.3f}  |u-uh|={eu:.3e} ({ru:.2f})  |q-qh|={eq:.3e} ({rq:.2f})")
                    else:
                        print(f"    h={maxh:5.3f}  |u-uh|={eu:.3e}         |q-qh|={eq:.3e}")
                    prev = (eu, eq)

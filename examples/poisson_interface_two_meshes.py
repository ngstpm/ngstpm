"""The interface problem of poisson_interface.py on two independent meshes:
Omega_1 (the unit disk) and Omega_2 (the box minus the disk), each meshed on
its own with straight elements, so Gamma_h^1 and Gamma_h^2 are two
non-matching polygons near Gamma (the paper's non-matching figure).

The weak form is line for line that of poisson_interface.py; only the
construction differs: each side's spaces live on its own mesh, D_i is the
whole mesh i and Gamma_h^i its boundary "interface", and the product space
and the forms come from the Interface (I.Space, I.BilinearForm, I.LinearForm,
I.GridFunction), since an NGSolve space lives on one mesh.
"""

from math import log2

from netgen.geom2d import SplineGeometry
from ngsolve import *

from poisson import solve_linear_system
from ngstpm import interior_ds, pair, transfer_paths


def meshes(maxh1, maxh2):
    g1 = SplineGeometry()
    g1.AddCircle((0, 0), 1, bc="interface")
    g2 = SplineGeometry()
    g2.AddRectangle((-1.5, -1.5), (1.5, 1.5), bc="outer")
    g2.AddCircle((0, 0), 1, bc="interface", leftdomain=0, rightdomain=1)
    return Mesh(g1.GenerateMesh(maxh=maxh1)), Mesh(g2.GenerateMesh(maxh=maxh2))


def solve(mesh1, mesh2, order, phi, kappa1, kappa2, f1, f2, gD, gN, g, tau=1.0, bonus=None):
    p1 = transfer_paths(mesh1, phi, facets=mesh1.Boundaries("interface"))   # D_1: all of mesh1
    p2 = transfer_paths(mesh2, phi, facets=mesh2.Boundaries("interface"))   # D_2: all of mesh2
    D1, D2 = p1.elements, p2.elements
    I = pair(p1, p2)                                                  # common subdivision of Gamma
    bonus = max(6, 2 * order) if bonus is None else bonus
    dI = [I.ds(side=i, bonus_intorder=bonus) for i in (1, 2)]

    fes1 = VectorL2(mesh1, order=order) * L2(mesh1, order=order) * FacetFESpace(mesh1, order=order)
    fes2 = (VectorL2(mesh2, order=order) * L2(mesh2, order=order)
            * FacetFESpace(mesh2, order=order, dirichlet="outer"))
    fes = I.Space(fes1, fes2)                                         # one space on two meshes
    (q1, u1, uhat1, q2, u2, uhat2), (r1, v1, mu1, r2, v2, mu2) = fes.TnT()

    lhs = I.BilinearForm(fes, condense=True)
    rhs = I.LinearForm(fes)
    n = specialcf.normal(2)

    # HDG on each side: plain NGSolve
    for q, u, uhat, r, v, mu, kappa, f, D, mesh in ((q1, u1, uhat1, r1, v1, mu1, kappa1, f1, D1, mesh1),
                                                    (q2, u2, uhat2, r2, v2, mu2, kappa2, f2, D2, mesh2)):
        dV = dx(definedonelements=D)
        dS = dx(element_boundary=True, definedonelements=D)
        qhat_n = q*n + tau*(u - uhat)
        lhs += 1/kappa(x, y)*q*r*dV - u*div(r)*dV + uhat*(r*n)*dS
        lhs += -q*grad(v)*dV + qhat_n*v*dS
        rhs += f*v*dV
        lhs += qhat_n*mu*interior_ds(mesh, D)                         # conservativity: interior faces

    # Gamma_h^1, flux:  qhat_1 . nubar - qhat_2 . nubar = gN . nubar  at xbar
    xbar, nubar = p1.target, p1.target_normal                         # side 1's paths end here
    qn1 = q1.Extrapolate(xbar)*nubar + tau*(u1 - uhat1)
    qn2 = q2.Extrapolate(xbar)*nubar - tau*(u2.Extrapolate() - uhat2.Extrapolate())
    lhs += (qn1 - qn2) * mu1 * dI[0]
    rhs += gN(*xbar) * nubar * mu1 * dI[0]

    # Gamma_h^2, trace:  u_1 - u_2 = gD  at xbar
    xbar = p2.target                                                  # side 2's paths end here
    ub1 = uhat1.Transfer(p1, gradient=-q1/kappa1(x, y))
    ub2 = uhat2.Transfer(p2, gradient=-q2/kappa2(x, y))
    lhs += (ub1 - ub2) * mu2 * dI[1]
    rhs += gD(*xbar) * mu2 * dI[1]

    gfu = I.GridFunction(fes)
    gfu.components[5].Set(g, definedon=mesh2.Boundaries("outer"))
    solve_linear_system(lhs, rhs, gfu)
    return gfu.components


if __name__ == "__main__":
    phi = lambda X, Y: sqrt(X*X + Y*Y) - 1
    kappa1, kappa2 = (lambda X, Y: CF(1.0)), (lambda X, Y: CF(10.0))
    u1 = lambda X, Y: sin(pi * X) * cos(pi * Y)
    u2 = lambda X, Y: exp(0.5 * X) * cos(0.5 * Y) + X * Y
    q1 = lambda X, Y: -kappa1(X, Y) * CF((pi * cos(pi * X) * cos(pi * Y), -pi * sin(pi * X) * sin(pi * Y)))
    q2 = lambda X, Y: -kappa2(X, Y) * CF((0.5 * exp(0.5 * X) * cos(0.5 * Y) + Y,
                                          -0.5 * exp(0.5 * X) * sin(0.5 * Y) + X))
    f1 = q1(x, y)[0].Diff(x) + q1(x, y)[1].Diff(y)
    f2 = q2(x, y)[0].Diff(x) + q2(x, y)[1].Diff(y)
    gD = lambda X, Y: u1(X, Y) - u2(X, Y)
    gN = lambda X, Y: q1(X, Y) - q2(X, Y)
    for k in (1, 2, 3):
        print(f"k = {k}")
        prev = None
        for maxh in (0.4, 0.2, 0.1):
            mesh1, mesh2 = meshes(maxh, 1.23 * maxh)                  # non-matching on Gamma
            sol = solve(mesh1, mesh2, k, phi, kappa1, kappa2, f1, f2, gD, gN, u2(x, y))
            eu = sqrt(Integrate((sol[1] - u1(x, y))**2 * dx(bonus_intorder=6), mesh1) + Integrate((sol[4] - u2(x, y))**2 * dx(bonus_intorder=6), mesh2))
            eq = sqrt(Integrate((sol[0] - q1(x, y))**2 * dx(bonus_intorder=6), mesh1) + Integrate((sol[3] - q2(x, y))**2 * dx(bonus_intorder=6), mesh2))
            rates = f"({log2(prev[0] / eu):.2f})  " if prev else "        "
            print(f"  h={maxh:5.3f}  |u-uh|={eu:.3e} {rates}|q-qh|={eq:.3e}")
            prev = (eu, eq)

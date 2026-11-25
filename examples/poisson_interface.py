"""Interface problem by mixed HDG with the Transfer Path Method:

    -div(kappa_i grad u_i) = f_i   in Omega_1 = {phi < 0}, Omega_2 = {phi > 0},
    u_1 - u_2 = gD,   (q_1 - q_2) . nu = gN   on Gamma = {phi = 0},
    u_2 = g on the boundary of the box,

with q_i = -kappa_i grad u_i (kappa_i may vary in space) and nu the normal of
Gamma from Omega_1 to Omega_2.

Each side has its own HDG unknowns (q_i, u_i, uhat_i) on a set of elements
D_h^i inside (or close to) Omega_i, with boundary Gamma_h^i towards Gamma:

    fitted     a mesh fitted to Gamma with straight elements: Gamma_h^1 and
               Gamma_h^2 coincide, at a distance O(h^2) from Gamma;
    snapped    a background mesh whose cut elements are collapsed onto Gamma
               (snapped_domains): Gamma_h^1 and Gamma_h^2 do not match and
               the gap between them is O(h^2);
    unfitted   D_h^i = the elements of a background mesh entirely inside
               Omega_i; the cut elements between them form a gap of width O(h).

The method is consistent on all three (piecewise polynomial solutions of
degree k are reproduced to roundoff). On the first two u converges with
order k+1 (q too on the fitted mesh; on the snapped one q has order k+1 for
k = 2, and 1.65, 1.86 for k = 1 and 4.20, 3.90 for k = 3); with
the O(h) gap of the unfitted mesh u and q converge with order 2 for k = 1
and about 2 for k = 2 (u: 2.2, 2.3; q: 2.2, 1.7), and irregularly for
k = 3 (u: 3.9, then 1.5), on h = 0.2, 0.1, 0.05.

Each side has its transfer path p_i = transfer_paths(mesh, phi, D_i): a point
x of Gamma_h^i is joined to xbar = p_i.target on Gamma. I = pair(p1, p2)
subdivides Gamma into segments reached from one facet of each side, so each x
has a partner x' on the facing Gamma_h^j whose own path ends at the same xbar
(the two paths may meet at xbar with a kink). Both sides move their trace
and flux to xbar,

    u_i(xbar)         ~  uhat_i.Transfer(p_i, gradient=-q_i/kappa_i)
                       =  uhat_i - int_path (E q_i)/kappa_i . dl,
    q_i(xbar) . nubar ~  q_i.Extrapolate(xbar) . nubar  (+ the stabilisation),

and the transmission conditions are imposed there: the flux condition is the
equation of uhat_1 on Gamma_h^1, the trace condition that of uhat_2 on
Gamma_h^2. Under I.ds(side=i) a trial function of side i lives on the element
K of x, one of the other side on the partner element K' of x' (evaluated at
x', or along its own path from x' to xbar).
"""

from math import log2

from netgen.geom2d import SplineGeometry
from ngsolve import *

from poisson import solve_linear_system
from ngstpm import interior_ds, pair, restrict, snapped_domains, transfer_paths, unfitted_domain


def solve(mesh, order, phi, D1, D2, kappa1, kappa2, f1, f2, gD, gN, g, tau=1.0, kind=None, bonus=None,
          condense=True):
    p1 = transfer_paths(mesh, phi, D1, kind=kind)                     # side 1: x -> p1.target on Gamma
    p2 = transfer_paths(mesh, phi, D2, kind=kind)                     # side 2
    I = pair(p1, p2)                                                  # common subdivision of Gamma
    bonus = max(6, 2 * order) if bonus is None else bonus             # ell, kappa, data: not polynomial
    dI = [I.ds(side=i, bonus_intorder=bonus) for i in (1, 2)]         # Gamma_h^1, Gamma_h^2

    V1, V2 = (restrict(VectorL2(mesh, order=order), D) for D in (D1, D2))
    W1, W2 = (restrict(L2(mesh, order=order), D) for D in (D1, D2))
    M1, M2 = (restrict(FacetFESpace(mesh, order=order, dirichlet="outer"), D) for D in (D1, D2))
    fes = V1 * W1 * M1 * V2 * W2 * M2
    (q1, u1, uhat1, q2, u2, uhat2), (r1, v1, mu1, r2, v2, mu2) = fes.TnT()

    lhs = BilinearForm(fes, condense=condense)
    rhs = LinearForm(fes)
    n = specialcf.normal(mesh.dim)
    X = (x, y, z)[:mesh.dim]

    # HDG on each side: plain NGSolve
    for q, u, uhat, r, v, mu, kappa, f, D in ((q1, u1, uhat1, r1, v1, mu1, kappa1, f1, D1),
                                              (q2, u2, uhat2, r2, v2, mu2, kappa2, f2, D2)):
        dV = dx(definedonelements=D)
        dS = dx(element_boundary=True, definedonelements=D)
        qhat_n = q*n + tau*(u - uhat)
        lhs += 1/kappa(*X)*q*r*dV - u*div(r)*dV + uhat*(r*n)*dS
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
    ub1 = uhat1.Transfer(p1, gradient=-q1/kappa1(*X))
    ub2 = uhat2.Transfer(p2, gradient=-q2/kappa2(*X))
    lhs += (ub1 - ub2) * mu2 * dI[1]
    rhs += gD(*xbar) * mu2 * dI[1]

    gfu = GridFunction(fes)
    for uhat in (gfu.components[2], gfu.components[5]):
        uhat.Set(g, definedon=mesh.Boundaries("outer"))
    solve_linear_system(lhs, rhs, gfu)
    return gfu.components


def background_mesh(maxh, L=1.5):
    geo = SplineGeometry()
    geo.AddRectangle((-L, -L), (L, L), bc="outer")
    return Mesh(geo.GenerateMesh(maxh=maxh))


def fitted_mesh(maxh, R=1.0, L=1.5):
    """Box with the disk of radius R meshed as its own subdomain; the elements
    stay straight (no mesh.Curve), so the interface of the mesh is a polygon."""
    geo = SplineGeometry()
    geo.AddRectangle((-L, -L), (L, L), bc="outer", leftdomain=2, rightdomain=0)
    geo.AddCircle((0, 0), R, bc="interface", leftdomain=1, rightdomain=2)
    return Mesh(geo.GenerateMesh(maxh=maxh))


def domains(family, maxh, phi):
    """mesh, D1, D2 of the given mesh family (phi: the unit disk for "fitted")."""
    if family == "fitted":
        mesh = fitted_mesh(maxh)
        D1 = BitArray([el.index == 0 for el in mesh.Elements(VOL)])   # domain 1: the disk
        return mesh, D1, ~D1
    mesh = background_mesh(maxh)
    if family == "snapped":
        return snapped_domains(mesh, phi)
    return mesh, unfitted_domain(mesh, phi)[0], unfitted_domain(mesh, phi, side=2)[0]


def errors(mesh, sol, D1, D2, u1, u2, q1, q2):
    eu = eq = 0
    for D, uh, qh, ue, qe in ((D1, sol[1], sol[0], u1, q1), (D2, sol[4], sol[3], u2, q2)):
        dV = dx(definedonelements=D, bonus_intorder=6)
        eu += Integrate((uh - ue(x, y)) ** 2 * dV, mesh)
        eq += Integrate((qh - qe(x, y)) ** 2 * dV, mesh)
    return sqrt(eu), sqrt(eq)


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else None             # or "facet-normal" (legacy)
    phi = lambda X, Y: sqrt(X*X + Y*Y) - 1                            # unit disk in (-1.5, 1.5)^2
    kappa1 = lambda X, Y: CF(1.0)                                     # constant here; any CF of (X, Y) works
    kappa2 = lambda X, Y: CF(10.0)
    u1 = lambda X, Y: sin(pi * X) * cos(pi * Y)
    u2 = lambda X, Y: exp(0.5 * X) * cos(0.5 * Y) + X * Y
    q1 = lambda X, Y: -kappa1(X, Y) * CF((pi * cos(pi * X) * cos(pi * Y), -pi * sin(pi * X) * sin(pi * Y)))
    q2 = lambda X, Y: -kappa2(X, Y) * CF((0.5 * exp(0.5 * X) * cos(0.5 * Y) + Y,
                                          -0.5 * exp(0.5 * X) * sin(0.5 * Y) + X))
    f1 = q1(x, y)[0].Diff(x) + q1(x, y)[1].Diff(y)                    # f = div q
    f2 = q2(x, y)[0].Diff(x) + q2(x, y)[1].Diff(y)
    gD = lambda X, Y: u1(X, Y) - u2(X, Y)
    gN = lambda X, Y: q1(X, Y) - q2(X, Y)

    for family in ("fitted", "snapped", "unfitted"):
        print(family)
        for k in (1, 2, 3):
            print(f"  k = {k}")
            prev = None
            for maxh in (0.2, 0.1, 0.05, 0.025):
                mesh, D1, D2 = domains(family, maxh, phi)
                sol = solve(mesh, k, phi, D1, D2, kappa1, kappa2, f1, f2, gD, gN, u2(x, y), kind=kind)
                eu, eq = errors(mesh, sol, D1, D2, u1, u2, q1, q2)
                if prev:
                    ru, rq = log2(prev[0] / eu), log2(prev[1] / eq)
                    print(f"    h={maxh:5.3f}  |u-uh|={eu:.3e} ({ru:.2f})  |q-qh|={eq:.3e} ({rq:.2f})")
                else:
                    print(f"    h={maxh:5.3f}  |u-uh|={eu:.3e}         |q-qh|={eq:.3e}")
                prev = (eu, eq)

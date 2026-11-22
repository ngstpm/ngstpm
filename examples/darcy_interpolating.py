"""Darcy flow (darcy.py) on interpolating meshes: Omega_h is bounded by the polygon
through points of Gamma (straight edges, vertices on Gamma, no mesh.Curve), so the
gap between Gamma_h and Gamma is O(h^2). The cases mirror the carved ones of
darcy_tpm.py (same exact solution p = cos(pi x) cos(pi y), same kappa):

    disk, flower, disk-tensor   p = g on Gamma, Omega_h inside the polygon;
    neumann                     u . nu = gN . nu on the circle r = 0.6, p = g on the
                                box (-1.3, 1.3)^2, Omega_h the box minus the polygon.

Two variants per case (the suffix of the case key):

    -tpm    the conditions of darcy_tpm.solve / solve_neumann, transferred from
            Gamma_h to Gamma along closest-point paths;
    -closest the data of Gamma taken at the closest point xbar and imposed at x, with no
            transfer (phat(x) = g(xbar); u . n + tau (p - phat) = gN(xbar) . nu(xbar)):
            the O(h^2) geometric error left uncorrected;
    -naive  the data imposed on Gamma_h as is: phat = g(x) (the Dirichlet values,
            as on a fitted boundary), or u . n + tau (p - phat) = gN(x) . n with n
            the normal of Gamma_h.
            Here g, gN are the exact fields, defined off Gamma too, so -naive is
            consistent (no geometric error); -closest is the test of the gap.
"""

import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import *

from convergence import coefficients, main
from darcy import darcy_data
from darcy_tpm import KAPPA, inverse
from poisson import solve_linear_system
from ngstpm import ParametricCurve, boundary_ds, interior_ds, transfer_paths

L = 1.3                                                               # the box of darcy_tpm.py


def circle_xy(t, R=1.0):
    return R * np.cos(t), R * np.sin(t)


def flower_xy(t):                                                     # poisson_tpm.GEOMETRIES["flower"] = 0
    r = 1 + 0.12 * np.cos(5 * t)
    return r * np.cos(t), r * np.sin(t)


CURVES = {"disk": circle_xy, "flower": flower_xy, "neumann": lambda t: circle_xy(t, 0.6)}
RADIUS = {"disk": lambda t: 1.0 + 0 * t, "flower": lambda t: 1 + 0.12 * np.cos(5 * t), "neumann": lambda t: 0.6 + 0 * t}


def polygon(gamma, maxh, ratio=0.7):
    """Points gamma(t_i), t_i uniform in [0, 2 pi), counterclockwise, with every chord
    at most ratio * maxh (netgen then does not split the segments)."""
    n = 8
    while True:
        P = np.column_stack(gamma(2 * np.pi * np.arange(n) / n))
        if np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).max() <= ratio * maxh:
            return P
        n = int(np.ceil(1.2 * n))


def interpolating_mesh(case, maxh):
    """The mesh and its polygon: inside it (disk, flower), or the box minus it
    (neumann; box boundary "box"). The polygon's segments are "gamma"."""
    P = polygon(CURVES[case], maxh)
    geo = SplineGeometry()
    ids = [geo.AppendPoint(*p) for p in P]
    hole = case == "neumann"                                          # then ds("gamma")'s normal points into Omega_h; use boundary_ds
    for i in range(len(ids)):                                         # counterclockwise: the disk on the left
        geo.Append(["line", ids[i], ids[(i + 1) % len(ids)]], leftdomain=0 if hole else 1,
                   rightdomain=1 if hole else 0, bc="gamma")
    if hole:
        geo.AddRectangle((-L, -L), (L, L), bc="box", leftdomain=1, rightdomain=0)
    return Mesh(geo.GenerateMesh(maxh=maxh)), P


def check_polygon(mesh, P, geometry, tol=1e-12):
    """The vertices of "gamma" are exactly the polygon points (netgen inserted none, so
    Gamma_h is the interpolant through them), and they lie on Gamma (r = R(theta), the
    polar form of the curve); returns the largest |r - R(theta)| of a vertex."""
    V = np.array([v.point for v in mesh.vertices])
    on = sorted({v.nr for el in mesh.Elements(BND) if el.mat == "gamma" for v in el.vertices})
    d = np.array([np.linalg.norm(P - V[i], axis=1).min() for i in on])
    Vg = V[on]
    gap = np.abs(np.linalg.norm(Vg, axis=1) - RADIUS[geometry](np.arctan2(Vg[:, 1], Vg[:, 0]))).max()
    assert len(on) == len(P) and d.max() < tol and gap < tol, (len(on), len(P), d.max(), gap)
    return gap


def solve(mesh, order, kappa, f, g, curve=None, gN=None, tau=1.0, transfer=True):
    """curve: the ParametricCurve of Gamma (variant -tpm), or None (-naive); with
    transfer=False (-closest) the data at xbar are imposed at x without a transfer. g, gN:
    callables of the coordinates; with gN, the flux condition on "gamma" and p = g
    on "box", else p = g on "gamma"."""
    X = (x, y)
    n = specialcf.normal(2)
    dV = dx
    dS = dx(element_boundary=True)
    dI = interior_ds(mesh)
    dB = boundary_ds(mesh, facets=mesh.Boundaries("gamma"), bonus_intorder=max(6, 2 * order))

    transferred = curve is not None or gN is not None
    V = VectorL2(mesh, order=order)
    W = L2(mesh, order=order)
    M = FacetFESpace(mesh, order=order, dirichlet="box" if transferred else ".*")
    fes = V * W * M

    (u, p, phat), (v, q, mu) = fes.TnT()
    uhat_n = u*n + tau*(p - phat)

    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    lhs += inverse(kappa)*u*v*dV - p*div(v)*dV + phat*(v*n)*dS
    lhs += -u*grad(q)*dV + uhat_n*q*dS
    rhs += f * q * dV
    lhs += uhat_n * mu * dI

    if curve is not None:
        path = transfer_paths(mesh, curve, kind="closest", order=order + 4, facets=mesh.Boundaries("gamma"))
        xbar, nubar = path.target, path.target_normal
        if not transfer:                                              # -closest: the data of Gamma, unmoved
            lhs += (phat if gN is None else uhat_n) * mu * dB
            rhs += (g(*xbar) if gN is None else gN(*xbar) * nubar) * mu * dB
        elif gN is None:
            lhs += phat.Transfer(path, gradient=-inverse(kappa)*u) * mu * dB
            rhs += g(*xbar) * mu * dB
        else:                                                         # nubar must point out of Omega_h
            assert Integrate(nubar * n * dB, mesh) > 0.9 * Integrate(CF(1) * dB, mesh)
            lhs += (u.Extrapolate(xbar)*nubar + tau*(p - phat)) * mu * dB
            rhs += gN(*xbar) * nubar * mu * dB
    elif gN is not None:
        lhs += uhat_n * mu * dB
        rhs += gN(*X) * n * mu * dB

    gfu = GridFunction(fes)
    gfu.components[2].Set(g(*X), definedon=mesh.Boundaries(".*"))   # the Dirichlet part (only its dofs count)
    solve_linear_system(lhs, rhs, gfu)
    return gfu.components


SETUPS = {"disk": ("disk", "I"), "flower": ("flower", "I"), "disk-tensor": ("disk", "tensor"), "neumann": ("neumann", "I")}
CASES = {f"interpolating-{case}-{variant}": {k: (0.2, 0.1, 0.05, 0.025) for k in (1, 2, 3)}
         for case in SETUPS for variant in ("tpm", "naive", "closest")}


def row(case, k, maxh, solution=False, pex=None):
    """Errors of u, p on Omega_h (and the solution) for p = cos(pi x) cos(pi y), or the
    pressure pex (a callable of the coordinates)."""
    name, variant = case[len("interpolating-"):].rsplit("-", 1)
    geometry, kappa = SETUPS[name][0], KAPPA[SETUPS[name][1]]
    pex = pex or (lambda X, Y: cos(pi * X) * cos(pi * Y))
    uex, f = darcy_data(pex(x, y), kappa)
    mesh, P = interpolating_mesh(geometry, maxh)
    gap = check_polygon(mesh, P, geometry)
    curve = ParametricCurve(CURVES[geometry]) if variant in ("tpm", "closest") else None
    transfer = variant != "closest"
    if geometry == "neumann":
        curve = curve and -curve                                      # the normal out of Omega_h: into the disk
        uN = lambda X, Y: -kappa * CF((-pi * sin(pi * X) * cos(pi * Y), -pi * cos(pi * X) * sin(pi * Y)))
        sol = solve(mesh, k, kappa, f, pex, curve, gN=uN, transfer=transfer)
    else:
        sol = solve(mesh, k, kappa, f, pex, curve, transfer=transfer)
    uh, ph, _ = sol
    dV = dx(bonus_intorder=4)
    r = {"|u-uh|": sqrt(Integrate((uh - uex) ** 2 * dV, mesh)), "|p-ph|": sqrt(Integrate((ph - pex(x, y)) ** 2 * dV, mesh)),
         "vertex-gap": gap}
    if solution:
        r["solution"] = coefficients(sol)
    return r


if __name__ == "__main__":
    main(row, CASES, ("|u-uh|", "|p-ph|"))

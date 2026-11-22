"""Darcy flow (darcy.py) on snapped meshes: the background mesh of darcy_tpm.py with
the vertices of the cut elements moved onto Gamma (ngstpm.snapped_domains), Omega_h
= D_h the elements that were entirely on the side of Omega; the gap between Gamma_h
and Gamma is O(h^2). Cases and variants as darcy_interpolating.py:

    disk, flower, disk-tensor   p = g on Gamma;
    neumann                     u . nu = gN . nu on the circle r = 0.6, p = g on the box;

    -tpm    the conditions of darcy_tpm.solve / solve_neumann along closest-point paths;
    -closest the data of Gamma taken at the closest point xbar and imposed at x, with no
            transfer (phat(x) = g(xbar); u . n + tau (p - phat) = gN(xbar) . nu(xbar)):
            the O(h^2) geometric error left uncorrected;
    -naive  the data imposed on Gamma_h as is (phat = g(x) weakly on the faces of
            Gamma_h, or u . n + tau (p - phat) = gN(x) . n, n the normal of Gamma_h).
            Here g, gN are the exact fields, defined off Gamma too, so -naive is
            consistent (no geometric error); -closest is the test of the gap.
"""

import numpy as np
from ngsolve import *

from convergence import coefficients, main
from darcy import darcy_data
from darcy_interpolating import CURVES, RADIUS
from darcy_tpm import KAPPA, inverse
from poisson import solve_linear_system
from poisson_tpm import GEOMETRIES, background_mesh
from ngstpm import (ParametricCurve, boundary_ds, boundary_facets, interior_ds, restrict, snapped_domains,
                    transfer_paths)

PHI = {"disk": GEOMETRIES["disk"], "flower": GEOMETRIES["flower"], "neumann": lambda X, Y: 0.6 - sqrt(X*X + Y*Y)}


def snapped_mesh(geometry, maxh):
    """The snapped mesh, D_h (side 1, {phi < 0}) and the checks: the shortest face of
    Gamma_h and smallest element of D_h (relative to maxh, maxh^2), the largest distance
    of a vertex of Gamma_h to Gamma, and how far the box vertices moved."""
    mesh0 = background_mesh(maxh)
    mesh, D, _ = snapped_domains(mesh0, PHI[geometry])
    V0 = np.array([v.point for v in mesh0.vertices])
    V = np.array([v.point for v in mesh.vertices])
    box = sorted({v.nr for el in mesh0.Elements(BND) for v in el.vertices})
    F = boundary_facets(mesh, D)[0]
    lengths = [np.linalg.norm(np.subtract(*V[[v.nr for v in mesh[f].vertices]])) for f in mesh.facets if F[f.nr]]
    areas = []
    for el in mesh.Elements(VOL):
        if D[el.nr]:
            a, b, c = V[[v.nr for v in el.vertices]]
            areas.append(0.5 * abs((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0]))
    on = sorted({v.nr for f in mesh.facets if F[f.nr] for v in mesh[f].vertices})
    Vg = V[on]
    curve = np.linalg.norm(Vg, axis=1) - RADIUS[geometry](np.arctan2(Vg[:, 1], Vg[:, 0]))     # r - R(theta)
    checks = {"min-face/h": min(lengths) / maxh, "min-area/h2": min(areas) / maxh**2,
              "max|r-R|(Gamma_h vertices)": np.abs(curve).max(), "box-moved": np.abs(V[box] - V0[box]).max()}
    return mesh, D, checks


def solve(mesh, D, order, kappa, f, g, curve=None, gN=None, tau=1.0, transfer=True):
    """curve: the ParametricCurve of Gamma (-tpm), or None (-naive); with transfer=False
    (-closest) the data at xbar are imposed at x without a transfer."""
    X = (x, y)
    n = specialcf.normal(2)
    dV = dx(definedonelements=D)
    dS = dx(element_boundary=True, definedonelements=D)
    dI = interior_ds(mesh, D)
    dB = boundary_ds(mesh, D, bonus_intorder=max(6, 2 * order))

    V = restrict(VectorL2(mesh, order=order), D)
    W = restrict(L2(mesh, order=order), D)
    M = restrict(FacetFESpace(mesh, order=order, dirichlet=".*"), D)  # the box, if D touches it
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
        path = transfer_paths(mesh, curve, D, kind="closest", order=order + 4)
        xbar, nubar = path.target, path.target_normal
        if not transfer:                                              # -closest: the data of Gamma, unmoved
            lhs += (phat if gN is None else uhat_n) * mu * dB
            rhs += (g(*xbar) if gN is None else gN(*xbar) * nubar) * mu * dB
        elif gN is None:
            lhs += phat.Transfer(path, gradient=-inverse(kappa)*u) * mu * dB
            rhs += g(*xbar) * mu * dB
        else:                                                         # nubar must point out of D_h
            assert Integrate(nubar * n * dB, mesh) > 0.9 * Integrate(CF(1) * dB, mesh)
            lhs += (u.Extrapolate(xbar)*nubar + tau*(p - phat)) * mu * dB
            rhs += gN(*xbar) * nubar * mu * dB
    elif gN is None:
        lhs += phat * mu * dB
        rhs += g(*X) * mu * dB
    else:
        lhs += uhat_n * mu * dB
        rhs += gN(*X) * n * mu * dB

    gfu = GridFunction(fes)
    gfu.components[2].Set(g(*X), definedon=mesh.Boundaries(".*"))   # the box
    solve_linear_system(lhs, rhs, gfu)
    return gfu.components


SETUPS = {"disk": ("disk", "I"), "flower": ("flower", "I"), "disk-tensor": ("disk", "tensor"), "neumann": ("neumann", "I")}
CASES = {f"snapped-{case}-{variant}": {k: (0.2, 0.1, 0.05, 0.025) for k in (1, 2, 3)}
         for case in SETUPS for variant in ("tpm", "naive", "closest")}


def row(case, k, maxh, solution=False, pex=None):
    """Errors of u, p on D_h (and the solution) for p = cos(pi x) cos(pi y), or pex; the mesh checks."""
    name, variant = case[len("snapped-"):].rsplit("-", 1)
    geometry, kappa = SETUPS[name][0], KAPPA[SETUPS[name][1]]
    pex = pex or (lambda X, Y: cos(pi * X) * cos(pi * Y))
    uex, f = darcy_data(pex(x, y), kappa)
    mesh, D, checks = snapped_mesh(geometry, maxh)
    curve = ParametricCurve(CURVES[geometry]) if variant in ("tpm", "closest") else None
    transfer = variant != "closest"
    if geometry == "neumann":
        curve = curve and -curve                                      # the normal out of D_h: into the disk
        uN = lambda X, Y: -kappa * CF((-pi * sin(pi * X) * cos(pi * Y), -pi * cos(pi * X) * sin(pi * Y)))
        sol = solve(mesh, D, k, kappa, f, pex, curve, gN=uN, transfer=transfer)
    else:
        sol = solve(mesh, D, k, kappa, f, pex, curve, transfer=transfer)
    uh, ph, _ = sol
    dV = dx(definedonelements=D, bonus_intorder=4)
    r = {"|u-uh|": sqrt(Integrate((uh - uex) ** 2 * dV, mesh)), "|p-ph|": sqrt(Integrate((ph - pex(x, y)) ** 2 * dV, mesh)),
         **checks}
    if solution:
        r["solution"] = coefficients(sol)
    return r


if __name__ == "__main__":
    main(row, CASES, ("|u-uh|", "|p-ph|"), {"min-face/h": ".2f", "min-area/h2": ".3f"})

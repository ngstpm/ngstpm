"""Stokes-Darcy (stokes_darcy.py) across the curved interface Gamma: y = 0.5 + 0.05 sin(2 pi x)
of the unit square (the curve of MATLAB's experiments/+BMS2026/+parameters/get_interface.m:22),
Stokes above, Darcy below.

Fitted family (MATLAB's fitted_split): two independent meshes, each with a polygonal boundary
along Gamma whose vertices lie on Gamma (straight elements), with the same interface vertices
("fitted-matching") or with the Darcy vertices 1.23 times sparser ("fitted-nonmatching"). The
two polygons leave a gap of width O(h^2) with overlaps; the paths of each side end on Gamma.

Snapped family ("snapped"): one netgen mesh of the unit square snapped onto Gamma
(snapped_domains; the vertices on the sides x = 0, 1 slide along them onto Gamma), Stokes on
D1, Darcy on D2, both sides' unknowns on the one mesh (solve with mesh1 is mesh2); the slivers
between D1 and D2 leave a gap of width O(h^2) with overlaps; errors are integrated over D1, D2.
On maxh 0.2, 0.1, 0.05, 0.025 all of L, u_s, P_s, u_d, p_d converge with order k+1 (last EOC:
2.08-2.16 for k = 1, 3.20-3.37 for k = 2, 4.21-4.48 for k = 3); ptilde_s alone has order 2 for
k = 3 (the level alpha carries the mean, as on the fitted family).

Unfitted family ("unfitted"): the same netgen square, D1 and D2 the elements entirely in Omega_s
and in Omega_d (unfitted_domain), a staircase gap of width O(h). The paths are vertical
(vertical_paths, as MATLAB's SmoothCurveHorizontal mapping="vertical"; the fitted and snapped
families keep the library's quasi-normal paths, which differ by O(h^2) there: snapped, k = 2,
maxh 0.05 and 0.025, agree to 3 digits). "unfitted-noteeth" also drops the elements with two
facets on Gamma_h (prune_teeth, as MATLAB); "unfitted-noteeth-dirichlet" has p_d = data on the
bottom (no pressure level). Polynomials of degree k are reproduced to roundoff (k = 1, 2, 3, also
on the meshes where the errors jump, with and without the tooth): no consistency defect was found.
On maxh 0.2, 0.141, 0.1, 0.0707, 0.05, 0.0354, 0.025 the rates are irregular; least-squares slopes
("unfitted-noteeth") about 2, 2.6-3.5, 3.7-4.1 for k = 1, 2, 3.
- The one tooth of the maxh 0.05 mesh (k = 1: e(L) 9 times larger) gives a local error in L_s, P_s
  near it and an offset of p_d; pruned, the jump is gone.
- Elsewhere (k = 1, 2 at maxh 0.141; p_d at 0.0354) the jump is mostly in p_d, as a constant over
  D2: p_d minus its mean converges about as k+1. This offset belongs to the pressure level of the
  all-Neumann Darcy side: with the Dirichlet bottom, p_d decreases monotonically (slope 1.9 for
  k = 1, 3.1 for k = 2) and the jump at maxh 0.141 moves into P_s. The pressure jump across the O(h)
  gap is poorly determined.
- The Poisson and Stokes interface examples on the same meshes and paths are also irregular
  (reports/agentA-tables/curved_unfitted_comparison.txt), on partly the same meshes but not field
  by field: e.g. at maxh 0.141 the Stokes interface is regular for k = 1 and jumps for k = 3, where
  Stokes-Darcy does not. The remaining irregularity is D21 (flux transfer over O(h) staircase paths).

The exact solution is the paper's (stokes_darcy.paper_data), with P_s shifted to zero mean over
the curved Omega_s = {Gamma(x) < y < 1} (|Omega_s| = 1/2), as MATLAB's
+StokesDarcyCoupling/get_manufactured_solution.m:54-57 does; T_mass, T_force use the normal of
the level set and are evaluated at xbar on Gamma.
"""


import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import *

from convergence import coefficients, main
from stokes_darcy import errors, norm, solve, stokes_darcy_data
from ngstpm import boundary_facets, snapped_domains, transfer_paths, unfitted_domain


def gamma(X):
    return 0.5 + 0.05 * np.sin(2 * np.pi * X)


def phi(X, Y):
    return 0.5 + 0.05 * sin(2 * pi * X) - Y                          # Omega_1 = Stokes = {phi < 0}


def stokes_mean(f, n=60):
    """The mean of f(x, y) (numpy) over Omega_s = {gamma(x) < y < 1}, |Omega_s| = 1/2, by
    Gauss-Legendre in x and in y on [gamma(x), 1]."""
    t, w = np.polynomial.legendre.leggauss(n)
    X, wx = 0.5 * (t + 1), 0.5 * w
    lo = gamma(X)
    Y = lo[:, None] + (1 - lo)[:, None] * 0.5 * (t[None, :] + 1)
    wy = (1 - lo)[:, None] * 0.5 * w[None, :]
    return np.sum(wx[:, None] * wy * f(X[:, None], Y)) / 0.5


def meshes(maxh, nonmatching=False):
    """Meshes of the Stokes part (above the polygon through (x_i, gamma(x_i))) and of the Darcy
    part (below its own polygon); boundaries "interface", "outer" (and "bottom" of the Darcy mesh).
    The interface segments are shorter than maxh, so netgen keeps their vertices on Gamma."""
    out = []
    for side, h in ((1, maxh), (2, 1.23 * maxh if nonmatching else maxh)):
        X = np.linspace(0, 1, int(np.ceil(1.1 / h)) + 1)
        curve = [(xi, gamma(xi)) for xi in X]
        geo = SplineGeometry()
        if side == 1:   # counterclockwise: along the interface, then up the right side, the top, down the left side
            pts = curve + [(1, 1), (0, 1)]
            bcs = ["interface"] * (len(curve) - 1) + ["outer"] * 3
        else:           # the bottom, up the right side, back along the interface, down the left side
            pts = [(0, 0), (1, 0)] + curve[::-1]
            bcs = ["bottom", "outer"] + ["interface"] * (len(curve) - 1) + ["outer"]
        ids = [geo.AppendPoint(*p) for p in pts]
        for i, bc in enumerate(bcs):
            geo.Append(["line", ids[i], ids[(i + 1) % len(ids)]], leftdomain=1, rightdomain=0, bc=bc)
        out.append(Mesh(geo.GenerateMesh(maxh=h)))
    return tuple(out)


def square(maxh):
    """A netgen mesh of the unit square, boundaries "bottom" and "outer" (the other sides)."""
    geo = SplineGeometry()
    ids = [geo.AppendPoint(*p) for p in ((0, 0), (1, 0), (1, 1), (0, 1))]
    for i, bc in enumerate(("bottom", "outer", "outer", "outer")):
        geo.Append(["line", ids[i], ids[(i + 1) % 4]], leftdomain=1, rightdomain=0, bc=bc)
    return Mesh(geo.GenerateMesh(maxh=maxh))


def snapped(maxh):
    """square(maxh) snapped onto Gamma: (mesh, D1, D2), D1 the Stokes elements (above), D2 the
    Darcy ones."""
    return snapped_domains(square(maxh), phi)


def unfitted(maxh, teeth=True):
    """square(maxh) and the elements entirely in Omega_s (D1) and entirely in Omega_d (D2); with
    teeth=False, without the elements of D_i with two or more facets on Gamma_h (prune_teeth)."""
    mesh = square(maxh)
    D1, D2 = unfitted_domain(mesh, phi)[0], unfitted_domain(mesh, phi, side=2)[0]
    return (mesh, D1, D2) if teeth else (mesh, prune_teeth(mesh, D1), prune_teeth(mesh, D2))


def prune_teeth(mesh, D):
    """D without its "teeth", the elements with two or more facets on Gamma_h, until there are none:
    MATLAB's prune_hanging_elements (src/mesh-generation/simplicial-grid/@SimplicialGrid/
    prune_hanging_elements.m), on by default for its unfitted ("discard") meshes (split.m:99-102).
    MATLAB's other filter there, prune_infeasible_elements (facets deviating more than 45 degrees
    from the normal of Gamma), is not applied (in an uncommitted trial run it dropped up to 28
    elements per mesh and did not make the rates more regular)."""
    D = BitArray(D)
    while True:
        gamma_h = boundary_facets(mesh, D)[0]
        teeth = [el.nr for el in mesh.Elements(VOL) if D[el.nr] and sum(gamma_h[f.nr] for f in el.facets) >= 2]
        if not teeth:
            return D
        for i in teeth:
            D[i] = False


def vertical_paths(mesh, geo, D=None, **kw):
    """transfer_paths(..., kind="vertical"): vertical paths, as MATLAB's SmoothCurveHorizontal with its
    default mapping="vertical" (src/dissimilar-coupling/geometry/curves/SmoothCurveHorizontal.m:9-10,
    20, 58-59), the target normal that of Gamma at xbar (SmoothCurveHorizontal.m:89-94). The
    quasi-normal paths do not work on the unfitted family: there the staircases of the two sides
    reach x = 0, 1 at different heights, the tilted paths of their end vertices reach Gamma on
    different sides of x = 0 (or 1), and pair rejects the unpaired end piece."""
    return transfer_paths(mesh, geo, D, kind="vertical", **kw)


def curved_data():
    """The paper's solution with P_s of zero mean over the curved Omega_s."""
    c = stokes_mean(lambda X, Y: np.exp(-X - Y))
    P_s = exp(-x - y) - c
    u_s = CF((pi * cos(pi * x) * sin(pi * y), -pi * cos(pi * y) * sin(pi * x)))
    p_d = cos(pi * x) * cos(pi * y)
    return stokes_darcy_data(u_s, P_s, p_d, phi)


# k = 3, maxh 0.025: 1.1 GB (snapped: 0.89 GB, unfitted: 0.77 GB). fitted-nonmatching starts at 0.1: at 0.2, pair rejects a piece of length 3e-11
# at the end x = 0 of Gamma, where the two sides' paths end 3e-11 apart (default path order 6).
MAXH = {"fitted-matching": (0.2, 0.1, 0.05, 0.025), "fitted-nonmatching": (0.1, 0.0707, 0.05, 0.0354, 0.025),
        "snapped": (0.2, 0.1, 0.05, 0.025),
        "unfitted": (0.2, 0.141, 0.1, 0.0707, 0.05, 0.0354, 0.025), "unfitted-noteeth": (0.2, 0.141, 0.1, 0.0707, 0.05, 0.0354, 0.025)}
CASES = {case: {k: hs for k in (1, 2, 3)} for case, hs in MAXH.items()}
CASES["unfitted-noteeth-dirichlet"] = {k: MAXH["unfitted-noteeth"] for k in (1, 2)}   # p_d = data on "bottom"


def row(case, k, maxh, solution=False):
    """Errors, |rho|, alpha, N (and the solution) for curved_data on the meshes of case and maxh."""
    data = curved_data()
    if case == "snapped" or case.startswith("unfitted"):
        mesh1, D1, D2 = snapped(maxh) if case == "snapped" else unfitted(maxh, teeth=case == "unfitted")
        mesh2, N = mesh1, D1.NumSet() + D2.NumSet()
    else:
        (mesh1, mesh2), D1, D2 = meshes(maxh, case == "fitted-nonmatching"), None, None
        N = mesh1.ne + mesh2.ne
    paths = vertical_paths if case.startswith("unfitted") else None
    dirichlet = "bottom" if case.endswith("-dirichlet") else None
    sol, rho, alpha = solve(mesh1, mesh2, k, phi, data, D1=D1, D2=D2, paths=paths, darcy_dirichlet=dirichlet)
    e = errors(mesh1, mesh2, sol, data, D1, D2)
    r = dict(zip(("e(L)", "u_s", "P_s", "u_d", "p_d"), e))
    r.update({"e(u)": np.hypot(e[1], e[3]), "e(p)": np.hypot(e[2], e[4]),
              "ptilde_s": norm(mesh1, sol[2] - alpha - data["P_s"], D1),
              "N": N, "|rho|": abs(rho), "alpha": alpha})
    if solution:
        r["solution"] = coefficients(sol)
    return r


if __name__ == "__main__":
    main(row, CASES, ("e(L)", "e(u)", "e(p)", "u_s", "P_s", "u_d", "p_d", "ptilde_s"),
         {"N": "6.0f", "|rho|": ".1e", "alpha": ".1e"}, hfmt="5.3f")

"""Linear elasticity on Omega = {phi < 0}, u = g on its boundary, by the HDG of
elasticity.py on an unfitted mesh with the Transfer Path Method
(Cardenas-Solano, IMA J. Numer. Anal. 2024, scheme (2.2)).

The HDG rows of elasticity.py live on the interior subdomain D_h; on its boundary
Gamma_h the equation of uhat is the trace transferred to the end xbar of the path,
using grad u = A sigma + S(gamma):

    uhat(x) + int_x^xbar (E (A sigma_h + S(gamma_h))) dl = g(xbar),

with E the extrapolation from the element of x and sigma_h the whole stress (the P_k
part and the curl of the bubbles; full=False keeps only the P_k part). Where the
domain reaches the mesh boundary, that part is fitted and uhat = P_M g there.

Examples (Cardenas-Solano 2024, Section 6), E = 1, tau_S = mu:
  1  the unit disk meshed as a subdomain of a box with straight elements, D_h = the
     disk elements (Gamma_h is a polygon at a distance O(h^2) from Gamma);
  2  the unit disk immersed in the background box (-1.3, 1.3)^2;
  3  a kidney (non-convex) immersed in the same box.
Nearly incompressible materials: taking tau = I in the first row,
(tr A sigma_h, 1) = <uhat, n> on Gamma_h, so the mean hydrostatic part of the stress error is

    c := mean of tr(sigma_h - sigma)/2 over D_h = (lam + mu)/|D_h| <uhat - u, n>_{Gamma_h}

(the paper's (3.20)): lam times the flux error of the transferred trace. It is the same
with the pressure split of elasticity.py (split=True, off by default as in the MATLAB
code, an exact reformulation), and the stress is robust in lam only up to c I; that is
what sigma_modulo_pressure measures.

R is the ratio (paper eq. (3.1)) max over the faces e of Gamma_h of the longest path
from e over the height of the element of e over e.

On the immersed examples D_h can be pruned as in the MATLAB code (prune=True; off by
default): elements with a face of Gamma_h whose normal is more than 45 degrees off the
normal of Gamma, and elements with two faces on Gamma_h, are removed.
"""

import numpy as np
from ngsolve import *

from elasticity import (HEADER, bounded_data, compliance, eoc_line, errors, finish, forms, lame,
                        manufactured, skew, spaces, stress)
from ngstpm import boundary_ds, boundary_facets, interior_ds, transfer_paths, unfitted_domain
from poisson_interface import fitted_mesh
from poisson_tpm import background_mesh


def solve(mesh, order, lam, mu, f, g, phi, D=None, tau=1.0, dirichlet=".*", full=True, split=False):
    """g(X, Y): the displacement on Gamma, a vector CF. dirichlet: the fitted part of
    the mesh boundary. Returns (sigma_h, u_h, gamma_h, uhat_h, ustar_h) and D."""
    D = unfitted_domain(mesh, phi)[0] if D is None else D
    p = transfer_paths(mesh, phi, D, order=order + 4)                 # x -> p.target on Gamma
    dB = gamma_ds(mesh, D, bonus_intorder=max(6, 2 * order))         # Gamma_h; g, ell not polynomial

    fes, top = spaces(mesh, order, dirichlet, D, split)
    lhs, rhs, (sP, w1, w2, u, gam, uhat), muh, sigmahat_n = forms(fes, lam, mu, f, tau, D)
    lhs += sigmahat_n * muh * interior_ds(mesh, D)
    sigma = stress(sP, w1, w2) if full else sP
    lhs += uhat.Transfer(p, gradient=compliance(sigma, lam, mu) + skew(gam)) * muh * dB
    rhs += g(*p.target) * muh * dB
    bc = (g(x, y), BND, mesh.Boundaries(dirichlet))
    return finish(lhs, rhs, fes, top, bc, mesh, order, lam, mu, D), D


def gamma_ds(mesh, D, **kw):
    """boundary_ds(mesh, D) integrated only on the elements of D at Gamma_h (the same
    integral; the transferred terms are not evaluated on the other elements)."""
    gamma = boundary_facets(mesh, D)[0]
    DG = BitArray(mesh.ne)
    DG.Clear()
    for e in mesh.facets:
        if gamma[e.nr]:
            for el in e.elements:
                if D[el.nr]:
                    DG.Set(el.nr)
    return boundary_ds(mesh, DG, facets=gamma, **kw)


def path_ratio(mesh, phi, D, npts=5):
    """R = max over the faces e of Gamma_h of max |ell| on e / (2 |K| / |e|), K the
    element of D at e; ell sampled at npts Gauss points of e, evaluated on K."""
    p = transfer_paths(mesh, phi, D)
    t = 0.5 * (np.polynomial.legendre.leggauss(npts)[0] + 1)
    R = 0
    gamma = boundary_facets(mesh, D)[0]
    for e in mesh.facets:
        if not gamma[e.nr]:
            continue
        K = next(el for el in e.elements if D[el.nr])
        A, B = (np.array(mesh[v].point) for v in e.vertices)
        trafo = mesh.GetTrafo(K)                                   # affine: x = p0 + M xi
        p0, p1, p2 = (np.array(trafo(*xi).point) for xi in ((0, 0), (1, 0), (0, 1)))
        M = np.column_stack((p1 - p0, p2 - p0))
        height = abs(np.linalg.det(M)) / np.linalg.norm(B - A)
        ell = max(abs(p.ell(trafo(*np.linalg.solve(M, A + s * (B - A) - p0)))) for s in t)
        R = max(R, ell / height)
    return R


# MATLAB unfitted-hdg, SimplicialGrid/split.m (get_active_elements): prune_infeasible_elements
# (max_deviation 45, interface_band 2), prune_hanging_elements, prune_disconnected_elements.
def prune(mesh, phi, D, angle=45):
    """D without its infeasible elements (a face of Gamma_h whose outward normal is more
    than angle degrees off the normal of Gamma at the projection of the face midpoint,
    judged only within 2 face lengths of Gamma), its teeth (elements with >= 2 faces on
    Gamma_h) and all but its largest face-connected component (by area); each filter to
    a fixed point, the three repeated until nothing changes."""
    D = BitArray(D)
    while True:
        n = D.NumSet()
        while _remove(D, _infeasible(mesh, phi, D, angle)):
            pass
        while _remove(D, _teeth(mesh, D)):
            pass
        _keep_largest_component(mesh, D)
        if D.NumSet() == n:
            return D


def _remove(D, elements):
    for K in elements:
        D.Clear(K)
    return bool(elements)


def _gamma_faces(mesh, D):
    """(facet, element of D, the other element) for the faces of Gamma_h."""
    out = []
    for e in mesh.facets:
        els = [el.nr for el in e.elements]
        if len(els) == 2 and D[els[0]] != D[els[1]]:
            out.append((e, *(els if D[els[0]] else els[::-1])))
    return out


def _grad(phi, X, d=1e-6):
    d *= max(1.0, np.linalg.norm(X))
    return np.array([(phi(X[0] + d, X[1]) - phi(X[0] - d, X[1])) / (2 * d),
                     (phi(X[0], X[1] + d) - phi(X[0], X[1] - d)) / (2 * d)])


def _project(phi, X, tol=1e-12, steps=50):
    """Newton projection onto {phi = 0} along grad phi (LevelSetCurve.project)."""
    for _ in range(steps):
        v = phi(*X)
        if abs(v) <= tol:
            return X
        g = _grad(phi, X)
        X = X - v * g / (g @ g)
    raise RuntimeError(f"prune: the projection of {X} onto phi = 0 did not converge")


def _infeasible(mesh, phi, D, angle):
    mids, dots = [], []
    for e, K, _ in _gamma_faces(mesh, D):
        A, B = (np.array(mesh[v].point) for v in e.vertices)
        m = 0.5 * (A + B)
        n = np.array([B[1] - A[1], A[0] - B[0]]) / np.linalg.norm(B - A)
        c = np.mean([mesh[v].point for v in mesh[ElementId(VOL, K)].vertices], axis=0)
        if n @ (m - c) < 0:
            n = -n                                                    # out of K
        P = _project(phi, m)
        if np.linalg.norm(m - P) <= 2 * np.linalg.norm(B - A):       # interface_band = 2
            g = _grad(phi, P)
            mids.append(K)
            dots.append(n @ g / np.linalg.norm(g))
    if not dots:
        return []
    side = np.sign(np.median(dots)) or 1.0
    return sorted({K for K, d in zip(mids, dots) if side * d < np.cos(np.radians(angle))})


def _teeth(mesh, D):
    count = {}
    for _, K, _ in _gamma_faces(mesh, D):
        count[K] = count.get(K, 0) + 1
    return sorted(K for K, c in count.items() if c >= 2)


def _keep_largest_component(mesh, D):
    comp = {}
    for start in (el.nr for el in mesh.Elements(VOL) if D[el.nr]):
        if start in comp:
            continue
        comp[start], stack = start, [start]
        while stack:
            K = stack.pop()
            for f in mesh[ElementId(VOL, K)].facets:
                for el in mesh[f].elements:
                    if D[el.nr] and el.nr not in comp:
                        comp[el.nr] = start
                        stack.append(el.nr)
    area = {}
    for K, c in comp.items():
        P = np.array([mesh[v].point for v in mesh[ElementId(VOL, K)].vertices])
        (a1, a2), (b1, b2) = P[1] - P[0], P[2] - P[0]
        area[c] = area.get(c, 0) + 0.5 * abs(a1 * b2 - a2 * b1)
    if len(area) > 1:
        keep = max(area, key=area.get)
        _remove(D, [K for K, c in comp.items() if c != keep])


def at(cf):
    """cf (a CF of x, y) as a callable of the coordinates, for the data at xbar."""
    return lambda X, Y: cf.Replace({x: X, y: Y})


EXAMPLES = {
    "Example 1 (fitted disk)": lambda X, Y: sqrt(X * X + Y * Y) - 1,
    "Example 2 (unfitted disk)": lambda X, Y: sqrt(X * X + Y * Y) - 1,
    "Example 3 (kidney)": lambda X, Y: ((2 * ((X + 0.5)**2 + Y * Y) - (X + 0.5))**2
                                        - ((X + 0.5)**2 + Y * Y) + 0.1),
}


def domain(name, maxh, phi, prune=False):
    """mesh, D, fitted (Dirichlet) boundary of the example, and the number of elements
    removed by prune (Example 1 is never pruned)."""
    if name.startswith("Example 1"):
        mesh = fitted_mesh(maxh)
        return mesh, BitArray([el.index == 0 for el in mesh.Elements(VOL)]), "outer", 0
    mesh = background_mesh(maxh)
    D = unfitted_domain(mesh, phi)[0]
    P = globals()["prune"](mesh, phi, D) if prune else D
    return mesh, P, ".*", D.NumSet() - P.NumSet()


def sigma_modulo_pressure(mesh, sigma_h, sigma, D):
    """|sigma - sigma_h - c I| on D_h, c the mean of tr(sigma_h - sigma)/2 over D_h."""
    dV = dx(definedonelements=D, bonus_intorder=4)
    E = sigma_h - sigma
    c = Integrate(Trace(E) * dV, mesh) / Integrate(CF(1) * dV, mesh) / 2
    return sqrt(Integrate(InnerProduct(E - c * Id(2), E - c * Id(2)) * dV, mesh))


def study(name, order, hs, nu=0.3, bounded=False, full=True, prune=False, record=None):
    """EOC table; with bounded, the last error column is |sigma - sigma_h - c I|; record:
    a study_io.Record that collects the levels."""
    phi = EXAMPLES[name]
    lam, mu = lame(1.0, nu)
    u = bounded_data(lam) if bounded else CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y)))
    sigma, gamma, f = manufactured(u, lam, mu)
    print(f"{name}, k = {order}, nu = {nu}" + ("" if full else ", P_k part of sigma transferred")
          + ("" if name.startswith("Example 1") else (", pruned" if prune else ", not pruned")))
    print(HEADER + (f"  {'|s-sh-cI|':>9}  eoc " if bounded else "") + "      R   #D_h  removed")
    prev, rows = None, []
    for h in hs:
        mesh, D, dirichlet, removed = domain(name, h, phi, prune)
        sol, D = solve(mesh, order, lam, mu, f, at(u), phi, D, dirichlet=dirichlet, full=full)
        e = errors(mesh, sol, u, sigma, gamma, D)
        if bounded:
            e = (*e, sigma_modulo_pressure(mesh, sol[0], sigma, D))
        R = path_ratio(mesh, phi, D)
        print(eoc_line(h, e, prev) + f"  {R:5.3f}  {D.NumSet():5d}  {removed:7d}")
        if record is not None:
            record.add(h, mesh, e, sol[1].gfu, R=R, nD=D.NumSet(), removed=removed)
        prev = (h, e)
        rows.append((e, R, D.NumSet(), removed))
    return rows


if __name__ == "__main__":
    from elasticity import COLUMNS
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(EXAMPLES))
    p.add_argument("--nu", type=float, default=0.3, help="Poisson ratio (default 0.3)")
    p.add_argument("--bounded", action="store_true", help="data bounded in lambda, adds |s-sh-cI|")
    p.add_argument("--prune", action="store_true", help="prune D_h (MATLAB rules)")
    p.add_argument("--pk", dest="full", action="store_false", help="transfer the P_k part of sigma only")
    a = parse(p)
    if a.k is None:
        # one process per case, within 2 GB each (k = 2 at h = 0.025 is about 2 GB: not a default)
        run_defaults(__file__, [["--case", name, "--k", k, "--h", *hs] for name in EXAMPLES
                                for k, hs in ((1, (0.2, 0.1, 0.05, 0.025)), (2, (0.2, 0.1, 0.05)))])
    else:
        record = Record(COLUMNS + (("|s-sh-cI|",) if a.bounded else ()))
        study(a.case, a.k, a.h, nu=a.nu, bounded=a.bounded, full=a.full, prune=a.prune, record=record)
        if a.out:
            save(a.out, record, dict(driver="elasticity_tpm", case=a.case, k=a.k, E=1.0, nu=a.nu,
                                     tau=1.0, bounded=a.bounded, full=a.full, prune=a.prune))

"""Quasi-normal transfer paths from Gamma_h to Gamma (Cockburn-Solano)."""

import itertools

import numpy as np
from ngsolve import (CF, Discontinuous, FacetFESpace, Grad, GridFunction, IfPos, InnerProduct, L2, VOL,
                     specialcf, sqrt, x, y, z)

from ._ngstpm import SideEval
from .geometry import boundary_facets, facet_set, whole_mesh


def _floor(v, eps=1e-12):
    return IfPos(v - eps, v, eps)


LEVEL_SET_KINDS = ("quasi-normal", "facet-normal")      # the kinds a level set (a callable) supports
ORDER = 6                                               # default degree of the ell, sigma fields


class TransferPath:
    """The straight transfer paths x -> target = x + ell sigma from Gamma_h of
    the element set D (one side) to Gamma; returned by transfer_paths.

    mesh, elements (D), geo, kind: what the paths were built from;
    facets   BitArray of the facets of Gamma_h of this side.
    ell      scalar CF, the signed length (> 0 in Omega_1, < 0 in Omega_2);
    sigma    unit vector CF, the direction;
    target   vector CF, the end x + ell sigma of the path, on Gamma;
    target_normal  unit normal of Gamma at target, from Omega_1 to Omega_2:
             normal(phi, *target) for a level set, sigma for closest-point paths.
    grad_d   matrix CF, the gradient of d = ell sigma, (grad_d)_ij = d_j d_i (for
             RegionIntegral); None for the legacy "facet-normal" kind.
    ell, sigma, grad_d and the start x of target are SideEval nodes: evaluated on
    the element of D at the integration point (an interface measure redirects
    them to the partner element).

    Data at the end of the path: a callable of the coordinates, g(X, Y), is
    evaluated there as g(*p.target) (a vector CF unpacks into its components).
    """

    def __init__(self, mesh, elements, facets, geo, kind, ell, sigma, target, target_normal, grad_d=None):
        self.mesh, self.elements, self.facets, self.geo, self.kind = mesh, elements, facets, geo, kind
        self.ell, self.sigma, self.target, self.target_normal = ell, sigma, target, target_normal
        self.grad_d = grad_d


def transfer_paths(mesh, geo, D=None, *, kind=None, order=ORDER, facets=None):
    """The TransferPath of the element set D (one side) to Gamma.

    geo: a level set phi (callable of the coordinate CFs, Gamma = {phi = 0}),
    or an explicit geometry (ParametricCurve, BSplineCurve, PointCloud).
    kind: how sigma is chosen; geo.kinds (LEVEL_SET_KINDS for a level set)
    lists the supported ones, the first is the default:
      "quasi-normal"  sigma = grad phi / |grad phi| at x, ell by Newton on
                      phi(x + ell sigma) = 0 (level sets);
      "closest"       the path ends at the closest point of Gamma, sigma is the
                      normal of Gamma there (explicit geometries; h below the
                      reach of Gamma);
      "facet-normal"  sigma = the outward facet normal (level sets). Legacy, not
                      for computing: on unfitted meshes these paths can miss Gamma.
    order: polynomial degree of the elementwise fields ell, sigma.
    D: element BitArray (default: the whole mesh).
    facets: facet BitArray of Gamma_h (default: boundary_facets(mesh, D), the
    faces between an element of D and one outside D), or a boundary Region. Pass
    it when Gamma_h is part of the mesh boundary, e.g. mesh.Boundaries("interface")
    on the mesh of one side (two meshes).
    """
    D = whole_mesh(mesh) if D is None else D
    facets = boundary_facets(mesh, D)[0] if facets is None else facet_set(mesh, facets)
    kinds = getattr(geo, "kinds", LEVEL_SET_KINDS)
    kind = kinds[0] if kind is None else kind
    if kind not in kinds:
        raise ValueError(f"kind {kind!r} not supported by this geometry; supported: {kinds}")
    if kind == "closest":
        ell, sigma, grad_d = _closest_point_paths(mesh, geo, D, order, facets=facets)
    else:
        ell, sigma, grad_d = _level_set_paths(mesh, geo, D, order, kind=kind)
    side = lambda cf: SideEval(cf, None, mesh, D)          # on this side's element (or its partner)
    ell, sigma = side(ell), side(sigma)
    grad_d = None if grad_d is None else side(grad_d)
    # on the partner of an interface measure the path ends at the integration point's xbar
    target = SideEval(side(CF((x, y, z)[:mesh.dim])) + ell * sigma, None, mesh, D, path_end=True)
    nbar = sigma if kind == "closest" else normal(geo, *target)
    return TransferPath(mesh, D, facets, geo, kind, ell, sigma, target, nbar, grad_d)


def _level_set_paths(mesh, phi, inner, degree, newton_steps=4, delta=1e-6, kind="quasi-normal"):
    """ell, sigma and grad_d (None for "facet-normal") of the level-set kinds,
    elementwise of the given degree.

    "quasi-normal": sigma = grad phi / |grad phi| at x; ell solves
    phi(x + ell sigma) = 0 by Newton from -phi/|grad phi|. Both are stored as
    elementwise GridFunctions on inner (projecting each iterate keeps the CF
    trees small).

    "facet-normal" (legacy, not for computing: on unfitted meshes the normal
    lines of staircase facets can miss Gamma, and Newton then returns an xbar
    off Gamma): sigma = the outward facet normal of the element
    (specialcf.normal), constant on each facet of Gamma_h; only defined on
    element boundaries. ell is the same Newton iteration, from 0, with each
    iterate projected on the facets of each element
    (Discontinuous(FacetFESpace): one value per element side of a facet).

    The Newton derivative is a central difference in ell; it only affects the
    convergence speed, not the endpoint (the residual uses phi itself).
    """
    X = (x, y, z)[:mesh.dim]
    if kind == "facet-normal":
        sigma = specialcf.normal(mesh.dim)
        V = Discontinuous(FacetFESpace(mesh, order=degree))
        at = lambda ell: phi(*(Xi + ell * sigma[i] for i, Xi in enumerate(X)))
        ell, new = GridFunction(V), GridFunction(V)
        for _ in range(newton_steps + 1):
            dphi = (at(ell + delta) - at(ell - delta)) / (2 * delta)
            new.Set(ell - at(ell) / dphi, dual=True, definedonelements=inner)
            ell, new = new, ell
        return ell, sigma, None
    V = L2(mesh, order=degree)
    p = phi(*X)
    grad = [p.Diff(Xi) for Xi in X]
    norm = _floor(sqrt(sum(gi * gi for gi in grad)))
    comps = [GridFunction(V) for _ in X]
    for c, gi in zip(comps, grad):
        c.Set(gi / norm, definedonelements=inner)
    sigma = CF(tuple(comps))

    def phi_at(ell):
        return phi(*(Xi + ell * sigma[i] for i, Xi in enumerate(X)))

    ell, new = GridFunction(V), GridFunction(V)
    ell.Set(-p / norm, definedonelements=inner)
    for _ in range(newton_steps):
        dphi = (phi_at(ell + delta) - phi_at(ell - delta)) / (2 * delta)
        new.Set(ell - phi_at(ell) / dphi, definedonelements=inner)
        ell, new = new, ell
    return ell, sigma, _grad_d(ell, comps)


def _grad_d(ell, comps):
    """The gradient of d = ell sigma, (grad d)_ij = d_j d_i, from the elementwise
    GridFunctions ell and sigma_i (the product rule; no re-projection)."""
    rows = [c * Grad(ell) + ell * Grad(c) for c in comps]
    return CF(tuple(rows), dims=(len(comps), len(comps)))


def _closest_point_paths(mesh, geo, inner, order, facets=None):
    """ell, sigma (and grad_d) as elementwise P_order GridFunctions, fitted (least squares
    of minimal norm: only the traces count) to the closest-point data at
    nodes on the facets of Gamma_h: those where an element of inner meets an
    element outside inner or, if given, the facets of the BitArray facets."""
    d = mesh.dim
    verts = np.array([v.point for v in mesh.vertices])
    els = [el for el in mesh.Elements(VOL) if inner[el.nr]]
    cen = np.zeros((mesh.ne, d))
    hK = np.ones(mesh.ne)
    for el in els:
        P = verts[[v.nr for v in el.vertices]]
        cen[el.nr] = P.mean(axis=0)
        hK[el.nr] = np.linalg.norm(P - cen[el.nr], axis=1).max()
    n = order + 2
    if d == 2:
        t = 0.5 * (np.polynomial.legendre.leggauss(n)[0] + 1)
        lam = np.column_stack((t, 1 - t))
    else:
        lam = np.array([(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]) / n
    mono = [e for e in itertools.product(range(order + 1), repeat=d) if sum(e) <= order]
    coef = np.zeros((d + 1, len(mono), mesh.ne))
    for el in els:
        nodes = []
        for f in el.facets:
            other = [e.nr for e in mesh[f].elements if e.nr != el.nr]
            if (facets[f.nr] if facets is not None else
                    other and not inner[other[0]]):
                nodes.append(lam @ verts[[v.nr for v in mesh[f].vertices]])
        if nodes:
            Xn = np.vstack(nodes)
            P, N = geo.closest(Xn)
            xi = (Xn - cen[el.nr]) / hK[el.nr]
            A = np.column_stack([np.prod(xi ** e, axis=1) for e in mono])
            rhs = np.column_stack((np.sum((P - Xn) * N, axis=1), N))
            coef[:, :, el.nr] = np.linalg.lstsq(A, rhs, rcond=None)[0].T
    V0 = L2(mesh, order=0)
    gfs = [GridFunction(V0) for _ in range(d + 1)]
    for gf, vals in zip(gfs, list(cen.T) + [hK]):
        gf.vec.FV().NumPy()[:] = vals
    xi = [(Xi - c) / gfs[d] for Xi, c in zip((x, y, z), gfs[:d])]
    psi = []
    for e in mono:
        p = CF(1.0)
        for xij, ej in zip(xi, e):
            if ej:
                p = p * xij ** ej
        psi.append(p)
    psi = CF(tuple(psi))
    V = L2(mesh, order=order)
    fields = []
    for c in coef:
        cm = GridFunction(V0 ** len(mono))
        cm.vec.FV().NumPy()[:] = c.ravel()
        gf = GridFunction(V)
        gf.Set(InnerProduct(cm, psi), definedonelements=inner)
        fields.append(gf)
    return fields[0], CF(tuple(fields[1:])), _grad_d(fields[0], fields[1:])


def normal(phi, *X, delta=1e-6):
    """Unit normal grad phi / |grad phi| of the level sets of phi at the point X
    (coordinate CFs, e.g. the end x + ell sigma of a transfer path), by central
    differences."""
    grad = [(phi(*(Xj + delta if j == i else Xj for j, Xj in enumerate(X)))
             - phi(*(Xj - delta if j == i else Xj for j, Xj in enumerate(X)))) / (2 * delta)
            for i in range(len(X))]
    return CF(tuple(grad)) / _floor(sqrt(sum(gi * gi for gi in grad)))

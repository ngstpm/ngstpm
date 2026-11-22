"""Interior subdomain D_h of a level-set domain on a background mesh."""

import numpy as np
from netgen.libngpy._meshing import NgException
from ngsolve import (BND, CF, BitArray, CoefficientFunction, Compress, FacetFESpace, GridFunction, Mesh,
                     Parameter, VOL, dx, x, y, z)
from ngsolve.comp import Region


def unfitted_domain(mesh, phi, side=1):
    """Interior subdomain of Omega = {phi < 0} on a background mesh; with
    side=2 that of the other side, {phi > 0} (for any kind of geometry).

    phi: callable of the coordinate CFs, phi(x, y) -> CF, or an explicit
    geometry (ParametricCurve, BSplineCurve, PointCloud), which stands for
    its signed distance.
    Returns (inner, gamma, ind): elements with phi < 0 at all vertices (D_h),
    facets with exactly one inner neighbour (Gamma_h), and a facet
    GridFunction equal to 1 on Gamma_h.
    """
    if side not in (1, 2):
        raise ValueError(f"side must be 1 ({{phi < 0}}) or 2 ({{phi > 0}}), not {side!r}")
    sign = 1 if side == 1 else -1
    vals = _vertex_values(mesh, phi)
    inner = BitArray(mesh.ne)
    inner.Clear()
    for el in mesh.Elements(VOL):
        inner[el.nr] = all(sign * vals[v.nr] < 0 for v in el.vertices)
    return (inner, *boundary_facets(mesh, inner))


def _vertex_values(mesh, phi):
    """phi at the mesh vertices (the signed distance for an explicit geometry)."""
    if hasattr(phi, "closest"):
        return phi.signed_distance(np.array([v.point for v in mesh.vertices]))
    p = phi(*(x, y, z)[:mesh.dim])
    return [p(mesh(*v.point)) for v in mesh.vertices]


CORNER_ANGLE = 30.0     # degrees: a boundary vertex where the boundary turns by more is a corner
PARALLEL = 0.1          # |projected grad phi| / |grad phi| below which a boundary vertex moves freely


def snapped_domains(mesh, phi, newton_steps=4):
    """Snap a background mesh onto Gamma = {phi = 0}.

    The vertices of the cut elements are moved onto Gamma (Newton along
    grad phi; to the closest point for an explicit geometry), so the cut elements collapse to slivers between D_h^1 (the
    elements that were entirely in {phi < 0}) and D_h^2 (those in {phi > 0}):
    the gap between them is O(h^2) instead of O(h), and Gamma_h^1, Gamma_h^2
    do not match. Elements left with all their vertices on Gamma are dropped
    from D_h^i. A moved vertex on the mesh boundary stays on it: Newton runs
    along the boundary (the step projected on its tangent space at the vertex,
    with the gradient of the signed distance, the normal of the closest point,
    for an explicit geometry, whose closest() must return normals). The
    tangent space is that of the mean of the boundary normals at the vertex; a
    vertex where the boundary turns by more than CORNER_ANGLE (30 degrees) is a
    corner, with no tangent direction. On a polygon approximating a curved boundary the
    vertex slides along the mean tangent and leaves the polygon by O(h * step)
    (there phi is evaluated from the coordinates alone: phi must depend on its
    arguments only).
    Where Gamma is nearly parallel to the boundary (the projected gradient below
    PARALLEL = 0.1 of |grad phi|, i.e. within 5.7 degrees, so that the step
    along the boundary would be more than 10 times the distance to Gamma), and
    at a corner, the vertex takes the unconstrained step of an interior vertex,
    as before this boundary treatment, and leaves the boundary. Newton along the boundary is not stopped at a corner: a step may
    overshoot past it (not handled).
    Returns (mesh, D1, D2), with the new mesh.
    """
    explicit = hasattr(phi, "closest")
    vals = _vertex_values(mesh, phi)
    if not explicit:
        X = (x, y, z)[:mesh.dim]
        p = phi(*X)
        grad = CF(tuple(p.Diff(Xi) for Xi in X))
    D1, D2 = BitArray(mesh.ne), BitArray(mesh.ne)
    moved = set()
    for el in mesh.Elements(VOL):
        D1[el.nr] = all(vals[v.nr] < 0 for v in el.vertices)
        D2[el.nr] = all(vals[v.nr] > 0 for v in el.vertices)
        if not (D1[el.nr] or D2[el.nr]):
            moved.update(v.nr for v in el.vertices)
    tangents = _boundary_tangents(mesh, moved)
    ngmesh = mesh.ngmesh.Copy()
    def value_and_gradient(c):
        if explicit:
            return phi.signed_distance(c[None])[0], phi.closest(c[None])[1][0]
        mp = mesh(*c)
        try:
            return p(mp), np.array(grad(mp))
        except NgException:             # slid off a polygonal boundary: phi of the coordinates alone
            return outside(c)

    def outside(c):
        args = [Parameter(float(ci)) for ci in c]
        q = phi(*args)
        mp = mesh(*mesh.vertices[0].point)
        return q(mp), np.array([q.Diff(a)(mp) for a in args])

    for i, pt in enumerate(ngmesh.Points()):
        slide = i in tangents and len(tangents[i]) > 0
        if slide:
            c, T = np.array(mesh.vertices[i].point), tangents[i]
            g = value_and_gradient(c)[1]
            slide = np.linalg.norm(T @ g) >= PARALLEL * np.linalg.norm(g)
        if slide:
            for _ in range(newton_steps):
                f, g = value_and_gradient(c)
                g = T.T @ (T @ g)                                 # along the boundary
                c = c - f * g / g.dot(g)
            for j in range(mesh.dim):
                pt[j] = c[j]
        elif i in moved:
            c = np.array(mesh.vertices[i].point)
            if explicit:
                c = phi.closest(c[None])[0][0]
            for _ in range(0 if explicit else newton_steps):
                mp = mesh(*c)
                g = np.array(grad(mp))
                c = c - p(mp) * g / g.dot(g)
            for j in range(mesh.dim):
                pt[j] = c[j]
    snapped = Mesh(ngmesh)
    for el in snapped.Elements(VOL):
        if all(v.nr in moved for v in el.vertices):
            D1[el.nr] = D2[el.nr] = False
    return snapped, D1, D2


def _boundary_tangents(mesh, vertices):
    """For the given vertices that lie on the mesh boundary: an orthonormal basis
    (rows) of the tangent space of the boundary there, the complement of the
    directions spanned by the normals of its boundary elements at the vertex.
    A direction counts when its singular value exceeds tan(CORNER_ANGLE / 2)
    times the largest (two unit normals at angle t have singular values
    sqrt(1 +- cos t), ratio tan(t / 2)); so a boundary turning by less than
    CORNER_ANGLE gives the tangent space of the mean normal, and a corner
    gives no rows."""
    normals = {}
    for el in mesh.Elements(BND):
        vs = [v.nr for v in el.vertices]
        if not vertices.intersection(vs):
            continue
        P = np.array([mesh.vertices[v].point for v in vs])
        n = (np.array([P[1, 1] - P[0, 1], P[0, 0] - P[1, 0]]) if mesh.dim == 2
             else np.cross(P[1] - P[0], P[2] - P[0]))
        for v in vertices.intersection(vs):
            normals.setdefault(v, []).append(n / np.linalg.norm(n))
    tangents = {}
    for v, N in normals.items():
        _, S, Vt = np.linalg.svd(np.array(N))
        tangents[v] = Vt[np.sum(S > np.tan(np.radians(CORNER_ANGLE) / 2) * S[0]):]
    return tangents


def boundary_facets(mesh, inner):
    """Gamma_h of the element set inner: the facets between an element of the
    set and one outside it. Returns (gamma, ind): a BitArray and a facet
    GridFunction equal to 1 on Gamma_h."""
    gamma = BitArray(mesh.nfacet)
    gamma.Clear()
    for f in mesh.facets:
        els = f.elements
        gamma[f.nr] = len(els) == 2 and inner[els[0].nr] != inner[els[1].nr]
    return gamma, _facet_indicator(mesh, gamma)


def _facet_indicator(mesh, facets):
    """Facet GridFunction (FacetFESpace, order 0) equal to 1 on the marked facets."""
    F0 = FacetFESpace(mesh, order=0)
    ind = GridFunction(F0)
    for f in mesh.facets:
        for d in F0.GetDofNrs(f):
            ind.vec[d] = 1.0 if facets[f.nr] else 0.0
    return ind


def _interior_facets(mesh, inner):
    """The facets whose two neighbours are both in the element set inner."""
    facets = BitArray(mesh.nfacet)
    facets.Clear()
    for f in mesh.facets:
        els = f.elements
        facets[f.nr] = len(els) == 2 and inner[els[0].nr] and inner[els[1].nr]
    return facets


class _FacetMeasure:
    """One-sided integral over some facets of an element set D, seen from D:
    dx(element_boundary=True, definedonelements=E, **kw) with the indicator of
    the facets multiplied into the integrand, E the elements of D with a marked
    facet (the integrand is never evaluated on the other elements of D, where
    0 * NaN would still be NaN). `cf * measure` is an ordinary SumOfIntegrals;
    `measure(**kw)` returns the measure with more options."""

    def __init__(self, D, ind, kw):
        self.D, self.ind, self.kw = D, ind, kw

    @classmethod
    def on(cls, mesh, D, facets, kw):
        """The measure of the facets (BitArray) seen from the elements of D."""
        E = BitArray(mesh.ne)
        E.Clear()
        for f in mesh.facets:
            if facets[f.nr]:
                for el in f.elements:
                    if D[el.nr]:
                        E[el.nr] = True
        return cls(E, _facet_indicator(mesh, facets), kw)

    def __call__(self, **kw):
        return _FacetMeasure(self.D, self.ind, {**self.kw, **kw})

    def __rmul__(self, cf):
        return (self.ind * cf) * dx(element_boundary=True, definedonelements=self.D, **self.kw)


def whole_mesh(mesh):
    """The element BitArray of all elements of mesh."""
    D = BitArray(mesh.ne)
    D.Set()
    return D


def facet_set(mesh, facets):
    """facets as a facet BitArray: a BitArray is returned as is; a boundary
    Region (e.g. mesh.Boundaries("interface")) gives the facets of its elements."""
    if not isinstance(facets, Region):
        return facets
    F = BitArray(mesh.nfacet)
    F.Clear()
    for el in facets.Elements():
        F[(el.edges if mesh.dim == 2 else el.faces)[0].nr] = True
    return F


def interior_ds(mesh, D=None, **kw):
    """Integral over the interior faces of the element set D (both neighbours
    in D), e.g. for the HDG conservativity row: lhs += qhat_n*mu*interior_ds(mesh, D).

    One-sided: each face is integrated from both of its elements, as
    dx(element_boundary=True, definedonelements=D, **kw) restricted to those
    faces, so specialcf.normal is the outward normal of the element K and
    proxies are K's. Faces on Gamma_h of D and mesh boundary (BND) faces are
    excluded; conditions on the mesh boundary are written with NGSolve's ds
    as usual. Options (bonus_intorder, ...) pass through, also later:
    interior_ds(mesh, D)(bonus_intorder=b). D=None: the whole mesh.
    """
    D = whole_mesh(mesh) if D is None else D
    return _FacetMeasure.on(mesh, D, _interior_facets(mesh, D), kw)


def boundary_ds(mesh, D=None, facets=None, **kw):
    """Integral over Gamma_h of the element set D: the faces with one neighbour
    in D and the other an element not in D, seen from D.

    One-sided element-boundary integral, as dx(element_boundary=True,
    definedonelements=D, **kw) restricted to those faces, so
    specialcf.normal is the outward normal of the element K in D and proxies
    are K's. Mesh boundary (BND) faces are excluded; conditions on the mesh
    boundary are written with NGSolve's ds as usual. Options
    (bonus_intorder, ...) pass through, also later:
    boundary_ds(mesh, D)(bonus_intorder=b).
    facets: a facet BitArray or boundary Region to use as Gamma_h instead (e.g.
    mesh.Boundaries("interface") when Gamma_h is part of the mesh boundary, two
    meshes). D=None: the whole mesh.
    """
    D = whole_mesh(mesh) if D is None else D
    if facets is not None:
        return _FacetMeasure.on(mesh, D, facet_set(mesh, facets), kw)
    return _FacetMeasure.on(mesh, D, boundary_facets(mesh, D)[0], kw)


_cf_mul = CoefficientFunction.__mul__


def _mul(self, other):
    # CoefficientFunction.__mul__ raises TypeError on unknown operands instead
    # of returning NotImplemented, so _FacetMeasure.__rmul__ is reached through this hook.
    if isinstance(other, _FacetMeasure):
        return other.__rmul__(self)
    return _cf_mul(self, other)


CoefficientFunction.__mul__ = _mul


def restrict(fes, inner):
    """Compress fes to the dofs of the elements marked in inner."""
    active = BitArray(fes.ndof)
    active.Clear()
    for el in fes.mesh.Elements(VOL):
        if inner[el.nr]:
            for d in fes.GetDofNrs(el):
                active[d] = True
    return Compress(fes, active_dofs=active)

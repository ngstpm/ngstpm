"""pair(p1, p2) in 3D (D10): pieces of the faces of Gamma_h^i by clipping in the
charts of their subtriangles, the per-face coverage check and the partner rules.

Tolerances (ngstpm.interface; in 3D relative to h = the larger of the two
sides' mean edge lengths of Gamma_h), measured on the configurations below
(sphere: ball and box minus ball meshed on their own; cylinder: open surface):
- pieces below MERGE_TOL h^2 = 1e-10 h^2 are dropped. Slivers that small
  appear where vertices of the two sides (nearly) coincide: on the sphere up to
  9.9e-11 h^2 per face (h1 = 0.2, h2 = 0.6; 2.0e-11 h^2 at h1 = 0.12), and with
  vertices moved by eps h from 1e-10 h^2 per face. A side's pieces tile
  Gamma_h^i to 5e-15 relative without such slivers, 1.9e-12 (sphere) and
  1.7e-10 (eps = 1e-10) with them;
- vertex images of one side (one per face at the vertex) differ by at most
  8.2e-9 h (sphere; measured with a temporary diagnostic);
- the partner x' of a rule point lies at most 2.3e-2 h outside its face F' on
  the sphere (h1 = 0.5, h2 = 0.35, h = 0.47; 1.6e-2 h at h ratios up to 4,
  h = 0.48), 2.5e-2 h with a coarse side of 58 faces (h = 0.70), 5.5e-3 h on the
  cylinder: GAP_TOL = 5e-2 h leaves a margin of 2.2 in the worst case. GAP_TOL h
  also bounds how far the path of x' may end from xbar, and the absorption of
  test_absorption (x' up to GAP_TOL h outside F'). The rim strips (depth GAP_TOL
  h) covered at most 1e-3 h^2 per face on the cylinder (measured with the
  charts in the planes of the image triangles; not re-measured).
Subdivision (m_F^2 subtriangles per face, the path map resolved to 0.2 GAP_TOL h
across the paths): the two-mesh, fitted, open-surface and absorption cases have
m = 1 on every face (asserted), and on the sphere (0.5, 0.35), the cylinder
(0.2, 0.3) and the fitted one-mesh case the segments and rules are bitwise those
without subdivision (order 6; measured, pair 0.91 s on that sphere before and
after). Carved sides of
one background mesh (unit sphere, box half-width about 1.5; measured, the larger
ones not in the suite): structured n = 6, 8, 12 and netgen maxh 0.3 (h = 0.62,
0.47, 0.31, 0.42) pair, m from 1 to 8 (mostly 2, 4), x' at most 4.6e-3, 1.0e-2,
1.4e-2 and 5.0e-2 h outside F' (netgen 0.3: 187 of 2.6e5 points absorbed, miss
up to 6.5e-3 h, partners on faces nearly parallel to their paths, n_F . sigma =
0.009), per face up to 6.9e-10 of the area dropped in slivers (each below
MERGE_TOL h^2: corner overlaps of subtriangles), per side 6e-13 - 4.4e-11; pair
15 - 21 s (the order-12 check of the rules on 37 - 57 thousand triangles).
Structured n = 10, 16, 20 contain a face in the plane z = 0 through the centre
(parallel to its paths: no chart; they fold too, n_F . sigma down to -0.03),
netgen 0.2 folds (faces with n_F . sigma < 0: 1 on side 1, 3 on side 2), netgen
0.4, 0.15 are not surfaces (an edge in 4 faces): pair raises.
"""

from functools import lru_cache

import numpy as np
import pytest
from netgen.csg import CSGeometry, Cylinder, OrthoBrick, Plane, Pnt, Sphere, Vec
from ngsolve import BitArray, CF, GridFunction, Integrate, L2, Mesh, VOL, ds, exp, sqrt, x, y, z
from ngsolve.meshes import MakeStructured3DMesh

from ngstpm import pair, transfer_paths
from ngstpm.interface import GAP_TOL, MERGE_TOL


def ball(X, Y, Z):
    return sqrt(X * X + Y * Y + Z * Z) - 1


def sphere_meshes(h1, h2, curvaturesafety=2):
    """The unit ball (maxh h1) and the box minus the ball (maxh h2 on the sphere;
    curvaturesafety < 2 allows coarser faces), meshed on their own: the bc
    "interface" on both."""
    g1 = CSGeometry()
    g1.Add(Sphere(Pnt(0, 0, 0), 1).bc("interface"))
    g2 = CSGeometry()
    g2.Add(OrthoBrick(Pnt(-1.5, -1.5, -1.5), Pnt(1.5, 1.5, 1.5)).bc("outer")
           - Sphere(Pnt(0, 0, 0), 1).bc("interface").maxh(h2))
    return Mesh(g1.GenerateMesh(maxh=h1)), Mesh(g2.GenerateMesh(maxh=max(h2, 0.5), curvaturesafety=curvaturesafety))


def two_mesh_paths(m1, m2, phi=ball):
    return [transfer_paths(m, phi, facets=m.Boundaries("interface")) for m in (m1, m2)]


@lru_cache
def sphere_paths(h1, h2, curvaturesafety=2):
    """The paths of sphere_meshes(h1, h2, curvaturesafety) (transfer_paths dominates
    the run time); tests must not modify them."""
    return tuple(two_mesh_paths(*sphere_meshes(h1, h2, curvaturesafety)))


def fitted_mesh(h):
    """One mesh fitted to the sphere: the ball (material "ball") and the rest of the box."""
    geo = CSGeometry()
    s = Sphere(Pnt(0, 0, 0), 1)
    geo.Add(s.mat("ball"))
    geo.Add((OrthoBrick(Pnt(-1.5, -1.5, -1.5), Pnt(1.5, 1.5, 1.5)) - s).mat("out"))
    return Mesh(geo.GenerateMesh(maxh=h))


def fitted_paths(h, m1=None, m2=None):
    """D_h^1 the ball of m1, D_h^2 the rest of the box in m2 (default: both
    fitted_mesh(h), one mesh)."""
    m1 = fitted_mesh(h) if m1 is None else m1
    m2 = m1 if m2 is None else m2
    D1, D2 = (BitArray([el.mat == "ball" for el in m.Elements(VOL)]) for m in (m1, m2))
    return transfer_paths(m1, ball, D1), transfer_paths(m2, ball, ~D2)


def face_data(mesh, F):
    """{facet nr: (p0, p1, p2)} with the vertex order of the mesh."""
    return {f.nr: np.array([mesh[v].point for v in f.vertices]) for f in mesh.facets if F[f.nr]}


def tri_area(P):
    return 0.5 * np.linalg.norm(np.cross(P[..., 1, :] - P[..., 0, :], P[..., 2, :] - P[..., 0, :]), axis=-1)


def check(I, p1, p2):
    """The checks of the pairing in 3D; returns the numbers (see the asserts)."""
    paths = (p1, p2)
    fd = [face_data(p.mesh, p.facets) for p in paths]
    h = max(np.mean([np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).mean() for P in d.values()]) for d in fd)
    S = I.segments()
    n = I.nsegments
    assert len({(a, b) for a, b in S.facet}) == n                  # one segment per pair (F, F')
    defect, side_tiling, inref, onface = -np.inf, 0.0, 0.0, 0.0
    for i in (0, 1):
        assert all(paths[i].elements[int(e)] for e in S.el[:, i])
        covered = {f: 0.0 for f in fd[i]}
        for s in range(n):
            covered[S.facet[s, i]] += S.area[s, i]
        A = {f: tri_area(P) for f, P in fd[i].items()}
        defect = max(defect, max((abs(covered[f] - A[f]) - 1e-12 * A[f]) / h ** 2 for f in A))
        side_tiling = max(side_tiling, abs(sum(covered.values()) - sum(A.values())) / sum(A.values()))
        m = S.tri_side == i
        r, X, seg = S.tri_ref[m], S.tri_x[m], S.tri_seg[m]
        inref = max(inref, max(-r.min(), (r.sum(axis=2) - 1).max()))   # in the reference triangle
        P = np.array([fd[i][S.facet[s, i]] for s in seg])
        Xr = P[:, None, 0] + r[..., :1] * (P[:, None, 1] - P[:, None, 0]) + r[..., 1:] * (P[:, None, 2] - P[:, None, 0])
        onface = max(onface, np.abs(Xr - X).max())
        a = np.zeros(n)
        np.add.at(a, seg, tri_area(X))
        assert np.allclose(a, S.area[:, i], rtol=1e-12, atol=1e-15)
    over = par = 0.0
    for side in (1, 2):
        i, j = side - 1, 2 - side
        R = I._rule(side, 6)
        w = np.zeros(n)
        np.add.at(w, R["seg"], R["weight"])
        assert np.allclose(w, S.area[:, i], rtol=1e-12, atol=1e-15)   # weights: the pieces' areas
        assert np.all(R["el"] == S.el[R["seg"], i]) and np.all(R["pel"] == S.el[R["seg"], j])
        G = np.array([fd[j][S.facet[s, j]] for s in R["seg"]])
        # x' in the reference coordinates of F', and how far outside F'
        E = np.stack([G[:, 1] - G[:, 0], G[:, 2] - G[:, 0]], axis=2)
        Et = np.transpose(E, (0, 2, 1))
        lam = np.linalg.solve(Et @ E, (Et @ (R["xp"] - G[:, 0])[..., None]))[..., 0]
        onface = max(onface, np.abs(G[:, 0] + np.einsum("nij,nj->ni", E, lam) - R["xp"]).max())
        bary = np.column_stack([1 - lam.sum(1), lam])
        hgt = np.stack([2 * tri_area(G) / np.linalg.norm(G[:, (k + 2) % 3] - G[:, (k + 1) % 3], axis=1)
                        for k in range(3)], axis=1)
        over = max(over, (np.maximum(-bary, 0) * hgt).max() / h)
        par = max(par, np.linalg.norm(np.cross(R["xbar"] - R["xp"], R["sigmap"]), axis=1).max())
    return dict(nseg=n, h=h, defect=defect, side_tiling=side_tiling, inref=inref, onface=onface, over=over, par=par)


def assert_pairing(r, sliver=0.0, side_tiling=1e-12):
    """sliver: the area (in h^2) of the slivers that may be dropped on a face;
    side_tiling: the relative defect of a side's pieces that this allows."""
    assert r["defect"] <= sliver           # per face: |area of the pieces - |F|| <= 1e-12 |F| + sliver h^2
    assert r["side_tiling"] < side_tiling  # per side, relative
    assert r["inref"] < 1e-12            # the triangles of a piece lie in the reference triangle of F
    assert r["onface"] < 1e-12           # tri_x on F, x' on the plane of F'
    assert r["over"] < GAP_TOL           # x' in F', up to the gap
    assert r["par"] < 1e-10              # xbar - x' parallel to sigma(x')


@pytest.mark.parametrize("h1,h2,order", [(0.5, 0.5, 1), (0.5, 0.35, 1), (0.5, 0.35, -1)])
def test_two_meshes(h1, h2, order):
    """Ball and box-minus-ball meshed on their own (order -1: the box minus the
    ball is side 1): each side's pieces tile its Gamma_h; with equal maxh netgen
    meshes the sphere alike, and each face is paired with its twin only."""
    p1, p2 = sphere_paths(h1, h2)[::order]
    I = pair(p1, p2)
    r = check(I, p1, p2)
    assert_pairing(r, sliver=10 * MERGE_TOL)
    assert np.all(I.segments().m == 1)                         # paths of length O(h^2): no subdivision
    assert r["nseg"] >= max(sum(p1.facets), sum(p2.facets))
    if h1 == h2:
        assert r["nseg"] == sum(p1.facets) == sum(p2.facets)


def test_fitted_one_mesh():
    """One mesh fitted to the sphere, D_h^1 and D_h^2 on either side: Gamma_h^1 =
    Gamma_h^2, each face paired with itself, x' = x up to the difference of the two
    sides' path fields on a face (2.2e-10 = 4.7e-10 h)."""
    p1, p2 = fitted_paths(0.5)
    I = pair(p1, p2)
    assert_pairing(check(I, p1, p2))
    S = I.segments()
    assert I.nsegments == sum(p1.facets) and np.all(S.facet[:, 0] == S.facet[:, 1])
    assert np.all(S.m == 1)
    for side in (1, 2):
        R = I._rule(side, 8)
        assert np.abs(R["xp"] - R["x"]).max() < 1e-9


def test_equal_copies():
    """Two copies of one fitted mesh (two meshes for pair; the ball of one, the rest
    of the box in the other): each face paired with its copy, x' = x up to the
    difference of the two sides' path fields (2.2e-10, as on one mesh)."""
    m = fitted_mesh(0.5)
    p1, p2 = fitted_paths(0.5, m, Mesh(m.ngmesh.Copy()))
    I = pair(p1, p2)
    assert_pairing(check(I, p1, p2))
    S = I.segments()
    assert I.nsegments == sum(p1.facets) and np.all(S.facet[:, 0] == S.facet[:, 1])
    assert np.all(S.area > 0)
    for side in (1, 2):
        R = I._rule(side, 8)
        assert np.abs(R["xp"] - R["x"]).max() < 1e-9


POLYS = [CF(1), x, y * z, x * x * y, x * y * z * z + y ** 3, x ** 5 + x * y * y * z * z]


@pytest.mark.parametrize("order", [1, -1])
def test_integrate_exact(order):
    """Integrate(p * I.ds(side=i)) = the integral over Gamma_h^i (NGSolve's ds on
    the bc "interface") for polynomials p of degree <= 5, the order of the rule."""
    paths = sphere_paths(0.5, 0.35)[::order]
    I = pair(*paths)
    for side, m in zip((1, 2), (p.mesh for p in paths)):
        for p in POLYS:
            ref = Integrate(p * ds("interface", bonus_intorder=8), m)
            scale = Integrate(sqrt(p * p) * ds("interface", bonus_intorder=8), m)
            assert abs(Integrate(p * I.ds(side=side), m) - ref) < 1e-12 * scale


def test_non_triangular_faces_raise():
    paths = []
    for _ in range(2):                                         # two meshes of hexahedra
        mesh = MakeStructured3DMesh(hexes=True, nx=2, ny=2, nz=2)
        D = BitArray(mesh.ne)
        D.Set()
        F = BitArray(mesh.nfacet)
        F.Clear()
        for f in mesh.facets:
            F[f.nr] = len(f.elements) == 1
        paths.append(type("P", (), dict(mesh=mesh, elements=D, facets=F, target=CF((x, y, z)), sigma=CF((1, 0, 0)))))
    with pytest.raises(Exception, match="not a triangle"):
        pair(*paths)


def moved_copy(mesh, eps, seed=1):
    """A copy of the fitted mesh with its sphere vertices moved tangentially (random
    directions) by eps h, h = 0.47 the mean edge of its surface."""
    c = mesh.ngmesh.Copy()
    rng = np.random.default_rng(seed)
    for p in c.Points():
        X = np.array(p.p)
        if abs(np.linalg.norm(X) - 1) < 1e-8:
            n = X / np.linalg.norm(X)
            t = rng.normal(size=3)
            t -= t.dot(n) * n
            X += eps * 0.47 / np.linalg.norm(t) * t
            for k in range(3):
                p[k] = X[k]
    return Mesh(c)


@pytest.mark.parametrize("eps", [1e-14, 1e-10, 1e-6, 1e-3])
def test_nearly_coincident(eps):
    """The fitted mesh (h = 0.5): side 1 its ball, side 2 the rest of the box in a
    copy with the sphere vertices moved by eps h:
    slivers of width ~eps h. Measured: per face up to 1.4e-10 h^2 dropped (below
    MERGE_TOL h^2 each; 1.3e-10 h^2 at eps = 1e-4), per side 1.7e-10 relative
    (eps = 1e-10); x' at most 1.0e-4 h outside F' (eps = 1e-3)."""
    m = fitted_mesh(0.5)
    p1, p2 = fitted_paths(0.5, m, moved_copy(m, eps))
    r = check(pair(p1, p2), p1, p2)
    assert_pairing(r, sliver=20 * MERGE_TOL, side_tiling=1e-9)
    assert r["over"] < 1e-3


@pytest.mark.parametrize("order", [1, -1])
def test_h_ratio(order):
    """Very different h: the ball at maxh 0.2, the box minus the ball at 0.8 with
    curvaturesafety 0.7 (mean edges of Gamma_h 0.209 and 0.698, ratio 3.3), both
    orders of the sides. Measured: x' at most 2.5e-2 h outside F' (h = 0.70; the
    coarse side has 58 faces). Not in the suite (slower): ball at 0.15 or 0.12
    against 0.6 (ratios 3.1, 3.8), x' at most 1.6e-2 h outside F' (h = 0.48)."""
    p1, p2 = sphere_paths(0.2, 0.8, 0.7)[::order]
    r = check(pair(p1, p2), p1, p2)
    assert_pairing(r, sliver=10 * MERGE_TOL)
    assert r["over"] < 0.03


def cylinder_meshes(h1, h2, R=0.5):
    """An open interface: the lateral surface of the cylinder r < R, 0 < z < 1, and
    the box (-1, 1)^2 x (0, 1) minus it, meshed on their own (bc "interface")."""
    cyl = lambda: Cylinder(Pnt(0, 0, -1), Pnt(0, 0, 2), R).bc("interface")
    g1 = CSGeometry()
    g1.Add(cyl() * Plane(Pnt(0, 0, 0), Vec(0, 0, -1)).bc("bottom") * Plane(Pnt(0, 0, 1), Vec(0, 0, 1)).bc("top"))
    g2 = CSGeometry()
    g2.Add(OrthoBrick(Pnt(-1, -1, 0), Pnt(1, 1, 1)).bc("outer") - cyl().maxh(h2))
    return Mesh(g1.GenerateMesh(maxh=h1)), Mesh(g2.GenerateMesh(maxh=max(h2, 0.4)))


def cylinder(X, Y, Z):
    return sqrt(X * X + Y * Y) - 0.5


@pytest.mark.parametrize("h1,h2", [(0.3, 0.3), (0.2, 0.3)])
def test_open_surface(h1, h2):
    """The rims (z = 0, 1) of the two sides' image surfaces differ; the strips of
    depth GAP_TOL h beyond the rim edges pair them. Measured: no slivers dropped
    (per side 1.4e-16 relative); x' at most 5.5e-3 h outside F'."""
    p1, p2 = two_mesh_paths(*cylinder_meshes(h1, h2), phi=cylinder)
    for a, b in ((p1, p2), (p2, p1)):
        I = pair(a, b)
        r = check(I, a, b)
        assert_pairing(r, sliver=10 * MERGE_TOL, side_tiling=1e-10)
        assert np.all(I.segments().m == 1)
        assert r["over"] < 0.01


def test_unpaired_piece_raises():
    """Side 2 covers only part of Gamma (its faces with x > 0.5 removed)."""
    p1, p2 = sphere_paths(0.5, 0.35)
    F2 = BitArray(p2.facets)
    for f in p2.mesh.facets:
        if F2[f.nr] and np.mean([p2.mesh[v].point[0] for v in f.vertices]) > 0.5:
            F2[f.nr] = False
    q2 = type("P", (), dict(mesh=p2.mesh, elements=p2.elements, facets=F2, target=p2.target, sigma=p2.sigma))
    with pytest.raises(Exception, match=r"unpaired piece of Gamma_h\^1 \(side 1\).*uncovered area"):
        pair(p1, q2)
    with pytest.raises(Exception, match=r"unpaired piece of Gamma_h\^2 \(side 2\)"):
        pair(q2, p1)                                           # and the other way round


def slab_mesh(t, h):
    """The slab (0, 1)^2 x (0, t) (material "slab") in the box (0, 1)^2 x (-0.3, t + 0.3)."""
    geo = CSGeometry()
    s = OrthoBrick(Pnt(0, 0, 0), Pnt(1, 1, t))
    geo.Add(s.mat("slab"))
    geo.Add((OrthoBrick(Pnt(0, 0, -0.3), Pnt(1, 1, t + 0.3)) - s).mat("out"))
    return Mesh(geo.GenerateMesh(maxh=h))


def slab_side(mesh, mat, t, sheet=None):
    """D_h = the elements of material mat; Gamma_h = the faces between the slab and
    the rest (sheet "top" or "bottom": only those at z = t or z = 0); identity paths."""
    D = BitArray([el.mat == mat for el in mesh.Elements(VOL)])
    F = BitArray(mesh.nfacet)
    F.Clear()
    for f in mesh.facets:
        els = [e.nr for e in f.elements]
        if len(els) == 2 and D[els[0]] != D[els[1]]:
            top = np.mean([mesh[v].point[2] for v in f.vertices]) > t / 2
            F[f.nr] = sheet is None or top == (sheet == "top")
    return type("P", (), dict(mesh=mesh, elements=D, facets=F, target=CF((x, y, z)), sigma=CF((0, 0, 1))))


@pytest.mark.parametrize("t", [0.3, 0.15])
def test_thin_slab(t):
    """Gamma = two sheets t < h apart: the slab (h = 0.37, 0.29) against the rest of
    the box (another mesh). Faces of the two sides on one sheet have opposed outward
    normals, on the other sheet equal ones: each face is paired on its own sheet only
    (before, the other sheet's faces were candidates too: a false overlap)."""
    m1, m2 = slab_mesh(t, 0.33), slab_mesh(t, 0.25)
    p1, p2 = slab_side(m1, "slab", t), slab_side(m2, "out", t)
    for a, b in ((p1, p2), (p2, p1)):
        I = pair(a, b)
        assert_pairing(check(I, a, b))
        for side in (1, 2):
            R = I._rule(side, 4)
            assert np.abs(R["xp"] - R["xbar"]).max() < 1e-14     # identity paths: x' = xbar = x


@pytest.mark.parametrize("mat2", ["out", "slab"])
def test_other_sheet_raises(mat2):
    """Side 1 the bottom sheet of the slab, side 2 only the top sheet (t = 0.15 < h):
    nothing pairs the bottom. With D_h^2 the rest of the box the top faces' outward
    normals equal side 1's (the faces near are all rejected by orientation: uncovered
    area, the two sides not on opposite sides of Gamma); with D_h^2 the slab they
    are opposed and the path lines (sigma = e_z) of the top pass through xbar, but
    the paths of the partners end t away from it."""
    t = 0.15
    p1 = slab_side(slab_mesh(t, 0.33), "slab", t, "bottom")
    p2 = slab_side(slab_mesh(t, 0.25), mat2, t, "top")
    msg = ("uncovered area .* must lie on opposite sides of Gamma" if mat2 == "out"
           else "the path of its partner .* > the gap tolerance")
    with pytest.raises(Exception, match=r"unpaired piece of Gamma_h\^1 \(side 1\).*" + msg):
        pair(p1, p2)


def tilted_side(shift, h, d=0.05, seed=0):
    """Discontinuous paths: the box (0, 1)^2 x (-0.3, 0.3) split at z = -d, D_h the
    part below, Gamma_h its top; straight paths to Gamma = {z = 0} with direction
    sigma = (t_K, 1), t_K constant per element K, |t_K| d = shift (random directions)
    on the elements with centroid in (0.2, 0.8)^2 (the rim stays in place), else 0."""
    geo = CSGeometry()
    b = OrthoBrick(Pnt(0, 0, -0.3), Pnt(1, 1, -d))
    geo.Add(b.mat("below"))
    geo.Add((OrthoBrick(Pnt(0, 0, -0.3), Pnt(1, 1, 0.3)) - b).mat("above"))
    mesh = Mesh(geo.GenerateMesh(maxh=h))
    D = BitArray([el.mat == "below" for el in mesh.Elements(VOL)])
    F = BitArray(mesh.nfacet)
    F.Clear()
    for f in mesh.facets:
        els = [e.nr for e in f.elements]
        F[f.nr] = len(els) == 2 and D[els[0]] != D[els[1]]
    tilt = GridFunction(L2(mesh, order=0) ** 2)
    rng = np.random.default_rng(seed)
    for el in mesh.Elements(VOL):
        c = np.mean([mesh[v].point for v in el.vertices], axis=0)
        if D[el.nr] and max(abs(c[0] - 0.5), abs(c[1] - 0.5)) < 0.3:
            t = rng.normal(size=2)
            tilt.vec.FV().NumPy()[[el.nr, mesh.ne + el.nr]] = shift / d / np.linalg.norm(t) * t
    sigma = CF((tilt[0], tilt[1], 1))
    return type("P", (), dict(mesh=mesh, elements=D, facets=F, target=CF((x, y, z)) - z * sigma, sigma=sigma))


@pytest.mark.parametrize("shift", [0.03, 0.04])
def test_absorption(shift):
    """The D14 absorption in 3D: the bottom of the slab (identity paths, h = 0.37)
    against tilted_side (h = 0.25; the path ends of a vertex's elements differ by up
    to 2 shift). Where no point of F' has its path line through xbar, x' is the point
    of F', at most GAP_TOL h outside it, whose line passes closest. Measured at shift
    0.03 = 0.08 h: 76 and 98 of about 2370 points (order 6, side 1 and 2) absorbed,
    missing by up to 2.4e-2 h and 3.6e-2 h; at 0.04 the miss exceeds GAP_TOL h."""
    p1 = slab_side(slab_mesh(0.3, 0.33), "slab", 0.3, "bottom")
    p2 = tilted_side(shift, 0.25)
    if shift > 0.035:
        with pytest.raises(Exception, match=r"unpaired piece of Gamma_h\^1 \(side 1\)"):
            pair(p1, p2)
        return
    I = pair(p1, p2)
    r = check(I, p1, p2)
    assert np.all(I.segments().m == 1)                         # affine on each face: the spread is absorbed
    assert r["defect"] <= 0 and r["side_tiling"] < 1e-12 and r["inref"] < 1e-12 and r["onface"] < 1e-12
    assert r["over"] <= GAP_TOL * (1 + 1e-12)                  # x' at most GAP_TOL h outside F'
    for side in (1, 2):
        R = I._rule(side, 6)
        miss = np.linalg.norm(np.cross(R["xbar"] - R["xp"], R["sigmap"]), axis=1)
        assert np.sum(miss > 1e-10) > 0.01 * len(miss)         # the absorption is reached
        assert miss.max() <= GAP_TOL * r["h"]


def carved_mesh(kind, n, width=3.1):
    """The background mesh of a box about (-1.5, 1.5)^3: structured, n^3 cubes of 6 tetrahedra,
    the box of that width with its corner at (-1.53, -1.56, -1.55) (off the centre of the
    sphere), or netgen, maxh n."""
    if kind == "structured":
        return MakeStructured3DMesh(hexes=False, nx=n, ny=n, nz=n, mapping=lambda X, Y, Z: (
            width * X - 1.53, width * Y - 1.56, width * Z - 1.55))
    geo = CSGeometry()
    geo.Add(OrthoBrick(Pnt(-1.5, -1.5, -1.5), Pnt(1.5, 1.5, 1.5)))
    return Mesh(geo.GenerateMesh(maxh=n))


def carved_paths(kind, n, width=3.1):
    """Carved sides of one background mesh: D_h^i the elements inside / outside the unit
    sphere, Gamma_h^i their boundaries in the mesh, quasi-normal paths of length O(h)."""
    from ngstpm import unfitted_domain
    mesh = carved_mesh(kind, n, width)
    paths = []
    for side in (1, 2):
        D, F, _ = unfitted_domain(mesh, ball, side)
        paths.append(transfer_paths(mesh, ball, D, facets=F))
    return paths


def face_integral(P, f, n=6):
    """The integral of the numpy function f over the triangles P (k, 3, 3): Gauss-Legendre on
    the square collapsed to the triangle (exact for degree <= 2 n - 2)."""
    g, w = np.polynomial.legendre.leggauss(n)
    g, w = (g + 1) / 2, w / 2
    u, v = np.meshgrid(g, g, indexing="ij")
    W = np.outer(w, w) * (1 - u)                               # x = P0 + u (P1 - P0) + v (1 - u) (P2 - P0)
    X = P[:, None, None, 0] + u[..., None] * (P[:, None, None, 1] - P[:, None, None, 0]) \
        + (v * (1 - u))[..., None] * (P[:, None, None, 2] - P[:, None, None, 0])
    return (2 * tri_area(P) * np.einsum("kij,ij->k", f(X[..., 0], X[..., 1], X[..., 2]), W)).sum()


def test_carved_pairs():
    """Carved sides (structured, n = 6, h = 0.62): the path map is 0.22 h (side 1) from affine on
    a face, so the faces are subdivided: m = 4, 8 on side 1 (10, 38 of 48 faces), 2, 4 on side 2
    (34, 110 of 144). Measured: per face up to 5.6e-10 of the area in slivers below MERGE_TOL h^2
    dropped (2.5e-10 h^2; small overlaps of subtriangles at their corners), per side 4.4e-11
    relative; x' at most 4.6e-3 h outside F'; no point absorbed. Integrate on the pieces = the
    integral over the faces of Gamma_h^i (numpy), up to the dropped slivers."""
    p1, p2 = carved_paths("structured", 6)
    I = pair(p1, p2)
    r = check(I, p1, p2)
    assert_pairing(r, sliver=10 * MERGE_TOL, side_tiling=1e-10)
    assert r["over"] < 0.01
    S = I.segments()
    for i, p in enumerate((p1, p2)):
        m = {f: k for f, k in zip(S.facet[:, i], S.m[:, i])}
        assert len(m) == sum(p.facets) and set(m.values()) <= {1, 2, 4, 8, 16}
        assert max(m.values()) > 1
        fd = face_data(p.mesh, p.facets)
        P = np.array(list(fd.values()))
        dropped = abs(tri_area(P).sum() - S.area[:, i].sum())
        for cf, f in ((CF(1), lambda X, Y, Z: 1 + 0 * X), (x ** 5 + x * y * y * z * z, lambda X, Y, Z: X ** 5 + X * Y * Y * Z * Z)):
            ref = face_integral(P, f)
            scale = face_integral(P, lambda X, Y, Z: np.abs(f(X, Y, Z)))
            fmax = np.abs(f(*P.reshape(-1, 3).T)).max()           # |f| at the vertices (|x| < 1.5)
            assert abs(Integrate(cf * I.ds(side=i + 1), p.mesh) - ref) < 1e-12 * scale + 2 * fmax * dropped


def folded_side(c, h=0.25, d=0.05, r=0.2):
    """As tilted_side, with sigma = (s, 0, 1), d s = -c (x - 1/2) g, g a Gaussian of width r at
    (1/2, 1/2): the path end of x is x + d s = x - c (x - 1/2) g, whose x-derivative at the
    centre is 1 - c. For c > 1 the paths fold."""
    geo = CSGeometry()
    b = OrthoBrick(Pnt(0, 0, -0.3), Pnt(1, 1, -d))
    geo.Add(b.mat("below"))
    geo.Add((OrthoBrick(Pnt(0, 0, -0.3), Pnt(1, 1, 0.3)) - b).mat("above"))
    mesh = Mesh(geo.GenerateMesh(maxh=h))
    D = BitArray([el.mat == "below" for el in mesh.Elements(VOL)])
    F = BitArray(mesh.nfacet)
    F.Clear()
    for f in mesh.facets:
        els = [e.nr for e in f.elements]
        F[f.nr] = len(els) == 2 and D[els[0]] != D[els[1]]
    g = exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / r ** 2)
    sigma = CF((-c / d * (x - 0.5) * g, 0, 1))
    return type("P", (), dict(mesh=mesh, elements=D, facets=F, target=CF((x, y, z)) - z * sigma, sigma=sigma))


@pytest.mark.parametrize("c", [0.5, 1.5])
def test_folded_paths(c):
    """The bottom of the slab (identity paths) against folded_side(c). c = 0.5: no fold, but the
    path map is not affine on the faces: m = 1, 2, 4 on 16, 14, 8 of the 38 faces of side 2 (1 on
    side 1); measured: up to 9.5e-12 h^2 per face dropped in slivers (1.7e-12 of the
    area of a side), x' at most 7.9e-3 h
    outside F', no point absorbed. c = 1.5: the images of the subtriangles near the centre are
    reversed: pair raises a fold (both orders of the sides)."""
    p1 = slab_side(slab_mesh(0.3, 0.33), "slab", 0.3, "bottom")
    p2 = folded_side(c)
    if c > 1:
        with pytest.raises(Exception, match=r"the paths of Gamma_h\^2 fold at the face .* oriented opposite"):
            pair(p1, p2)
        with pytest.raises(Exception, match=r"the paths of Gamma_h\^1 fold at the face"):
            pair(p2, p1)
        return
    I = pair(p1, p2)
    assert_pairing(check(I, p1, p2), sliver=10 * MERGE_TOL, side_tiling=1e-11)
    S = I.segments()
    assert S.m[:, 0].max() == 1 and S.m[:, 1].max() > 1


@pytest.mark.parametrize("kind,n,msg", [("structured", 7, r"the paths of Gamma_h\^1 fold at the face .* same side"),
                                        ("netgen", 0.4, r"Gamma_h\^1 is not a surface .* \(4 faces meet\)")])
def test_carved_raises(kind, n, msg):
    """Carved sides where Gamma_h^1 folds or is not a surface. Structured, n = 7, box width 3.0
    (h = 0.51): faces of Gamma_h^1 whose outward normal is against the radial paths (here
    counted independently; n_F . sigma down to -0.41 at a vertex): seen along the paths the image
    of one of them and its neighbour's lie on the same side of their common edge. Netgen, maxh
    0.4 (h = 0.48): 4 faces of Gamma_h^1 meet at an edge."""
    paths = carved_paths(kind, n, 3.0)
    if kind == "structured":
        p = paths[0]
        D = p.elements
        against = 0
        for f in p.mesh.facets:
            if p.facets[f.nr]:
                P = np.array([p.mesh[v].point for v in f.vertices])
                el = next(e for e in f.elements if D[e.nr])
                c = np.mean([p.mesh[v].point for v in p.mesh[el].vertices], axis=0)
                nF = np.cross(P[1] - P[0], P[2] - P[0])
                nF *= np.sign(nF.dot(P[0] - c))
                against += np.min(P @ nF) < 0                    # radial paths x / |x| from inside
        assert against > 0
    with pytest.raises(Exception, match=msg):
        pair(*paths)

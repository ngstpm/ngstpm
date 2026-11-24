"""pair(p1, p2) in 3D (D10): pieces of the faces of Gamma_h^i by clipping in the
chart of each face, the per-face coverage check and the partner rules.

Tolerances (ngstpm.interface; in 3D relative to h = the larger of the two
sides' mean edge lengths of Gamma_h), measured on the configurations below
(sphere: ball and box minus ball meshed on their own; cylinder: open surface):
- pieces below MERGE_TOL h^2 = 1e-10 h^2 are dropped. Slivers that small
  appear where vertices of the two sides (nearly) coincide: on the sphere up to
  3.0e-11 h^2 (h1 = 0.2, h2 = 0.6; elsewhere < 1e-17 h^2), at the rims of the
  cylinder 9.9e-11 h^2, and with vertices moved by eps h from 1e-10 h^2 per
  face. A side's pieces tile Gamma_h^i to 5e-15 relative without such slivers,
  6.6e-13 (sphere), 4.7e-12 (cylinder) and 1.7e-10 (eps = 1e-10) with them;
- vertex images of one side (one per face at the vertex) differ by at most
  8.2e-9 h (sphere; measured with a temporary diagnostic);
- the partner x' of a rule point lies at most 1.5e-2 h outside its face F' on
  the sphere at h ratios 1 to 3.8 (h = 0.47, 0.48), 2.7e-2 h with a coarse side of
  58 faces (h = 0.70), 5.5e-3 h on the cylinder: GAP_TOL = 5e-2 h leaves a margin
  of 1.9 in the worst case. The rim strips (depth GAP_TOL h) cover at most
  1e-3 h^2 per face on the cylinder.
"""

from functools import lru_cache

import numpy as np
import pytest
from netgen.csg import CSGeometry, Cylinder, OrthoBrick, Plane, Pnt, Sphere, Vec
from ngsolve import BitArray, CF, Integrate, Mesh, VOL, ds, sqrt, x, y, z
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
    MERGE_TOL h^2 each; 2.4e-10 h^2 at eps = 1e-4), per side 1.7e-10 relative
    (eps = 1e-10); x' at most 8.8e-5 h outside F' (eps = 1e-3)."""
    m = fitted_mesh(0.5)
    p1, p2 = fitted_paths(0.5, m, moved_copy(m, eps))
    r = check(pair(p1, p2), p1, p2)
    assert_pairing(r, sliver=20 * MERGE_TOL, side_tiling=1e-9)
    assert r["over"] < 1e-3


@pytest.mark.parametrize("order", [1, -1])
def test_h_ratio(order):
    """Very different h: the ball at maxh 0.2, the box minus the ball at 0.8 with
    curvaturesafety 0.7 (mean edges of Gamma_h 0.209 and 0.698, ratio 3.3), both
    orders of the sides. Measured: x' at most 2.7e-2 h outside F' (h = 0.70; the
    coarse side has 58 faces). Not in the suite (slower): ball at 0.15 or 0.12
    against 0.6 (ratios 3.1, 3.8), x' at most 1.4e-2 h outside F' (h = 0.48)."""
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
    depth GAP_TOL h beyond the rim edges pair them (strips cover up to 1e-3 h^2
    per face). Measured: slivers up to 9.9e-11 h^2 dropped at the rims, per side
    4.7e-12 relative (h1 = h2 = 0.3); x' at most 5.5e-3 h outside F'."""
    p1, p2 = two_mesh_paths(*cylinder_meshes(h1, h2), phi=cylinder)
    for a, b in ((p1, p2), (p2, p1)):
        r = check(pair(a, b), a, b)
        assert_pairing(r, sliver=10 * MERGE_TOL, side_tiling=1e-10)
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
    normals equal side 1's (no candidates: uncovered area); with D_h^2 the slab they
    are opposed and the path lines (sigma = e_z) of the top pass through xbar, but
    the paths of the partners end t away from it."""
    t = 0.15
    p1 = slab_side(slab_mesh(t, 0.33), "slab", t, "bottom")
    p2 = slab_side(slab_mesh(t, 0.25), mat2, t, "top")
    msg = "uncovered area" if mat2 == "out" else "the path of its partner .* > the gap tolerance"
    with pytest.raises(Exception, match=r"unpaired piece of Gamma_h\^1 \(side 1\).*" + msg):
        pair(p1, p2)

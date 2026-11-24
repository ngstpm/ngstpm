"""pair(p1, p2) in 3D (D10): pieces of the faces of Gamma_h^i by clipping in the
chart of each face, the per-face coverage check and the partner rules.

Tolerances (ngstpm.interface; in 3D relative to h = the larger of the two
sides' mean edge lengths of Gamma_h), measured on the configurations below:
- pieces below MERGE_TOL h^2 = 1e-10 h^2 are dropped; on the two-mesh sphere
  the largest dropped sliver was 3.0e-11 h^2 (h1 = 0.2, h2 = 0.6), so the
  pieces of a side tile Gamma_h^i to 6.6e-13 relative there and to 5e-15
  elsewhere;
- vertex images of one side (one per face at the vertex) differ by at most
  8.2e-9 h (two-mesh sphere);
- the partner x' of a rule point lies at most 1.5e-2 h outside its face F'
  (two-mesh sphere, h1 = 0.2, h2 = 0.6; 1.2e-2 h at h ratio 1 to 1.7), against
  GAP_TOL = 5e-2 h.
"""

import numpy as np
import pytest
from netgen.csg import CSGeometry, OrthoBrick, Pnt, Sphere
from ngsolve import BitArray, CF, Integrate, Mesh, VOL, ds, sqrt, x, y, z
from ngsolve.meshes import MakeStructured3DMesh

from ngstpm import pair, transfer_paths
from ngstpm.interface import GAP_TOL, MERGE_TOL


def ball(X, Y, Z):
    return sqrt(X * X + Y * Y + Z * Z) - 1


def sphere_meshes(h1, h2):
    """The unit ball (maxh h1) and the box minus the ball (maxh h2 on the sphere),
    meshed on their own: the bc "interface" on both."""
    g1 = CSGeometry()
    g1.Add(Sphere(Pnt(0, 0, 0), 1).bc("interface"))
    g2 = CSGeometry()
    g2.Add(OrthoBrick(Pnt(-1.5, -1.5, -1.5), Pnt(1.5, 1.5, 1.5)).bc("outer")
           - Sphere(Pnt(0, 0, 0), 1).bc("interface").maxh(h2))
    return Mesh(g1.GenerateMesh(maxh=h1)), Mesh(g2.GenerateMesh(maxh=max(h2, 0.5)))


def two_mesh_paths(m1, m2, phi=ball):
    return [transfer_paths(m, phi, facets=m.Boundaries("interface")) for m in (m1, m2)]


def fitted_paths(h):
    """One mesh fitted to the sphere: D_h^1 the ball, D_h^2 the rest of the box."""
    geo = CSGeometry()
    s = Sphere(Pnt(0, 0, 0), 1)
    geo.Add(s.mat("ball"))
    geo.Add((OrthoBrick(Pnt(-1.5, -1.5, -1.5), Pnt(1.5, 1.5, 1.5)) - s).mat("out"))
    mesh = Mesh(geo.GenerateMesh(maxh=h))
    D1 = BitArray([el.mat == "ball" for el in mesh.Elements(VOL)])
    return transfer_paths(mesh, ball, D1), transfer_paths(mesh, ball, ~D1)


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
        lam = np.array([np.linalg.lstsq(e, d, rcond=None)[0] for e, d in zip(E, R["xp"] - G[:, 0])])
        onface = max(onface, np.abs(G[:, 0] + np.einsum("nij,nj->ni", E, lam) - R["xp"]).max())
        bary = np.column_stack([1 - lam.sum(1), lam])
        hgt = np.stack([2 * tri_area(G) / np.linalg.norm(G[:, (k + 2) % 3] - G[:, (k + 1) % 3], axis=1)
                        for k in range(3)], axis=1)
        over = max(over, (np.maximum(-bary, 0) * hgt).max() / h)
        par = max(par, np.linalg.norm(np.cross(R["xbar"] - R["xp"], R["sigmap"]), axis=1).max())
    return dict(nseg=n, h=h, defect=defect, side_tiling=side_tiling, inref=inref, onface=onface, over=over, par=par)


def assert_pairing(r, sliver=0.0):
    """sliver: the area (in h^2) of the slivers that may be dropped on a face."""
    assert r["defect"] <= sliver           # per face: |area of the pieces - |F|| <= 1e-12 |F| + sliver h^2
    assert r["side_tiling"] < 1e-12        # per side, relative
    assert r["inref"] < 1e-12            # the triangles of a piece lie in the reference triangle of F
    assert r["onface"] < 1e-12           # tri_x on F, x' on the plane of F'
    assert r["over"] < GAP_TOL           # x' in F', up to the gap
    assert r["par"] < 1e-10              # xbar - x' parallel to sigma(x')


@pytest.mark.parametrize("h1,h2", [(0.5, 0.5), (0.5, 0.35), (0.35, 0.5)])
def test_two_meshes(h1, h2):
    """Ball and box-minus-ball meshed on their own: each side's pieces tile its
    Gamma_h; with equal maxh netgen meshes the sphere alike, and each face is
    paired with its twin only."""
    m1, m2 = sphere_meshes(h1, h2)
    p1, p2 = two_mesh_paths(m1, m2)
    I = pair(p1, p2)
    r = check(I, p1, p2)
    assert_pairing(r, sliver=10 * MERGE_TOL)
    assert r["nseg"] >= max(sum(p1.facets), sum(p2.facets))
    if h1 == h2:
        assert r["nseg"] == sum(p1.facets) == sum(p2.facets)


def test_fitted_one_mesh():
    """One mesh fitted to the sphere, D_h^1 and D_h^2 on either side: Gamma_h^1 =
    Gamma_h^2, each face paired with itself, x' = x up to the difference of the two
    sides' path fields on a face (2.2e-10 = 4.7e-10 h; 1.3e-14 for equal paths, below)."""
    p1, p2 = fitted_paths(0.5)
    I = pair(p1, p2)
    assert_pairing(check(I, p1, p2))
    S = I.segments()
    assert I.nsegments == sum(p1.facets) and np.all(S.facet[:, 0] == S.facet[:, 1])
    for side in (1, 2):
        R = I._rule(side, 8)
        assert np.abs(R["xp"] - R["x"]).max() < 1e-9


def test_equal_copies():
    """Two copies of one mesh (two meshes for pair): each face paired with its copy, x' = x."""
    m = Mesh(sphere_meshes(0.5, 0.5)[0].ngmesh)
    m2 = Mesh(m.ngmesh.Copy())
    p1, p2 = two_mesh_paths(m, m2)
    I = pair(p1, p2)
    assert_pairing(check(I, p1, p2))
    S = I.segments()
    assert I.nsegments == sum(p1.facets) and np.all(S.facet[:, 0] == S.facet[:, 1])
    assert np.all(S.area > 0)
    for side in (1, 2):
        R = I._rule(side, 8)
        assert np.abs(R["xp"] - R["x"]).max() < 1e-10


POLYS = [CF(1), x, y * z, x * x * y, x * y * z * z + y ** 3, x ** 5 + x * y * y * z * z]


@pytest.mark.parametrize("h1,h2", [(0.5, 0.35), (0.35, 0.5)])
def test_integrate_exact(h1, h2):
    """Integrate(p * I.ds(side=i)) = the integral over Gamma_h^i (NGSolve's ds on
    the bc "interface") for polynomials p of degree <= 5, the order of the rule."""
    meshes = sphere_meshes(h1, h2)
    I = pair(*two_mesh_paths(*meshes))
    for side, m in zip((1, 2), meshes):
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

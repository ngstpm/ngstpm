"""TransferPath: one side's paths from Gamma_h to Gamma, by kind."""

import sys
from pathlib import Path

import numpy as np
import pytest
from ngsolve import Integrate, atan2, cos, dx, sqrt, x, y

from ngstpm import ParametricCurve, boundary_facets, normal, unfitted_domain
from ngstpm.paths import transfer_paths

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson_tpm import background_mesh  # noqa: E402


def disk(X, Y):
    return sqrt(X * X + Y * Y) - 1


def flower(X, Y):
    return sqrt(X * X + Y * Y) - 1 - 0.12 * cos(5 * atan2(Y, X))


def flower_xy(t):                       # the zero set of flower, t the polar angle
    r = 1 + 0.12 * np.cos(5 * t)
    return r * np.cos(t), r * np.sin(t)


def gamma_h_mean(mesh, D, f):
    """Mean of the CF f over the faces of Gamma_h of D, seen from D."""
    ind = boundary_facets(mesh, D)[1]
    dG = dx(element_boundary=True, definedonelements=D, bonus_intorder=4)
    return Integrate(ind * f * dG, mesh) / Integrate(ind * dG, mesh)


def test_1_quasi_normal_disk_ends_on_gamma():
    """|phi(target)| ~ 0 on Gamma_h; target_normal is the radial unit vector."""
    mesh = background_mesh(0.3)
    D = unfitted_domain(mesh, disk)[0]
    p = transfer_paths(mesh, disk, D)
    assert p.kind == "quasi-normal" and p.elements is D and p.mesh is mesh and p.geo is disk
    assert sqrt(gamma_h_mean(mesh, D, disk(*p.target) ** 2)) < 1e-7
    assert gamma_h_mean(mesh, D, p.ell) > 0                                 # x in {phi < 0}, sigma outwards
    radial = p.target / sqrt(p.target * p.target)
    assert sqrt(gamma_h_mean(mesh, D, (p.target_normal - radial) ** 2)) < 1e-9


@pytest.mark.parametrize("side", [1, 2])
def test_2_closest_flower_ends_on_curve(side):
    """Closest-point paths of a ParametricCurve end on it (its level set
    vanishes there), and sigma is the normal of the curve at the target."""
    mesh = background_mesh(0.15)
    geo = ParametricCurve(flower_xy)
    geo, phi = (geo, flower) if side == 1 else (-geo, lambda *X: -flower(*X))
    D = unfitted_domain(mesh, geo)[0]
    p = transfer_paths(mesh, geo, D)
    assert p.kind == "closest"
    assert sqrt(gamma_h_mean(mesh, D, phi(*p.target) ** 2)) < 1e-5
    assert sqrt(gamma_h_mean(mesh, D, (p.sigma - p.target_normal) ** 2)) == 0.0
    assert sqrt(gamma_h_mean(mesh, D, (p.sigma - normal(phi, *p.target)) ** 2)) < 1e-4


def test_3_unsupported_kind_raises():
    mesh = background_mesh(0.3)
    D = unfitted_domain(mesh, disk)[0]
    with pytest.raises(ValueError, match="quasi-normal"):
        transfer_paths(mesh, disk, D, kind="closest")
    geo = ParametricCurve(flower_xy)
    with pytest.raises(ValueError, match="closest"):
        transfer_paths(mesh, geo, D, kind="quasi-normal")


def test_6_data_at_target():
    """g(*p.target) is g of the coordinates of the end of the path."""
    mesh = background_mesh(0.3)
    D = unfitted_domain(mesh, flower)[0]
    p = transfer_paths(mesh, flower, D)
    g = lambda X, Y: X * X - 2 * X * Y + Y
    assert len(list(p.target)) == 2
    xb = (x + p.ell * p.sigma[0], y + p.ell * p.sigma[1])
    assert gamma_h_mean(mesh, D, (g(*p.target) - g(*xb)) ** 2) == 0.0


def disk_mesh(maxh=0.25):
    """A mesh of the unit disk alone (straight elements), its boundary bc "interface"."""
    from netgen.geom2d import SplineGeometry
    from ngsolve import Mesh
    g = SplineGeometry()
    g.AddCircle((0, 0), 1, bc="interface")
    return Mesh(g.GenerateMesh(maxh=maxh))


def bnd_facets(mesh, bc):
    from ngsolve import BND, BitArray
    F = BitArray(mesh.nfacet)
    F.Clear()
    for el in mesh.Elements(BND):
        if el.mat == bc:
            F[el.edges[0].nr] = True
    return F


def test_7_facets():
    """p.facets defaults to boundary_facets(mesh, D); D defaults to the whole
    mesh, and Gamma_h may be given as BND facets (one side of two meshes): the
    paths then start there and end on Gamma."""
    mesh = background_mesh(0.3)
    D = unfitted_domain(mesh, disk)[0]
    p = transfer_paths(mesh, disk, D)
    assert list(p.facets) == list(boundary_facets(mesh, D)[0]) and sum(p.facets) > 10
    mesh = disk_mesh()
    F = bnd_facets(mesh, "interface")
    for geo, phi in ((disk, disk), (ParametricCurve(lambda t: (np.cos(t), np.sin(t))), disk)):
        p = transfer_paths(mesh, geo, facets=F)
        assert sum(p.elements) == mesh.ne and p.facets is F
        dB = dx(element_boundary=True, bonus_intorder=4)
        res = sqrt(Integrate(bnd_ind(mesh, F) * phi(*p.target) ** 2 * dB, mesh))
        assert res < 1e-7                       # quasi-normal 9.2e-9, closest 4.2e-11 (L2 norm on Gamma_h)


def bnd_ind(mesh, F):
    from ngstpm.geometry import _facet_indicator
    return _facet_indicator(mesh, F)


def wave(X, Y):                         # the graph y = 0.5 + 0.05 sin(2 pi x) in the unit square
    from ngsolve import pi, sin
    return 0.5 + 0.05 * sin(2 * pi * X) - Y


@pytest.mark.parametrize("side", [1, 2])
def test_8_vertical(side):
    """kind="vertical": sigma = (0, -1) (the sign of d phi/dy, from Omega_1 to Omega_2) on both
    sides, so the path keeps x; the target is on Gamma and target_normal is the normal of Gamma."""
    from stokes_darcy_curved import square
    mesh = square(0.1)
    D = unfitted_domain(mesh, wave, side=side)[0]
    p = transfer_paths(mesh, wave, D, kind="vertical")
    assert p.kind == "vertical"
    assert gamma_h_mean(mesh, D, p.sigma[0] ** 2 + (p.sigma[1] + 1) ** 2) < 1e-28     # sigma_2: L2 projection of -1
    assert gamma_h_mean(mesh, D, (p.target[0] - x) ** 2) == 0.0
    assert (gamma_h_mean(mesh, D, p.ell) > 0) == (side == 1)
    assert sqrt(gamma_h_mean(mesh, D, wave(*p.target) ** 2)) < 1e-10
    from ngsolve import CF, cos, pi
    gp = 0.1 * pi * cos(2 * pi * p.target[0])                               # gamma'(x) at the target
    exact = -CF((-gp, 1)) / sqrt(1 + gp * gp)                               # from Omega_1 (above) to Omega_2
    assert sqrt(gamma_h_mean(mesh, D, (p.target_normal - exact) ** 2)) < 1e-8   # normal(): central differences


@pytest.mark.parametrize("maxh", [0.1, 0.05])
def test_9_vertical_area(maxh):
    """|D_h^i| + the region swept by the vertical paths of each side = 1/2 on the unfitted mesh."""
    from ngsolve import CF
    from ngstpm import pair
    from stokes_darcy_curved import square
    mesh = square(maxh)
    Ds = [unfitted_domain(mesh, wave, side=s)[0] for s in (1, 2)]
    paths = [transfer_paths(mesh, wave, D, kind="vertical") for D in Ds]
    I = pair(*paths)
    for i, (D, p) in enumerate(zip(Ds, paths)):
        area = (Integrate(CF(1) * dx(definedonelements=D), mesh)
                + Integrate(CF(1).RegionIntegral(p) * I.ds(side=i + 1), mesh))
        assert abs(area - 0.5) < 1e-9


def kidney(X, Y):
    r2 = (X + 0.5) ** 2 + Y * Y
    return (2 * r2 - (X + 0.5)) ** 2 - r2 + 0.1


@pytest.mark.parametrize("maxh,bound", [(0.2, 5e-2), (0.1, 2e-3), (0.05, 2e-5)])
def test_quasi_normal_targets_on_gamma(maxh, bound):
    """Pointwise Newton on the faces of Gamma_h: also where the level set has a critical
    point next to Gamma_h (the kidney at maxh = 0.2, where the fields projected over the
    whole element left the targets off Gamma by more than an element size). Observed
    residuals (element sizes): 1.4e-2, 4.5e-4, 2.4e-6."""
    from ngstpm import unfitted_domain
    mesh = background_mesh(maxh)
    p = transfer_paths(mesh, kidney, unfitted_domain(mesh, kidney)[0])
    assert p.residual < bound


def test_quasi_normal_raises():
    from ngstpm import unfitted_domain
    mesh = background_mesh(0.3)
    D = unfitted_domain(mesh, disk)[0]
    with pytest.raises(RuntimeError, match="Newton did not converge|tangent|vanishes"):
        transfer_paths(mesh, lambda X, Y: 1 + X * X + 0 * Y, D)       # no zero along the rays
    with pytest.raises(RuntimeError, match="off Gamma"):
        transfer_paths(mesh, disk, D, order=1, check=1e-9)            # a fit too coarse for the check
    assert transfer_paths(mesh, disk, D, order=1, check=False).residual > 1e-9


def test_level_set_of_its_arguments_only():
    """A level set that reads the point (x, y) instead of its arguments raises; it used to
    be evaluated at a freed mapped point (segmentation fault)."""
    from ngstpm import unfitted_domain
    mesh = background_mesh(0.3)
    D = unfitted_domain(mesh, disk)[0]
    with pytest.raises(ValueError, match="arguments only"):
        transfer_paths(mesh, lambda X, Y: sqrt(X * X + Y * Y) - 1 - 0.3 * x, D)


@pytest.mark.parametrize("c", [100.0, 1000.0])
def test_quasi_normal_shifted_circle(c):
    """Far from the origin the rounding of phi exceeds an absolute tolerance: Newton
    stops at rounding. Unit circle centred at (c, c), expanded (the worst conditioned form)."""
    from netgen.geom2d import SplineGeometry
    from ngsolve import Mesh
    from ngstpm import unfitted_domain
    geo = SplineGeometry()
    geo.AddRectangle((c - 1.3, c - 1.3), (c + 1.3, c + 1.3))
    mesh = Mesh(geo.GenerateMesh(maxh=0.2))
    phi = lambda X, Y: X * X - 2 * c * X + Y * Y - 2 * c * Y + 2 * c * c - 1
    D = unfitted_domain(mesh, phi)[0]
    assert 50 < D.NumSet() < mesh.ne
    p = transfer_paths(mesh, phi, D)
    assert p.residual < 1e-6            # observed: 2.1e-8 (c = 100), 4.5e-8 (c = 1000); 3.3e-9 at c = 0


A2 = np.array((1.0, 0.7))                 # Fable's ellipse (x / a)^2 + (y / b)^2 = 1; a = b = 1: the unit circle


def ellipse(a, b):
    """Level set, ParametricCurve and the two pointwise path maps d(X) = target(X) - X of the
    ellipse, each by its own Newton iteration (independent of transfer_paths)."""
    A = np.array((a, b))
    phi = lambda X, Y: sqrt(X * X / a**2 + Y * Y / b**2) - 1
    curve = ParametricCurve(lambda t: (a * np.cos(t), b * np.sin(t)))
    grad = lambda P: P / A**2 / np.sqrt(np.sum(P**2 / A**2))

    def ray(X):                         # along grad phi / |grad phi| at X, Newton on phi(X + l s) = 0
        g = grad(X)
        s, ell = g / np.linalg.norm(g), -(np.sqrt(np.sum(X**2 / A**2)) - 1) / np.linalg.norm(g)
        for _ in range(60):
            P = X + ell * s
            ell -= (np.sqrt(np.sum(P**2 / A**2)) - 1) / (grad(P) @ s)
        return ell * s

    def closest(X):                     # Newton on the curve parameter
        t = np.arctan2(X[1] / b, X[0] / a)
        for _ in range(60):
            g, dg, d2 = A * (np.cos(t), np.sin(t)), A * (-np.sin(t), np.cos(t)), -A * (np.cos(t), np.sin(t))
            t -= ((g - X) @ dg) / (dg @ dg + (g - X) @ d2)
        return A * (np.cos(t), np.sin(t)) - X
    return phi, curve, ray, closest


def surface_grad_error(mesh, p, dmap, nodes=5, eps=1e-5):
    """Largest |grad_d tau - (d(X + eps tau) - d(X - eps tau)) / (2 eps)| over Gauss points X of
    the faces of Gamma_h, tau the unit tangent of the face, d the pointwise path map dmap.
    grad_d is evaluated on the element of D (grad_d tau does not depend on the normal used)."""
    from ngsolve import VOL
    verts = np.array([v.point for v in mesh.vertices])
    t = 0.5 * (np.polynomial.legendre.leggauss(nodes)[0] + 1)
    worst = 0.0
    for el in mesh.Elements(VOL):
        if not p.elements[el.nr]:
            continue
        trafo = mesh.GetTrafo(el)
        p0, p1, p2 = (np.array(trafo(*xi).point) for xi in ((0, 0), (1, 0), (0, 1)))
        M = np.column_stack((p1 - p0, p2 - p0))
        for f in el.facets:
            if not p.facets[f.nr]:
                continue
            A, B = verts[[v.nr for v in mesh[f].vertices]]
            tau = (B - A) / np.linalg.norm(B - A)
            for ti in t:
                X = A + ti * (B - A)
                G = np.array(p.grad_d(trafo(*np.linalg.solve(M, X - p0)))).reshape(2, 2)
                fd = (dmap(X + eps * tau) - dmap(X - eps * tau)) / (2 * eps)
                worst = max(worst, np.linalg.norm(G @ tau - fd))
    return worst


def normal_part(mesh, p):
    """L2 norm over the faces of Gamma_h of grad_d n, n the face normal (the integration points
    are facet points, as in RegionIntegral)."""
    from ngsolve import InnerProduct, specialcf
    from ngstpm.geometry import _facet_indicator
    n = specialcf.normal(2)
    dG = dx(element_boundary=True, definedonelements=p.elements, bonus_intorder=4)
    Gn = p.grad_d * n
    return sqrt(Integrate(_facet_indicator(mesh, p.facets) * InnerProduct(Gn, Gn) * dG, mesh))


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("kind", ["closest", "quasi-normal"])
@pytest.mark.parametrize("b", [1.0, 0.7])
def test_surface_grad_d(b, kind, side):
    """grad_d is the surface gradient on Gamma_h, grad d (I - n n^T): its tangential part
    against central differences of the pointwise path map along the faces, its normal part
    zero. Unit disk (b = 1) and Fable's ellipse (b = 0.7), h = 0.1, order 6. Measured
    (tangential, side 1 / side 2): disk closest 2.6e-10 / 1.9e-10,
    quasi-normal 1.1e-9 / 9.2e-10; ellipse closest 4.3e-8 / 8.4e-9, quasi-normal
    1.3e-7 / 3.6e-8; normal part < 1e-15."""
    phi, curve, ray, closest = ellipse(1.0, b)
    geo = curve if kind == "closest" else phi
    if side == 2:
        geo = -curve if kind == "closest" else (lambda X, Y: -phi(X, Y))
    mesh = background_mesh(0.1)
    D = unfitted_domain(mesh, geo)[0]
    p = transfer_paths(mesh, geo, D, kind=kind)
    assert surface_grad_error(mesh, p, closest if kind == "closest" else ray) < 1e-6
    assert normal_part(mesh, p) < 1e-14


@pytest.mark.parametrize("side", [1, 2])
def test_surface_grad_d_vertical(side):
    """kind="vertical": d(x) = (0, gamma(x_1) - x_2) for the graph x_2 = gamma(x_1) (wave), so
    grad_d tau = (0, gamma'(x_1) tau_1 - tau_2) on Gamma_h. Measured (h = 0.1, order 6): 1.6e-8 (side 1),
    3.7e-8 (side 2); normal part < 1e-15."""
    from stokes_darcy_curved import square
    mesh = square(0.1)
    geo = wave if side == 1 else (lambda X, Y: -wave(X, Y))
    D = unfitted_domain(mesh, geo)[0]
    p = transfer_paths(mesh, geo, D, kind="vertical")
    dmap = lambda X: np.array((0.0, 0.5 + 0.05 * np.sin(2 * np.pi * X[0]) - X[1]))
    assert surface_grad_error(mesh, p, dmap) < 1e-6
    assert normal_part(mesh, p) < 1e-14


def test_surface_grad_d_ellipsoid():
    """3D, quasi-normal, the ellipsoid with semi-axes (1, 0.8, 0.9), h = 0.35, order 4: grad_d t
    against central differences of the pointwise ray map along the two edge directions t of
    each face of Gamma_h, at 4 points of the face; the normal part is zero. Measured: 1.5e-2
    (6.4e-2 at h = 0.5: the fit of the coarse 3D fields), normal part 2.2e-16."""
    from ngsolve import VOL, InnerProduct, specialcf
    from ngstpm.geometry import _facet_indicator
    from poisson_tpm_3d import background_mesh as background_mesh_3d
    A = np.array((1.0, 0.8, 0.9))
    phi = lambda X, Y, Z: sqrt(X * X / A[0]**2 + Y * Y / A[1]**2 + Z * Z / A[2]**2) - 1
    level = lambda P: np.sqrt(np.sum(P**2 / A**2)) - 1
    grad = lambda P: P / A**2 / np.sqrt(np.sum(P**2 / A**2))

    def ray(X):
        g = grad(X)
        s, ell = g / np.linalg.norm(g), -level(X) / np.linalg.norm(g)
        for _ in range(60):
            ell -= level(X + ell * s) / (grad(X + ell * s) @ s)
        return ell * s
    mesh = background_mesh_3d(0.35)
    D = unfitted_domain(mesh, phi)[0]
    p = transfer_paths(mesh, phi, D, order=4, check=False)
    verts = np.array([v.point for v in mesh.vertices])
    worst, eps = 0.0, 1e-5
    for el in mesh.Elements(VOL):
        faces = [f for f in el.facets if p.facets[f.nr]] if D[el.nr] else []
        if not faces:
            continue
        trafo = mesh.GetTrafo(el)
        p0, *pi = (np.array(trafo(*xi).point) for xi in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)))
        M = np.column_stack([q - p0 for q in pi])
        for f in faces:
            F = verts[[v.nr for v in mesh[f].vertices]]
            for lam in ((1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)):
                X = np.array(lam) @ F
                G = np.array(p.grad_d(trafo(*np.linalg.solve(M, X - p0)))).reshape(3, 3)
                for t in (F[1] - F[0], F[2] - F[0]):
                    t = t / np.linalg.norm(t)
                    worst = max(worst, np.linalg.norm(G @ t - (ray(X + eps * t) - ray(X - eps * t)) / (2 * eps)))
    assert worst < 4e-2
    n = specialcf.normal(3)
    Gn = p.grad_d * n
    dG = dx(element_boundary=True, definedonelements=D, bonus_intorder=2)
    assert sqrt(Integrate(_facet_indicator(mesh, p.facets) * InnerProduct(Gn, Gn) * dG, mesh)) < 1e-14


@pytest.mark.parametrize("kind,side,area", [("closest", 1, 0.44371217329203494), ("closest", 2, 0.5693245572258439),
                                            ("quasi-normal", 1, 0.4437121181386269),
                                            ("quasi-normal", 2, 0.5693245613852174)])
def test_region_area_with_surface_grad_d(kind, side, area):
    """RegionIntegral sees grad d only on tangent vectors: the swept areas of the ellipse at
    h = 0.2 (order 6) are those computed with the full gradient of 7c3d87f (at e6894fd)
    (tolerance 1e-11: the fits are least squares, LAPACK-dependent in the last digits)."""
    from ngsolve import CF
    from ngstpm import boundary_ds
    phi, curve, _, _ = ellipse(1.0, 0.7)
    geo = curve if kind == "closest" else phi
    if side == 2:
        geo = -curve if kind == "closest" else (lambda X, Y: -phi(X, Y))
    mesh = background_mesh(0.2)
    D = unfitted_domain(mesh, geo)[0]
    p = transfer_paths(mesh, geo, D, kind=kind)
    assert abs(Integrate(CF(1).RegionIntegral(p) * boundary_ds(mesh, D, bonus_intorder=10), mesh) - area) < 1e-11


def test_surface_grad_d_partner():
    """On an interface measure, grad_d of the other side is evaluated on the partner element,
    with the normal n' of the partner's face: it is G(x') (I - n' n'^T), G the exact gradient of
    the disk's path map d(X) = (1/|X| - 1) X at the partner point x' (with the own normal n
    instead of n' the difference is O(1)). Fable's check (fable5/g_item1.py, part 4)."""
    from ngsolve import CF, Id, InnerProduct, OuterProduct, specialcf
    from ngstpm import pair
    from ngstpm._ngstpm import SideEval
    mesh = background_mesh(0.2)
    Ds = [unfitted_domain(mesh, disk, side=s)[0] for s in (1, 2)]
    paths = [transfer_paths(mesh, disk, D) for D in Ds]
    I = pair(*paths)
    n = specialcf.normal(2)
    for i, o in ((0, 1), (1, 0)):
        npr = SideEval(n, None, mesh, Ds[o])                  # the normal as grad_d of side o sees it
        xp = SideEval(CF((x, y)), None, mesh, Ds[o])          # the partner point x'
        r = sqrt(xp[0] ** 2 + xp[1] ** 2)
        G = (1 / r - 1) * Id(2) - OuterProduct(xp, xp) / r ** 3
        dI = I.ds(side=i + 1, bonus_intorder=4)
        E = paths[o].grad_d - G * (Id(2) - OuterProduct(npr, npr))
        Eown = paths[o].grad_d - G * (Id(2) - OuterProduct(n, n))
        assert Integrate(InnerProduct(E, E) * dI, mesh) < 1e-10
        assert Integrate(InnerProduct(Eown, Eown) * dI, mesh) > 1e-1


def test_grad_d_facet_normal_none():
    """The legacy "facet-normal" paths live on the facets only: no grad_d, and RegionIntegral
    refuses them (their paths can miss Gamma: the area is O(h) wrong)."""
    from ngsolve import CF
    mesh = background_mesh(0.3)
    D = unfitted_domain(mesh, disk)[0]
    p = transfer_paths(mesh, disk, D, kind="facet-normal")
    assert p.grad_d is None
    with pytest.raises(NotImplementedError, match="miss Gamma"):
        CF(1).RegionIntegral(p)

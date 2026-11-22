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

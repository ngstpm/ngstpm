"""f.RegionIntegral(p): integrals over the region between Gamma_h and Gamma swept by the paths."""

import sys
from pathlib import Path

import numpy as np
import pytest
from ngsolve import CF, BilinearForm, GridFunction, Integrate, L2, NumberSpace, dx, pi, x, y

from ngstpm import ParametricCurve, boundary_ds, transfer_paths, unfitted_domain

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson_tpm import GEOMETRIES, background_mesh  # noqa: E402

AREA = {"disk": pi, "flower": pi * (1 + 0.12**2 / 2), "annulus": pi * (1 - 0.5**2)}


def setup(phi, maxh, order=5, mesh=None):
    mesh = mesh or background_mesh(maxh)
    inner = unfitted_domain(mesh, phi)[0]
    p = transfer_paths(mesh, phi, inner, order=order)
    return mesh, inner, p, boundary_ds(mesh, inner, bonus_intorder=10)


@pytest.mark.parametrize("name,tol", [("disk", 1e-10), ("flower", 1e-6), ("annulus", 1e-9)])
def test_area(name, tol):
    mesh, inner, p, dB = setup(GEOMETRIES[name], 0.1)
    region = Integrate(CF(1).RegionIntegral(p) * dB, mesh)
    assert abs(region - (AREA[name] - Integrate(CF(1) * dx(definedonelements=inner), mesh))) < tol


def test_polynomial():
    f = x * x + y * y + x * y + x                                     # int over the unit disk: pi/2
    mesh, inner, p, dB = setup(GEOMETRIES["disk"], 0.1)
    region = Integrate(f.RegionIntegral(p) * dB, mesh)
    assert abs(region - (pi / 2 - Integrate(f * dx(definedonelements=inner), mesh))) < 1e-10


def test_function_gridfunction_trial_agree():
    """A polynomial of degree k as a function of x, y, as a GridFunction and as a trial function
    (a form integrates the facets with order 2k + bonus, Integrate with its own: enough bonus)."""
    f = 1 + x - 2 * y + x * y + 3 * y * y
    mesh, inner, p, _ = setup(GEOMETRIES["flower"], 0.2)
    dB = boundary_ds(mesh, inner, bonus_intorder=20)
    W, R = L2(mesh, order=2), NumberSpace(mesh)
    fh = GridFunction(W)
    fh.Set(f, definedonelements=inner)
    a = BilinearForm(trialspace=W, testspace=R)
    a += W.TrialFunction().RegionIntegral(p) * R.TestFunction() * dB
    a.Assemble()
    trial = a.mat.CreateColVector()
    trial.data = a.mat * fh.vec
    region = [Integrate(g.RegionIntegral(p) * dB, mesh) for g in (f, fh)] + [trial[0]]
    assert max(region) - min(region) < 1e-14


def test_closest_point_paths():
    circle = ParametricCurve(lambda t: (np.cos(t), np.sin(t)))
    mesh, inner, p, dB = setup(circle, 0.2, order=6)
    region = Integrate(CF(1).RegionIntegral(p) * dB, mesh)
    assert abs(region - (pi - Integrate(CF(1) * dx(definedonelements=inner), mesh))) < 1e-8


def test_volume_3d():
    from poisson_tpm_3d import GEOMETRIES as GEOMETRIES_3D, background_mesh as background_mesh_3d
    mesh, inner, p, _ = setup(GEOMETRIES_3D["ball"], None, mesh=background_mesh_3d(0.5))
    dB = boundary_ds(mesh, inner, bonus_intorder=6)
    region = Integrate(CF(1).RegionIntegral(p) * dB, mesh)
    assert abs(region - (4 * pi / 3 - Integrate(CF(1) * dx(definedonelements=inner), mesh))) < 1e-3


def test_test_function_raises():
    mesh, inner, p, dB = setup(GEOMETRIES["disk"], 0.3)
    with pytest.raises(TypeError):
        L2(mesh, order=1).TestFunction().RegionIntegral(p)
    legacy = transfer_paths(mesh, GEOMETRIES["disk"], inner, kind="facet-normal")
    with pytest.raises(NotImplementedError):
        CF(1).RegionIntegral(legacy)

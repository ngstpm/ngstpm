import sys
from pathlib import Path

import numpy as np
import pytest
from ngsolve import CF, Norm

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

import ngstpm  # noqa: E402
from ngstpm import transfer_paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from poisson_tpm import GEOMETRIES, background_mesh, solve  # noqa: E402

g = lambda X, Y: X * X - 2 * X * Y + Y         # in P_2: reproduced exactly by k = 2 (test_tpm)


@pytest.fixture(scope="module")
def quadratic():
    mesh = background_mesh(0.3)
    phi = GEOMETRIES["flower"]
    (qh, uh, _), inner = solve(mesh, 2, CF(-2.0), g, phi)
    return mesh, qh, uh, transfer_paths(mesh, phi, inner, order=6)


def drawn(ax):
    """(points, values) of each tripcolor of ax."""
    out = []
    for c in ax.collections:
        tri = getattr(c, "_triangulation", None)
        if tri is not None:
            out.append((np.column_stack((tri.x, tri.y)), np.asarray(c.get_array())))
    return out


@pytest.mark.parametrize("gradient", [False, True])
def test_extend_reproduces_quadratic(quadratic, gradient):
    """u_h = g: the samples on D_h and on the band (transferred or extrapolated) equal g."""
    mesh, qh, uh, p = quadratic
    ax = ngstpm.plot(uh, p, extend=True, gradient=-qh if gradient else None)
    pieces = drawn(ax)
    assert len(pieces) == 2
    for P, U in pieces:
        assert np.abs(U - g(P[:, 0], P[:, 1])).max() < 1e-9


def test_gamma_is_on_phi_zero(quadratic):
    """The drawn Gamma (the path ends) lies on {phi = 0}, up to the elementwise P_6 fit of
    ell, sigma (3e-3 at h = 0.3, 1e-8 at h = 0.075); paths of length 0 would give 0.36."""
    mesh, qh, uh, p = quadratic
    ax = ngstpm.plot(uh, p, edges=False)
    X, Y = np.concatenate(ax.collections[-1].get_segments()).T
    phi = np.sqrt(X**2 + Y**2) - 1 - 0.12 * np.cos(5 * np.arctan2(Y, X))     # the flower
    assert np.abs(phi).max() < 1e-2


def test_no_extend_draws_D_only(quadratic):
    mesh, qh, uh, p = quadratic
    assert len(drawn(ngstpm.plot(uh, p))) == 1
    assert len(drawn(ngstpm.plot(uh))) == 1                     # mesh from the GridFunction


def test_arguments(quadratic):
    mesh, qh, uh, p = quadratic
    with pytest.raises(TypeError, match="scalar"):
        ngstpm.plot(qh, p)
    with pytest.raises(ValueError, match="TransferPath"):
        ngstpm.plot(uh, extend=True)
    with pytest.raises(ValueError, match="extend"):
        ngstpm.plot(uh, p, gradient=-qh)
    ngstpm.plot(Norm(qh), p)                                    # scalar CF of GridFunctions

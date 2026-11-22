import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from darcy_interpolating import check_polygon, interpolating_mesh, row  # noqa: E402

POLY = {1: lambda X, Y: 1 + X - 2 * Y, 2: lambda X, Y: X * X - X * Y + 2 * Y * Y + X}


@pytest.mark.parametrize("geometry", ["disk", "flower", "neumann"])
def test_polygon_interpolates_gamma(geometry):
    """The vertices of Gamma_h are the polygon points, on Gamma to round-off."""
    mesh, P = interpolating_mesh(geometry, 0.2)
    assert check_polygon(mesh, P, geometry) < 1e-14


@pytest.mark.parametrize("k", [1, 2])
def test_polynomial_reproduced_exactly_tpm(k):
    """p in P_k is reproduced by the transferred Dirichlet condition across the O(h^2) gap.
    Measured (maxh 0.2): |u-uh| 3.0e-14 (k = 1), 8.5e-14 (k = 2)."""
    r = row("interpolating-disk-tpm", k, 0.2, pex=POLY[k])
    assert max(r["|u-uh|"], r["|p-ph|"]) < 1e-11

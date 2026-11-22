import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from darcy_snapped import row, snapped_mesh  # noqa: E402

POLY = {1: lambda X, Y: 1 + X - 2 * Y, 2: lambda X, Y: X * X - X * Y + 2 * Y * Y + X}


@pytest.mark.parametrize("geometry", ["disk", "flower", "neumann"])
def test_snapped_mesh_checks(geometry):
    """Gamma_h vertices on Gamma, no degenerate face or element of D_h, the box untouched."""
    _, _, c = snapped_mesh(geometry, 0.2)
    assert c["max|r-R|(Gamma_h vertices)"] < 1e-12 and c["box-moved"] == 0.0
    assert c["min-face/h"] > 0.3 and c["min-area/h2"] > 0.1


@pytest.mark.parametrize("k", [1, 2])
def test_polynomial_reproduced_exactly_tpm(k):
    """p in P_k is reproduced by the transferred Dirichlet condition across the O(h^2) gap."""
    r = row("snapped-disk-tpm", k, 0.2, pex=POLY[k])
    assert max(r["|u-uh|"], r["|p-ph|"]) < 1e-11

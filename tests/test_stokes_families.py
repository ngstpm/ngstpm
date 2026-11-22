import sys
from pathlib import Path

import numpy as np
import pytest
from ngsolve import CF, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import stokes_interpolating  # noqa: E402
import stokes_snapped  # noqa: E402

POLY = {1: (CF((x + y, x - y)), x - 2 * y), 2: (CF((x * x, -2 * x * y)), x * x - y * y)}


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("family", ["interpolating", "snapped"])
def test_polynomial_reproduced_exactly_tpm(family, k):
    """u, p in P_k (div u = 0) are reproduced by the transferred Dirichlet condition and the
    RegionIntegral level across the O(h^2) gap. Measured (maxh 0.2): <= 3.2e-12."""
    module = stokes_interpolating if family == "interpolating" else stokes_snapped
    u, p = POLY[k]
    r = module.row(f"{family}-disk-tpm", k, 0.2, uex=u, pex=p)
    assert max(r["|L-Lh|"], r["|u-uh|"], r["|p-ph|"]) < 1e-10


_t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
_R = 1 + 0.12 * np.cos(5 * _t)                                                # the flower r = R(t)
_C = np.mean(_R**4 / 4) / np.mean(_R**2 / 2)                                  # mean of x^2 + y^2 over it
# k = 1: p odd in y (zero mean on Omega and Omega_h); k = 2: p = |x|^2 - C, zero mean on Omega only
FLOWER = {1: (CF((x + y, x - y)), y), 2: (CF((x * x, -2 * x * y)), x * x + y * y - _C)}


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("family", ["interpolating", "snapped"])
def test_polynomial_reproduced_flower_tpm(family, k):
    """As above on the flower, where Gamma is concave and Omega_h is not inside Omega: the
    signed RegionIntegral fixes the level. Measured (maxh 0.2, k = 2): alpha -2.2e-3 / -9.1e-3,
    |p-ph| 4.9e-13 / 1.1e-9 (interpolating / snapped; snapped: the accuracy of the fitted paths)."""
    module = stokes_interpolating if family == "interpolating" else stokes_snapped
    u, p = FLOWER[k]
    r = module.row(f"{family}-flower-tpm", k, 0.2, uex=u, pex=p)
    assert max(r["|L-Lh|"], r["|u-uh|"], r["|p-ph|"]) < 1e-7
    if k == 2:
        assert abs(r["alpha"]) > 1e-3                                         # the level is exercised


@pytest.mark.parametrize("family", ["interpolating", "snapped"])
def test_traction_runs(family):
    """The traction case on both families, both variants, k = 1, maxh 0.2: errors of the
    size of the carved run (measured: |s-sh| 0.255 / 0.264 interpolating)."""
    module = stokes_interpolating if family == "interpolating" else stokes_snapped
    for variant in ("tpm", "closest"):
        r = module.row(f"{family}-traction-{variant}", 1, 0.2)
        assert r["|s-sh|"] < 0.5 and r["|u-uh|"] < 0.2

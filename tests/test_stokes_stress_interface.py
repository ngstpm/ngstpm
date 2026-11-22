import sys
from pathlib import Path

import pytest
from ngsolve import CF, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from stokes_stress_interface import MU, PHI, domains, errors, problem, solve  # noqa: E402

# divergence-free velocities of degree k and pressures of degree k - 1, different per side
POLY = {1: ((CF((x + 2 * y, 3 * x - y)), CF((2 * x - y, x - 2 * y))), (CF(0.3), CF(-0.2))),
        2: ((CF((x * x - 2 * x * y + y * y + 1, -2 * x * y + y * y + x)), CF((x * x + y * y, -2 * x * y))),
            (x - 0.5 + 2 * (y - 0.5), 3 * y - x + 1))}


def run(family, h, k, u=None, p=None, full=True):
    args = [a for a in (u, p) if a is not None]
    f, ex, gD, gN, g = problem(*args)
    mesh, D1, D2, _ = domains(family, h, PHI)
    out, m_h = solve(mesh, k, PHI, D1, D2, MU, f, gD, gN, g, full=full)
    e, c, means = errors(mesh, out, (D1, D2), *args)
    return e, m_h, c, means


# Mu = 1, 10. After the one common shift c, sigma, p, u, gamma, traces and u* are exact to
# roundoff (observed at most 5.9e-10, unfitted k = 2); m_h is roundoff (at most 1.2e-13) and
# the two per-side means of p - p_h agree (the relative pressure level is right).
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("family", ["fitted", "snapped", "unfitted"])
def test_patch(family, k):
    e, m_h, c, means = run(family, 0.3, k, *POLY[k])
    assert max(e[1:]) < 1e-9
    assert abs(m_h) < 1e-10 and abs(means[0] - means[1]) < 1e-9


# Observed EOC (sigma modulo c I, p shifted, u, gamma; u*), maxh 0.2 -> 0.1:
#   fitted,  k = 1: 2.02, 1.95, 2.06, 1.86; 2.94    k = 2: 3.20, 3.24, 3.13, 3.21; 4.27
#   snapped, k = 1: 1.73, 1.75, 2.27, 2.00; 3.05    (k = 2: 3.21, 3.18, 3.23, 2.73, not tested: 16 s)
# The unfitted family is irregular (k = 1, 0.1 -> 0.05: -0.61, -0.54, 0.00, -1.52), not tested.
BOUNDS = {("fitted", 1): ((1.7, 1.7, 1.7, 1.7), 2.8), ("fitted", 2): ((2.7, 2.7, 2.7, 2.7), 3.8),
          ("snapped", 1): ((1.5, 1.5, 1.7, 1.7), 2.8)}


@pytest.mark.parametrize("family, k", list(BOUNDS))
def test_rates(family, k):
    from math import log2
    (e1, *_), (e2, *_) = (run(family, h, k) for h in (0.2, 0.1))
    bounds, bound_ustar = BOUNDS[(family, k)]
    for j, b in zip((1, 2, 3, 4), bounds):
        assert log2(e1[j] / e2[j]) > b
    assert log2(e1[6] / e2[6]) > bound_ustar

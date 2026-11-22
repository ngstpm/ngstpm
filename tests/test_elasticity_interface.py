import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, sqrt, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import elasticity  # noqa: E402
from elasticity_interface import PHI, U, domains, lame, problem, side_errors, solve  # noqa: E402

POLY = {1: (CF((1 + 2 * x - y, -3 + x + 4 * y)), CF((x - y, 2 + 3 * x + y))),
        2: (CF((x * x - 2 * x * y + y, 3 * y * y + x * y - x)), CF((y * y + x, x * x - x * y)))}


def run(family, h, k, u=None, materials=None):
    args = [a for a in (u, materials) if a is not None]
    lam, mu, f, ex, gD, gN, g = problem(*args)
    mesh, D1, D2, _ = domains(family, h, PHI)
    sol = solve(mesh, k, PHI, D1, D2, lam, mu, f, gD, gN, g)
    return side_errors(mesh, sol, (D1, D2), *args), (mesh, lam, mu, f, ex, g)


def together(e):
    return [sqrt(a * a + b * b) for a, b in zip(*e)]


# Displacements of degree k, different on the two sides (E = 1, nu = 0.3 and E = 10,
# nu = 0.25): all 8 error columns on both sides below 1e-9 (observed at most 4.1e-10, on the
# unfitted family, k = 2).
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("family", ["fitted", "snapped", "unfitted"])
def test_patch(family, k):
    e, _ = run(family, 0.3, k, POLY[k])
    assert max(max(e[0]), max(e[1])) < 1e-9


# Observed EOC (sigma, u, gamma), both sides together:
#   fitted,   maxh 0.2 -> 0.1:  k = 1: 2.08, 2.06, 1.95;  k = 2: 3.18, 3.13, 3.26
#   snapped,  maxh 0.2 -> 0.1:  k = 1: 1.67, 2.30, 1.85;  (k = 2: 3.24, 3.17, 2.73)
#   unfitted, maxh 0.1 -> 0.05:  k = 1: 2.40, 2.04, 3.10;  (k = 2: 1.72, 2.65, 3.04)
#   (pruned, for comparison: k = 1: 1.72, 1.93, 1.65; k = 2: 1.54, 1.50, 1.13)
# The cases in parentheses are not tested (12 s and 28 s); on the unfitted family the rates
# are irregular, see stage3-unfitted-*.txt.
BOUNDS = {("fitted", 1): (1.7, 1.7, 1.7), ("fitted", 2): (2.7, 2.7, 2.7),
          ("snapped", 1): (1.5, 1.7, 1.5), ("unfitted", 1): (1.5, 1.7, 1.5)}


@pytest.mark.parametrize("family, k", list(BOUNDS))
def test_rates(family, k):
    hs = (0.1, 0.05) if family == "unfitted" else (0.2, 0.1)
    e1, e2 = (together(run(family, h, k)[0]) for h in hs)
    for a, b, bound in zip(e1[:3], e2[:3], BOUNDS[(family, k)]):
        assert log2(a / b) > bound


def test_same_material():
    """One material and one smooth displacement on both sides (gD = gN = 0): on the
    fitted family the errors are those of a single-domain solve on the same mesh
    (observed ratios 1.000 to 1.005 at maxh 0.2 and 0.1)."""
    materials = (lame(1.0, 0.3),) * 2
    u0 = U[0]
    e, (mesh, lam, mu, f, ex, g) = run("fitted", 0.2, 1, (u0, u0), materials)
    single = elasticity.solve(mesh, 1, lam[0], mu[0], f[0], u0, dirichlet="outer")
    es = elasticity.errors(mesh, single, u0, ex[0][0], ex[0][1])
    for a, b in zip(together(e)[:3], es[:3]):
        assert a < 1.02 * b and b < 1.02 * a

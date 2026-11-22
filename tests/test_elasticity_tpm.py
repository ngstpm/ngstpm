import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, Id, InnerProduct, Integrate, Trace, cos, dx, pi, sin, sqrt, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import elasticity_tpm_neumann as neumann  # noqa: E402
from elasticity import bounded_data, errors, lame, manufactured  # noqa: E402
from elasticity_tpm import EXAMPLES, _infeasible, _teeth, at, domain, prune, solve  # noqa: E402
from ngstpm import unfitted_domain  # noqa: E402
from poisson_tpm import background_mesh  # noqa: E402

LAM, MU = lame(1.0, 0.3)
POLY = {1: CF((1 + 2 * x - y, -3 + x + 4 * y)),
        2: CF((x * x - 2 * x * y + y, 3 * y * y + x * y - x))}
SMOOTH = CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y)))


def run(name, maxh, k, u, lam=LAM, mu=MU, pruned=False):
    phi = EXAMPLES[name]
    sigma, gamma, f = manufactured(u, lam, mu)
    mesh, D, dirichlet, _ = domain(name, maxh, phi, pruned)
    sol, D = solve(mesh, k, lam, mu, f, at(u), phi, D, dirichlet=dirichlet)
    return errors(mesh, sol, u, sigma, gamma, D)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("name, maxh", [("Example 1 (fitted disk)", 0.3),
                                        ("Example 2 (unfitted disk)", 0.3),
                                        ("Example 3 (kidney)", 0.15)])
def test_patch_dirichlet(name, maxh, k):
    """A displacement of degree k is reproduced to roundoff (sigma, u, gamma, u*)."""
    e = run(name, maxh, k, POLY[k])
    assert max(e) < 1e-10


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("full", [False, True])
def test_patch_traction(k, full):
    u = POLY[k]
    sigma, gamma, f = manufactured(u, LAM, MU)
    mesh = neumann.background_mesh(0.3)
    sol, D = neumann.solve(mesh, k, LAM, MU, f, u, at(sigma), neumann.PHI, full=full)
    assert max(errors(mesh, sol, u, sigma, gamma, D)) < 1e-10


# maxh 0.1 -> 0.05, E = 1, nu = 0.3. Observed EOC (sigma, u, gamma):
#   Example 1, k = 1: 2.07, 2.07, 2.03;   k = 2: 3.08, 3.08, 3.07
#   Example 2 (default, not pruned), k = 1: 2.01, 1.92, 2.12;   k = 2: 3.12, 2.92, 2.93
#   Example 2 pruned, for comparison, k = 1: 2.21, 1.91, 2.46;   k = 2: 3.73, 2.99, 3.54
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("name, pruned", [("Example 1 (fitted disk)", True),
                                          ("Example 2 (unfitted disk)", True),
                                          ("Example 2 (unfitted disk)", False)])
def test_rates(name, pruned, k):
    e1, e2 = (run(name, h, k, SMOOTH, pruned=pruned) for h in (0.1, 0.05))
    for a, b in zip(e1[:3], e2[:3]):
        assert log2(a / b) > k + 0.7


def deviatoric_part_error(name, maxh, k, nu):
    """Errors of Example `name` with data bounded in lambda: u, gamma, u* and
    sigma - sigma_h up to its mean hydrostatic part c I, and that c."""
    lam, mu = lame(1.0, nu)
    u = bounded_data(lam)
    phi = EXAMPLES[name]
    sigma, gamma, f = manufactured(u, lam, mu)
    mesh, D, dirichlet, _ = domain(name, maxh, phi)
    sol, D = solve(mesh, k, lam, mu, f, at(u), phi, D, dirichlet=dirichlet)
    e = errors(mesh, sol, u, sigma, gamma, D)
    dV = dx(definedonelements=D, bonus_intorder=4)
    E = sol[0] - sigma
    c = Integrate(Trace(E) * dV, mesh) / Integrate(CF(1) * dV, mesh) / 2
    dev = sqrt(Integrate(InnerProduct(E - c * Id(2), E - c * Id(2)) * dV, mesh))
    return (e[1], e[2], e[5], dev), c


# On Gamma_h the transferred trace uhat is not exactly compatible, <uhat, n> != (div u, 1),
# and the mean of tr(sigma_h) is fixed by (lambda + mu) times that defect (Cardenas-Solano
# 2024, proof of Lemma 3.3, step 3, eq. (3.20)): |sigma - sigma_h| grows like lambda (5.4e4 at
# nu = 0.49999999 against 5.4 at nu = 0.4999, Example 2, k = 1, maxh 0.1), all of it
# in the constant c I. Everything else is robust in lambda: observed ratios between the two
# values of nu at most 1.001 (k = 1) and 1.099 (k = 2).
@pytest.mark.parametrize("k", [1, 2])
def test_nearly_incompressible(k):
    name = "Example 2 (unfitted disk)"
    (e1, c1), (e2, c2) = (deviatoric_part_error(name, 0.1, k, nu) for nu in (0.4999, 0.49999999))
    for a, b in zip(e1, e2):
        assert b < 1.2 * a
    assert abs(c2) > 1e3 * abs(c1)


def test_pressure_split_is_exact_with_transfer():
    """The pressure split is an exact reformulation on a transferred boundary too, so
    it does not remove the hydrostatic constant c of test_nearly_incompressible."""
    name = "Example 2 (unfitted disk)"
    lam, mu = lame(1.0, 0.4999)
    u = bounded_data(lam)
    phi = EXAMPLES[name]
    sigma, gamma, f = manufactured(u, lam, mu)
    mesh, D, dirichlet, _ = domain(name, 0.2, phi)
    dV = dx(definedonelements=D)
    a, b = (solve(mesh, 1, lam, mu, f, at(u), phi, D, dirichlet=dirichlet, split=s)[0] for s in (False, True))
    for p, q in zip(a[:3], b[:3]):
        assert sqrt(Integrate(InnerProduct(p - q, p - q) * dV, mesh)) < 1e-8


# MATLAB's pruning (elasticity_tpm.prune) on the kidney: 142 -> 131 elements at h = 0.1,
# 623 -> 602 at h = 0.05 (11 and 21 removed).
@pytest.mark.parametrize("h, removed", [(0.1, 11), (0.05, 21)])
def test_prune(h, removed):
    phi = EXAMPLES["Example 3 (kidney)"]
    mesh = background_mesh(h)
    D0 = unfitted_domain(mesh, phi)[0]
    D = prune(mesh, phi, D0)
    assert D0.NumSet() - D.NumSet() == removed
    assert not _teeth(mesh, D) and not _infeasible(mesh, phi, D, 45)
    again = prune(mesh, phi, D)                                         # idempotent
    assert again.NumSet() == D.NumSet() and (D & ~again).NumSet() == 0

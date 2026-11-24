"""The interface problem on one carved background mesh in 3D
(examples/poisson_interface_3d.py: the solve of examples/poisson_interface.py on
the unit sphere, D_h^i pruned at max_normal_deviation=85), coarsest level."""

from pathlib import Path
import sys

import numpy as np
import pytest
from ngsolve import CF, x, y, z

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import poisson_interface_3d as ex  # noqa: E402
from poisson_interface import solve  # noqa: E402

MAXH, K = 0.4, 1


@pytest.fixture(scope="module")
def domains():
    return ex.domains(MAXH)


def run(domains, u1, u2, q1, q2, condense=True):
    mesh, D1, D2 = domains
    X = (x, y, z)
    f1 = sum(q1(*X)[i].Diff(c) for i, c in enumerate(X))
    f2 = sum(q2(*X)[i].Diff(c) for i, c in enumerate(X))
    sol = solve(mesh, K, ex.phi, D1, D2, ex.kappa1, ex.kappa2, f1, f2,
                lambda *P: u1(*P) - u2(*P), lambda *P: q1(*P) - q2(*P), u2(*X), condense=condense)
    return sol


def test_affine_solution_is_reproduced(domains, monkeypatch):
    """Piecewise affine u_i (constant q_i, kappa 1 and 10) solve the discrete problem on the
    carved sides: the data are taken at the path ends on Gamma and the transfer is exact
    for affine u_i, so neither the O(h) gap nor the pruning leaves an error. Measured
    |u - uh| = 6.0e-14, |q - qh| = 2.5e-12."""
    k1, k2 = 1.0, 10.0
    u1 = lambda X, Y, Z: X - 2 * Y + 3 * Z + 1
    u2 = lambda X, Y, Z: 0.5 * X + Y - Z + 2
    q1 = lambda X, Y, Z: CF((-k1, 2 * k1, -3 * k1))
    q2 = lambda X, Y, Z: CF((-0.5 * k2, -k2, k2))
    monkeypatch.setattr(ex, "u1", u1); monkeypatch.setattr(ex, "u2", u2)
    monkeypatch.setattr(ex, "q1", q1); monkeypatch.setattr(ex, "q2", q2)
    eu, eq = ex.errors(domains[0], run(domains, u1, u2, q1, q2), domains[1], domains[2])
    print(f"affine: |u-uh|={eu:.1e} |q-qh|={eq:.1e}")
    assert eu < 1e-11 and eq < 1e-10


def test_condensed_equals_uncondensed_and_coarsest_level(domains):
    """Condensing the interface coupling gives the uncondensed solution on the carved sides
    (measured max |difference| = 1.6e-13 relative to the largest coefficient), and the coarsest
    level of the example: N = 2052 active tetrahedra, |u - uh| = 4.919e-02, |q - qh| = 3.549."""
    mesh, D1, D2 = domains
    sols = [run(domains, ex.u1, ex.u2, ex.q1, ex.q2, condense=c) for c in (True, False)]
    a, b = ([c.vec.FV().NumPy().copy() for c in s] for s in sols)
    rel = max(abs(ca - cb).max() for ca, cb in zip(a, b)) / max(abs(cb).max() for cb in b)
    print(f"condensed - uncondensed: {rel:.1e}")
    assert rel <= 1e-12
    assert sum(D1) + sum(D2) == 2052
    eu, eq = ex.errors(mesh, sols[0], D1, D2)
    assert np.isclose(eu, 4.919e-02, rtol=1e-3) and np.isclose(eq, 3.549, rtol=1e-3)


def test_pruning_that_empties_a_side_raises():
    """At 45 degrees the pruning erodes D_h^1 of the coarsest level to nothing (the band of
    two face diameters does not bound it on tetrahedra): it raises instead of returning an
    empty side."""
    with pytest.raises(RuntimeError, match=r"removed every element of D_h\^1"):
        ex.domains(MAXH, max_normal_deviation=45)

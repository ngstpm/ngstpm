import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, InnerProduct, Integrate, cos, pi, sin, sqrt, x, y
from ngsolve.meshes import MakeStructured2DMesh

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from stokes import solve, stokes_data  # noqa: E402


def errors(sol, exact, mesh, k):
    return [sqrt(Integrate(InnerProduct(a - b, a - b), mesh, order=2 * k + 6))
            for a, b in zip(sol, exact)]


def paper_case(nu=1.0):
    u = CF((sin(pi * x) * sin(pi * y), cos(pi * x) * cos(pi * y)))
    p = sin(2 * pi * x) * sin(2 * pi * y)
    L, f = stokes_data(u, p, nu)
    return u, p, L, f


def test_polynomial_reproduced_exactly():
    """A divergence-free quadratic u and a linear zero-mean p lie in the k = 2 spaces."""
    nu, tau = 0.5, 3.0
    u = CF((x * x + y, -2 * x * y + x))
    p = x - y
    L, f = stokes_data(u, p, nu)
    mesh = MakeStructured2DMesh(quads=False, nx=3, ny=3)
    Lh, uh, ph, *_ = solve(mesh, 2, f, u, nu=nu, tau=tau)
    assert max(errors((Lh, uh, ph), (L, u, p), mesh, 2)) < 1e-10


def test_condensed_equals_uncondensed():
    u, p, L, f = paper_case()
    mesh = MakeStructured2DMesh(quads=False, nx=4, ny=4)
    a = solve(mesh, 2, f, u, nu=0.7, tau=2.0)
    b = solve(mesh, 2, f, u, nu=0.7, tau=2.0, condense=False)
    for ca, cb in zip(a, b):  # L, u, p, uhat, multiplier: compare coefficient vectors
        diff = ca.vec.CreateVector()
        diff.data = ca.vec - cb.vec
        assert max(abs(t) for t in diff) < 1e-10


@pytest.mark.parametrize("k", [1, 2])
def test_convergence_rate(k):
    u, p, L, f = paper_case()
    errs = []
    for N in (8, 16):
        mesh = MakeStructured2DMesh(quads=False, nx=N, ny=N)
        Lh, uh, ph, *_ = solve(mesh, k, f, u)
        errs.append(errors((Lh, uh, ph), (L, u, p), mesh, k))
    for e0, e1 in zip(*errs):
        assert log2(e0 / e1) > k + 1 - 0.15


def test_pressure_zero_mean():
    u, p, L, f = paper_case()
    mesh = MakeStructured2DMesh(quads=False, nx=4, ny=4)
    _, _, ph, *_ = solve(mesh, 1, f, u)
    assert abs(Integrate(ph, mesh)) < 1e-13

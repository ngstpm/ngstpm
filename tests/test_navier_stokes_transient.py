import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, InnerProduct, Integrate, sqrt, x, y
from ngsolve.meshes import MakeStructured2DMesh

pytest.importorskip("scipy")                  # examples/superlu.py (the Picard solves)

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import navier_stokes_transient as nt  # noqa: E402

LINEAR = CF((x + 2 * y, 3 * x - y))           # divergence free; in P_1, so k = 1 is exact in space


def time_errors(scheme, dts, T=0.4):
    """|sigma - sigma_h|, |u - u_h|, |p - p_h| at T for u = LINEAR cos t, p = 0 (only the
    time error is left)."""
    mesh = MakeStructured2DMesh(quads=False, nx=4, ny=4)
    out = []
    for dt in dts:
        _, _, (u, p, sigma, _), sol, p_h, *_, failed = nt.solve("dirichlet", 1, None, dt, T,
                                                             scheme=scheme, mesh=mesh, U=LINEAR,
                                                             P=CF(0.0))
        assert not failed
        out.append([sqrt(Integrate(InnerProduct(a - b, a - b), mesh))
                    for a, b in ((sigma, sol[0]), (u, sol[1]), (p, p_h))])
    return out


def rates(E):
    return [[log2(a / b) for a, b in zip(E[i], E[i + 1])] for i in range(len(E) - 1)]


def test_euler_first_order_in_time():
    for r in rates(time_errors("euler", (0.1, 0.05, 0.025))):
        assert all(0.9 < ri < 1.15 for ri in r), r


def test_cn_euler_start_rates():
    """CN after one Euler step: u second order; sigma and p first order, the O(dt) pressure
    error of the Euler step carried undamped by the trapezoid (as fsi_transient.py). Also
    guards the acceleration update (an aliased NGSolve vector expression zeroed a^n, which
    made u first order too)."""
    for rs, ru, rp in rates(time_errors("cn", (0.1, 0.05, 0.025))):
        assert ru > 1.8, ru
        assert 0.9 < rs < 1.15 and 0.9 < rp < 1.15, (rs, rp)


@pytest.mark.parametrize("case", ["dirichlet", "mixed"])
def test_space_rates_cn(case):
    """k = 1, dt = h^2 (sigma, p are O(dt) after the Euler start): rates about 2 for sigma, u,
    p and 3 for u**; Picard converges at every step. mixed: do-nothing on right|top, where
    u . n = cos t > 0."""
    E = []
    for h in (0.2, 0.1):
        _, e, ex = nt.row(case, 1, h, h * h, 0.16)[1:]
        assert not ex["failed"] and ex["max_iter"] <= 6
        E.append(e)
    r = dict(zip(nt.COLUMNS["fitted"], rates(E)[0]))
    for c in ("|s-sh|", "|p-ph|", "|u-uh|"):
        assert r[c] > 1.8, (c, r[c])
    assert r["|u-u**|"] > 2.6, r["|u-u**|"]


@pytest.mark.parametrize("case, rate", [("Example 2 (unfitted disk)", 1.6), ("traction", 1.4)])
def test_unfitted_euler(case, rate):
    """TPM, k = 1, Euler, dt = h^2, T = 0.04 (1 and 4 steps): sigma, p and u as the steady loop
    (navier_stokes_stress.py, h = 0.2 -> 0.1: Example 2 2.23, 2.34, 1.94; traction 1.68, 1.54,
    2.02; here 2.42, 2.69, 1.72 and 1.61, 1.46, 1.78)."""
    E = []
    for h in (0.2, 0.1):
        _, e, ex = nt.row(case, 1, h, h * h, 0.04, scheme="euler")[1:]
        assert not ex["failed"]
        E.append(dict(zip(nt.COLUMNS[nt.ns.kind(case)], e)))
    for c in ("|s-sh|", "|u-uh|", "|p-ph-c|" if "|p-ph-c|" in E[0] else "|p-ph|"):
        assert log2(E[0][c] / E[1][c]) > rate, (c, log2(E[0][c] / E[1][c]))


def test_unfitted_cn_elliptic_start():
    """TPM, CN, small dt on a coarse mesh (h = 0.2, dt = 0.0025, 32 steps): with the L2
    projection u^0 the first step kicks a^1 by O(h^2/dt) and CN carries it in sigma
    (|s - s_h| 2.07); the Stokes projection u^0 is discretely consistent (0.369, Euler 0.347)."""
    case, args = "Example 2 (unfitted disk)", (1, 0.2, 0.0025, 0.08)
    es = {(s, st): nt.row(case, *args, scheme=s, start=st)[2][0]
          for s, st in (("cn", "elliptic"), ("cn", "l2"), ("euler", "elliptic"))}
    assert es["cn", "elliptic"] < 1.2 * es["euler", "elliptic"], es
    assert es["cn", "l2"] > 3 * es["cn", "elliptic"], es

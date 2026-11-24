import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import slip_paradox as sp  # noqa: E402


def rotating(N, k, transfer):
    mesh = sp.polygon_mesh(N)
    sol, _, m_h = sp.solve(mesh, k, sp.MU, CF((0, 0)), sp.OMEGA * sp.XPERP, transfer=transfer)
    return sp.diagnostics(mesh, sol, k), m_h


@pytest.mark.parametrize("k", [1, 2])
def test_tpm_reproduces_the_rigid_rotation(k):
    """u = Omega x^perp is linear with sigma = 0: TPM is exact up to the path fit."""
    d, m_h = rotating(16, k, True)
    assert d["err_exact"] < 1e-8
    assert abs(d["M_s"]) < 1e-8
    assert abs(d["vort_iface"] - 2 * sp.OMEGA) < 1e-8
    assert d["torque_balance"] < 1e-10
    assert abs(m_h) < 1e-10


def test_polygonal_slip_shows_the_paradox():
    """S: the torque on the polygon is far from the exact 0 and decays slowly in N;
    N = 32, k = 1 against ngsolve-tpm (experiments/slip_paradox/results/runs.csv,
    M_s = 1.290445, vort_iface = 0.916582, on its fluid+solid mesh)."""
    d16, _ = rotating(16, 1, False)
    d32, m_h = rotating(32, 1, False)
    assert d32["M_s"] == pytest.approx(1.290445, rel=2e-3)
    assert d32["vort_iface"] == pytest.approx(0.916582, rel=2e-3)
    assert 0.5 < d32["M_s"] < d16["M_s"]
    assert log2(d16["M_s"] / d32["M_s"]) < 1                       # rate below 1
    assert d32["torque_balance"] < 1e-10                             # conservative scheme
    assert abs(m_h) < 1e-12                                          # uhat . n_h = 0


def test_tpm_manufactured_slip_converges_optimally():
    """Inhomogeneous slip data at xbar: |u - u_h| = O(h^{k+1}), k = 2."""
    k = 2
    rows = sp.manufactured(k, [32, 64])
    (N0, h0, e0, _), (N1, h1, e1, m1) = rows
    eoc = [log2(a / b) / log2(h0 / h1) for a, b in zip(e0, e1)]
    assert eoc[0] > k + 0.7                                          # |s - sh|
    assert eoc[3] > k + 0.7                                          # |u - uh|
    assert eoc[6] > k + 1.7                                          # |u - u*|
    assert abs(m1) < 1e-6

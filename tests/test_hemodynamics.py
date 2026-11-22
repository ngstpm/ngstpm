import sys
from pathlib import Path

import numpy as np
import pytest
from ngsolve import BND, CF, GridFunction, Integrate, Mesh, ds, dx, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import fsi_transient as ft  # noqa: E402
import hemodynamics as hm  # noqa: E402
from component_bc import DIRICHLET, NEUMANN, _global, path_length  # noqa: E402


SMOKE_PEAKS = (0.5757143633407381, 2821.5151284409994, 0.00029589293710215766)   # observed at 7e58c9c, macOS arm64


@pytest.fixture(scope="module")
def channel():
    return hm.channel_mesh(0.2)


# ---- C1: the channel mesh ----------------------------------------------------------------------

def test_channel_mesh(channel):
    """|Omega_F| = 3, |Omega_S| = 0.6, |Sigma| = 6 seen from either side, boundary names, the
    fluid below y = 0.5 and the wall above it (by element index), zero-length transfer paths."""
    mesh, DF, DS = channel
    assert Integrate(CF(1) * dx(definedonelements=DF), mesh) == pytest.approx(3, abs=1e-12)
    assert Integrate(CF(1) * dx(definedonelements=DS), mesh) == pytest.approx(0.6, abs=1e-12)
    assert set(mesh.GetBoundaries()) == {"inflow", "outflow", "symmetry", "top", "wall_in", "wall_out", "interface"}
    assert Integrate(CF(1) * ds(definedon=mesh.Boundaries("interface")), mesh) == pytest.approx(6, abs=1e-12)
    for e in mesh.Elements():
        ys = [mesh[v].point[1] for v in e.vertices]
        assert (max(ys) <= 0.5 + 1e-12) if DF[e.nr] else (min(ys) >= 0.5 - 1e-12)
    from ngstpm import pair, transfer_paths
    pF, pS = (transfer_paths(mesh, hm.PHI, D, order=4) for D in (DF, DS))
    I = pair(pF, pS)
    for side in (1, 2):
        assert Integrate(CF(1) * I.ds(side=side), mesh) == pytest.approx(6, abs=1e-10)
    for P, D in ((pF, DF), (pS, DS)):
        assert path_length(mesh, P, D) < 1e-10


# ---- C2: the benchmark data (50), (51), the boundary table (52) and the inlet pulse ------------

def test_benchmark_constants():
    """The data (50) with mu_F = 0.035 (not the printed 1), the pulse (51), and the snapshot steps
    computed from the times 4, 8, 12 ms."""
    assert hm.MATERIAL == dict(mu_F=0.035, rho_F=1.0, lam_S=1.7e6, mu_S=0.575e6, rho_S=1.1, b0=4e6)
    assert (hm.P_MAX, hm.T_MAX) == (1.333e4, 3e-3)
    assert hm.SNAPSHOT_TIMES == (4e-3, 8e-3, 12e-3)
    assert hm.snapshot_steps(1e-4) == (40, 80, 120)
    assert hm.snapshot_steps(5e-5) == (80, 160, 240)
    assert hm.snapshot_steps(2.5e-4) == (16, 32, 48)


def test_boundary_table(channel):
    """(52) component by component: the dofs of the Dirichlet components are not free, those of
    the traction components are; the traction datum is (p_in(t), 0) on the inflow only."""
    assert hm.TABLE_F == {b: (NEUMANN, DIRICHLET) for b in ("inflow", "outflow", "symmetry")}
    assert hm.TABLE_S == {"top": (DIRICHLET, NEUMANN), "wall_in": (DIRICHLET, DIRICHLET),
                          "wall_out": (DIRICHLET, DIRICHLET)}
    run = hm.setup(0.3, order=1)
    for table, comp in ((hm.TABLE_F, 5), (hm.TABLE_S, 13)):
        for b, types in table.items():
            for c, t in enumerate(types):
                dofs = _global(run.fes, comp, b, c)
                assert dofs and all(run.free[d] == (t == NEUMANN) for d in dofs)
    mesh = run.mesh
    (_, _, gN), _ = hm.boundary_data(mesh)
    for time, p in ((1.5e-3, hm.P_MAX), (0.75e-3, hm.P_MAX / 2), (4e-3, 0.0)):
        ft.t.Set(time)
        assert Integrate(hm.p_in(), mesh) / 3.6 == pytest.approx(p, rel=1e-12, abs=1e-9)
        for b, val in (("inflow", 0.5 * p), ("outflow", 0), ("symmetry", 0)):
            for c in (0, 1):
                got = Integrate(gN[c] * ds(definedon=mesh.Boundaries(b)), mesh)
                assert got == pytest.approx(val if c == 0 else 0, abs=1e-9 * hm.P_MAX)


# ---- C3: MATLAB's profile recipes on known fields ---------------------------------------------

def test_profiles_exact():
    """Polynomial traces of degree k (projected exactly by P_M) and a quadratic pressure: the flow
    and displacement profiles return (2/3) u_x and eta_y at their sample points, the pressure
    the mean of the 3 vertex values of every bottom fluid element (for a quadratic it differs
    from the value at the centroid, so the recipe is discriminated from the centroid rule)."""
    run = hm.setup(0.3, order=2)
    mesh = run.mesh
    gf = GridFunction(run.fes)
    c = gf.components
    ft.facet_projection(c[5], CF((x * x - y + 1, x * y)), run.DF)
    ft.facet_projection(c[13], CF((y, x * x - 2 * x + 0.5)), run.DS)
    X, V = hm.flow_profile(mesh, c[5])
    assert len(X) == 3 * len(list(mesh.Boundaries("symmetry").Elements())) and np.all(np.diff(X) >= 0)
    assert np.abs(V - 2 / 3 * (X ** 2 + 1)).max() < 1e-10
    X, V = hm.displacement_profile(mesh, c[13], 2)
    assert len(X) == 3 * len(list(mesh.Boundaries("interface").Elements()))
    assert np.abs(V - (X ** 2 - 2 * X + 0.5)).max() < 1e-10
    p = lambda X, Y: 2 + 3 * X + X * X - 4 * X * Y + 5 * Y * Y
    X, P = hm.pressure_profile(mesh, p(x, y), run.DF)
    bottom = [e for e in mesh.Elements() if run.DF[e.nr] and min(mesh[v].point[1] for v in e.vertices) < 1e-10]
    assert len(X) == len(bottom) and np.all(np.diff(X) >= 0)
    V = np.array([[mesh[v].point for v in e.vertices] for e in bottom])          # (element, vertex, xy)
    xc, mean, centroid = V[:, :, 0].mean(1), p(V[:, :, 0], V[:, :, 1]).mean(1), p(*V.mean(1).T)
    i = np.argsort(xc)
    np.testing.assert_array_equal(X, xc[i])
    assert np.abs(P - mean[i]).max() < 1e-10
    assert np.abs(mean - centroid).max() > 1e-3                 # the centroid rule would fail


# ---- the march on the channel: smoke ----------------------------------------------------------

def test_benchmark_smoke(tmp_path):
    """5 steps at h = 0.3, k = 1: Euler then CN (two factorizations), finite state, profiles at
    steps 4 and 5 (the inflow pushes the flow forward), flat memory, --out round trip with the
    snapshot times; regression of the peaks at step 5."""
    run, snaps, rss, timing = hm.benchmark(0.3, 1, 1e-4, 5, snapshots=(4, 5), log=lambda *a, **k: None)
    assert run.factorizations == 2 and run.thetas == [1.0, 0.5, 0.5, 0.5, 0.5]
    assert np.all(np.isfinite(run.U.FV().NumPy()))
    P = snaps[5]
    assert P["flow"][1][0] > 0 and P["pressure"][1].max() > 0 and np.abs(P["displacement"][1]).max() > 0
    np.testing.assert_array_equal(P["pressure_prev"][0], P["pressure"][0])
    # regression: flow, pressure, d_y peaks at step 5
    assert (P["flow"][1].max(), P["pressure"][1].max(), P["displacement"][1].max()) == pytest.approx(
        SMOKE_PEAKS, rel=1e-6)
    hm.save_run(str(tmp_path / "run.npz"), run, snaps, rss, timing, dict(k=1))
    d = np.load(tmp_path / "run.npz")
    np.testing.assert_array_equal(d["flow_5"], np.vstack(P["flow"]))
    np.testing.assert_array_equal(d["dofs_4"], snaps[4]["U"])
    np.testing.assert_array_equal(d["snapshots"], [4, 5])
    np.testing.assert_allclose(d["snapshot_times"], [4e-4, 5e-4], rtol=1e-14)
    assert Mesh(str(tmp_path / "run_mesh.vol.gz")).ne == run.mesh.ne


# ---- the solid stabilisation is carried --------------------------------------------------------

def test_solid_stabilisation_carried(monkeypatch):
    """setup passes tau = (1, S_solid/mu_S) to FSITransient, and two S_solid values (1 = MATLAB,
    mu_S = decision 2) give different states after 3 steps."""
    taus = []

    class Recording(ft.FSITransient):
        def __init__(self, *a, tau=(1.0, 1.0), **kw):
            taus.append(tau)
            super().__init__(*a, tau=tau, **kw)
    monkeypatch.setattr(hm, "FSITransient", Recording)
    U = []
    for S in (1.0, hm.MATERIAL["mu_S"]):
        run = hm.setup(0.3, order=1, S_solid=S)
        run.march(3, start="euler")
        U.append(run.U.FV().NumPy().copy())
    assert taus == [(1.0, 1.0 / hm.MATERIAL["mu_S"]), (1.0, 1.0)]
    assert np.all(np.isfinite(U[0])) and np.all(np.isfinite(U[1]))
    assert np.abs(U[0] - U[1]).max() > 1e-6 * np.abs(U[0]).max()

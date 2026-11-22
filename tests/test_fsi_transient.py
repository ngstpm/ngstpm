import sys
from math import log2, pi
from pathlib import Path

import numpy as np
import pytest
from ngsolve import GridFunction, InnerProduct, Integrate, ds, dx, sin, sqrt, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import fsi_transient as ft  # noqa: E402
from component_bc import two_layer_mesh  # noqa: E402


def march(k, data, h=0.3, dt=0.1, nsteps=4, start="euler", b0=0.0, monitor=None, **kw):
    mesh, DF, DS = two_layer_mesh(h)
    return ft.run(mesh, DF, DS, k, mat=ft.material(b0=b0), dt=dt, nsteps=nsteps, start=start, monitor=monitor,
                  **data, **kw)


# ---- B1: exactness (MATLAB tests/test_fsi_transient_polynomial.m) ----------------------------

# Observed at maxh 0.3, dt 0.1, 4 steps, k = 2: worst of the 12 columns 2.3e-13 (b0 = 0 and 2).
@pytest.mark.parametrize("b0", [0.0, 2.0])
def test_exact_matlab_data(b0):
    """u_F = curl(x y (1 + t)), p = (x + y)(1 + t), eta = (x, y)(1 + t + t^2/2): Newmark exact for
    constant acceleration, backward Euler and CN for u linear in t; every step, every column.
    Euler start then CN: two factorizations, theta = 1 at step 1 only."""
    worst = []
    r = march(2, ft.matlab_polynomial(), b0=b0, monitor=lambda r: worst.append(max(r.errors(r.exact))))
    assert len(worst) == 4 and max(worst) < 1e-9
    assert r.factorizations == 2 and r.thetas == [1.0, 0.5, 0.5, 0.5]


# Observed: worst 5.6e-14 (k = 1), 3.5e-13 (k = 2), every start.
@pytest.mark.parametrize("start", ft.STARTS)
@pytest.mark.parametrize("k", [1, 2])
def test_exact_polynomial(k, start):
    """Degree-k fields (p of degree k - 1), u_F, p linear in t, eta quadratic, b0 = 1.5; the CN
    start reads the projected exact fluid acceleration; one factorization unless start = euler."""
    worst = []
    r = march(k, ft.polynomial(k), start=start, b0=1.5, monitor=lambda r: worst.append(max(r.errors(r.exact))))
    assert max(worst) < 1e-9
    assert r.factorizations == (2 if start == "euler" else 1)


def test_kinematic_row_is_velocity_match():
    """With the Newmark update the kinematic row reads uhat_F - v_hat = P_M g on Sigma (gap 0,
    conforming facets): checked on the B2 data (non-zero g) after 3 steps."""
    r = march(2, ft.from_rest(2), dt=0.05, nsteps=3, b0=2.0)
    gu, gv = GridFunction(r.fes), GridFunction(r.fes)
    gu.vec.data, gv.vec.data = r.U, r.V
    uF, eta_dot = r.exact[0], r.exact[5]
    dS = dx(element_boundary=True, definedonelements=r.DS, bonus_intorder=4)
    # P_M g on the solid facets: project g = u_F - eta' with the facet projection of fsi_transient
    Pg = GridFunction(r.fes.components[13])
    ft.facet_projection(Pg, uF - eta_dot, r.DS)
    on_sigma = r.mesh.Boundaries("interface")
    diff = gu.components[5] - gv.components[13] - Pg
    err = sqrt(Integrate(InnerProduct(diff, diff) * ds(definedon=on_sigma, bonus_intorder=4), r.mesh))
    ref = sqrt(Integrate(InnerProduct(Pg, Pg) * ds(definedon=on_sigma, bonus_intorder=4), r.mesh))
    assert ref > 1e-3 and err < 1e-10 * max(1, ref)


def test_superlu_equals_umfpack():
    """Same dof vector after 3 steps with SuperLU and UMFPACK (B2 data, k = 2)."""
    a, b = (march(2, ft.from_rest(2), dt=0.1, nsteps=3, b0=2.0, solver=s).U.FV().NumPy().copy()
            for s in ("umfpack", "superlu"))
    assert np.abs(a - b).max() < 1e-10 * max(1.0, np.abs(a).max())


# ---- B2: order in time alone (space-polynomial data from rest) -------------------------------

# Observed EOC at maxh 0.3, T = 1, dt 1/32 -> 1/64 (fluid sigma, p, u, gamma, |Pu-uh^|, u*; solid
# sigma, eta, gamma, |Pe-eh^|, eta*, eta'):
#   euler k = 1: 1.01 1.01 2.01 1.99 2.00 2.01; 1.98 1.97 1.98 1.98 1.98 1.93
#   euler k = 2: 1.01 1.01 2.01 1.94 2.00 2.01; 2.01 2.00 2.00 2.00 2.00 2.07
#   cn (control, exact projected a(0)) k = 1, 2: 2.00 in every column
#   euler-all (control) k = 1, 2: 0.97-1.00 in every column
# sigma_F and p at EOC 1 with the Euler start: the O(dt) pressure error of the backward Euler step
# (u_F'' (0) != 0) is carried by the trapezoid undamped with alternating sign
# (reports/agentB-tables/goal5-hemo/stageB-b2-diag.txt: mean(p - p_h) +6.1e-3, -5.9e-3, ...,
# -6.0e-3 at dt 1/32; half of that at dt 1/64; the CN control 1e-5 .. 5e-5).
FLUID_ALGEBRAIC = (0, 1)                 # F|s-sh|, F|p-ph|


def eocs(k, start, steps=(32, 64), T=1.0):
    e = []
    for N in steps:
        r = march(k, ft.from_rest(k), dt=T / N, nsteps=N, start=start, b0=2.0)
        e.append(r.errors(r.exact))
    return [log2(a / b) for a, b in zip(*e)]


@pytest.mark.parametrize("k", [1, 2])
def test_time_order_euler_start(k):
    r = eocs(k, "euler")
    for j, v in enumerate(r):
        if j in FLUID_ALGEBRAIC:
            assert 0.9 < v < 1.1
        else:
            assert v > 1.8


@pytest.mark.parametrize("k", [1, 2])
def test_time_order_cn_control(k):
    assert min(eocs(k, "cn")) > 1.8


def test_time_order_euler_control():
    assert all(0.85 < v < 1.15 for v in eocs(1, "euler-all"))


# ---- B3: Meddahi 2025 Example 1 ----------------------------------------------------------------

def self_eocs(k, h, uF, pF, eta, steps=(8, 16, 32)):
    """EOC of |X_dt - X_dt/2| at T = 0.3 (Meddahi Example 1 material and tables), fixed mesh."""
    m, T = ft.MEDDAHI, ft.T_MEDDAHI
    mesh, DF, DS = two_layer_mesh(h)
    states = []
    for N in steps:
        r = ft.run(mesh, DF, DS, k, uF, pF, eta, m["mat"], T / N, tables=ft.DIRICHLET_TABLES, nsteps=N)
        U, V = r.U.CreateVector(), r.V.CreateVector()
        U.data, V.data = r.U, r.V
        states.append((U, V))
    d = [ft.differences(r, *states[i], *states[i + 1]) for i in range(len(steps) - 1)]
    return [[log2(a / b) for a, b in zip(d[i], d[i + 1])] for i in range(len(d) - 1)]


# Observed k = 2, maxh 1/16, dt = 0.3/8, 0.3/16, 0.3/32: EOC 1.95-2.06 in the 10 columns of
# fsi_transient.DIFF_COLUMNS (goal5-hemo/stageB-b3-dt-k2-euler.txt: 1.95-2.14 down to 0.3/128).
def test_meddahi_self_convergence():
    m = ft.MEDDAHI
    assert min(self_eocs(2, 0.0625, m["uF"], m["pF"], m["eta"])[0]) > 1.8


# On coarse meshes the L2-projected initial accelerations (eta''(0) = 2 FIELD) do not satisfy the
# discrete equations; Newmark and the trapezoid carry that undamped and the dt self-convergence is
# irregular (k = 1, maxh 1/8: 0.35-1.90). With time factors vanishing to second order at t = 0
# the zero initial state is consistent: observed 1.99-2.01 in every column (k = 1, 2, maxh 1/8,
# both starts; goal5-hemo/stageB-b3-initial.txt).
def test_meddahi_fields_consistent_start():
    P = sin(2 * pi * x) * sin(2 * pi * y)
    r = self_eocs(1, 0.125, ft.times(ft.FIELD, sin(ft.t) ** 3), P * sin(ft.t) ** 3, ft.times(ft.FIELD, sin(ft.t) ** 4))
    assert min(r[0]) > 1.9


def test_meddahi_field_is_divergence_free():
    """The y coefficient -3/2 of (45) makes u_F divergence-free (pdftotext prints 2/3)."""
    mesh, DF, _ = two_layer_mesh(0.2)
    u = ft.MEDDAHI["uF"]
    ft.t.Set(0.7)
    div = u[0].Diff(x) + u[1].Diff(y)
    assert sqrt(Integrate(div * div * dx(definedonelements=DF, bonus_intorder=6), mesh)) < 1e-12

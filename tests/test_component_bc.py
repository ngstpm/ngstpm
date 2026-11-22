import sys
from math import log2
from pathlib import Path

import pytest
from netgen.geom2d import unit_square
from ngsolve import CF, BitArray, InnerProduct, Integrate, Mesh, ds, dx, specialcf, sqrt, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import elasticity as el  # noqa: E402
import fsi_interface as fi  # noqa: E402
import stokes_stress as st  # noqa: E402
from component_bc import (BENCHMARK, DIRICHLET, NEUMANN, NONE, PHI, TABLE_F, TABLE_S, free_components,  # noqa: E402
                          solve, solve_fsi, square_errors, square_problem, two_layer_errors, two_layer_mesh)

D2, N2 = (DIRICHLET, DIRICHLET), (NEUMANN, NEUMANN)
# degree-k fields: elasticity u; Stokes divergence-free u and p of degree k - 1 (fsi patch: also
# the solid displacement)
POLY_U = {1: CF((1 + 2 * x - y, -3 + x + 4 * y)), 2: CF((x * x - 2 * x * y + y, 3 * y * y + x * y - x))}
POLY_F = {1: (CF((x + 2 * y, 3 * x - y)), CF(0.3)),
          2: (CF((x * x - 2 * x * y + y * y + 1, -2 * x * y + y * y + x)), x - 0.5 + 2 * (y - 0.5))}


def square(h):
    return Mesh(unit_square.GenerateMesh(maxh=h))


# ---- A2: one domain, the four face types of the benchmark, non-zero data everywhere ----------

# Observed at maxh 0.3: all 8 (elasticity) / 7 (Stokes) columns at most 4e-12.
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("problem", ["elasticity", "stokes"])
def test_patch(problem, k):
    if problem == "elasticity":
        e, _ = square_errors(square(0.3), problem, k, u=POLY_U[k])
    else:
        e, _ = square_errors(square(0.3), problem, k, u=POLY_F[k][0], p=POLY_F[k][1])
    assert max(e) < 1e-9


@pytest.mark.parametrize("k", [1, 2])
def test_all_dirichlet_is_elasticity_solve(k):
    """The all-Dirichlet table gives the dof vector of elasticity.solve (dirichlet=".*")."""
    mesh = square(0.25)
    lam, mu, u, _, sigma, gamma, f = square_problem("elasticity")
    sol, _ = solve(mesh, k, lam, mu, f, {b: D2 for b in mesh.GetBoundaries()}, u)
    ref = el.solve(mesh, k, lam, mu, f, u)
    diff = sol[1].gfu.vec.FV().NumPy() - ref[1].gfu.vec.FV().NumPy()
    assert abs(diff).max() < 1e-10


@pytest.mark.parametrize("problem", ["elasticity", "stokes"])
def test_traction_sides_are_gN_path(problem):
    """Full traction on right and top, Dirichlet on left and bottom: the dof vector of the gN
    path of elasticity.solve / stokes_stress.solve (no constraint), k = 2."""
    mesh, k = square(0.25), 2
    lam, mu, u, p, sigma, gamma, f = square_problem(problem)
    gN = sigma * specialcf.normal(2)
    sol, _ = solve(mesh, k, lam, mu, f, {"left": D2, "bottom": D2, "right": N2, "top": N2}, u, gN)
    if problem == "elasticity":
        ref = el.solve(mesh, k, lam, mu, f, u, gN, "left|bottom")
    else:
        ref, _, m_h = st.solve(mesh, k, mu, f, u, gN, "left|bottom")
        assert m_h is None
    diff = sol[1].gfu.vec.FV().NumPy() - ref[1].gfu.vec.FV().NumPy()
    assert abs(diff).max() < 1e-10


@pytest.mark.parametrize("k", [1, 2])
def test_swap_changes_free_count(k):
    """Top ("dirichlet", "neumann") -> ("neumann", "neumann") frees exactly the scalar dofs
    of the top faces, (k + 1) per face."""
    mesh = square(0.25)
    fes, top = el.spaces(mesh, k, NONE)
    swapped = dict(BENCHMARK, top=N2)
    # (FreeDofs returns the space's own BitArray, cleared in place: copy it per table)
    n0, n1 = (free_components(fes, t, el.free_dofs(fes, top, True, free=BitArray(fes.FreeDofs(True)))).NumSet()
              for t in (BENCHMARK, swapped))
    faces = len(list(mesh.Boundaries("top").Elements()))
    assert n1 - n0 == (k + 1) * faces


# Observed EOC, maxh 0.1 -> 0.05 (0.2 -> 0.1 in parentheses), table BENCHMARK:
#   elasticity k = 1: sigma 2.11, u 2.07, gamma 2.05, |Pu-uh^| 3.15, u* 3.16  (2.23 2.16 2.24 3.29 3.32)
#   elasticity k = 2: 3.08 3.06 3.38 4.18 4.20                                (3.32 3.27 3.91 4.52 4.62)
#   Stokes k = 1: sigma 2.04, p 1.98, u 2.07, gamma 1.93, |Pu-uh^| 2.93, u* 2.99  (2.18 2.11 2.15 1.96 3.01 3.17)
#   Stokes k = 2: 3.09 3.19 3.08 3.00 4.05 4.07                                (3.25 3.30 3.28 2.94 3.98 4.15)
# Bounds k + 0.7 (L2 fields), k + 1.8 (|P_M u - uhat|_h, u*).
PICK = {"elasticity": ((0, 1, 2), (4, 5)), "stokes": ((0, 1, 2, 3), (4, 5))}


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("problem", ["elasticity", "stokes"])
def test_rates(problem, k):
    (e1, _), (e2, _) = (square_errors(square(h), problem, k) for h in (0.1, 0.05))
    fields, traces = PICK[problem]
    for j in fields:
        assert log2(e1[j] / e2[j]) > k + 0.7
    for j in traces:
        assert log2(e1[j] / e2[j]) > k + 1.8


# ---- A3: steady FSI on the fitted two-layer rectangle, zero-length paths ---------------------

def test_two_layer_mesh():
    mesh, DF, DS = two_layer_mesh(0.2)
    area = lambda D: Integrate(CF(1) * dx(definedonelements=D), mesh)
    length = lambda b: Integrate(CF(1) * ds(definedon=mesh.Boundaries(b)), mesh)
    assert abs(area(DF) - 1) < 1e-12 and abs(area(DS) - 0.5) < 1e-12
    assert abs(length("interface") - 1) < 1e-12 and abs(length("wall") - 1) < 1e-12
    assert set(mesh.GetBoundaries()) == set(TABLE_F) | set(TABLE_S) | {"interface"}


# Observed at maxh 0.3, alpha = 1000: all 11 columns at most 7.1e-11, mean of p - p_h 6e-14,
# and ||ell||_{L2(Gamma_h)} = 0 exactly on both sides: the first solve with zero-length paths
# (Extrapolate, Transfer, PathIntegral with dl = 0) works.
@pytest.mark.parametrize("k", [1, 2])
def test_fsi_patch(k):
    e, mean, ell, _ = two_layer_errors(0.3, k, *POLY_F[k], POLY_U[k])
    assert max(e) < 1e-9 and abs(mean) < 1e-11
    assert ell == [0.0, 0.0]


# Observed EOC at alpha = 1000, maxh 0.1 -> 0.05 (fluid sigma, p, u, gamma, |Pu-uh^|, u*; solid
# sigma, eta, gamma, |Pe-eh^|, eta*):
#   k = 1: 2.81 2.82 3.10 2.83 3.10 3.10;  2.05 1.98 2.15 3.12 3.11
#   k = 2: 3.65 3.64 5.62 3.64 5.62 5.62;  3.13 3.04 3.52 4.30 4.36
# Over 0.2 -> 0.1 the fluid u, |Pu-uh^| and u* are erratic (k = 2: 1.77; k = 1: 2.60): see
# test_fsi_translation_mode. Bounds k + 0.7 (L2 fields), k + 1.8 (traces, u*, eta*).
BOUNDS = lambda k: (k + 0.7,) * 4 + (k + 1.8,) * 2 + (k + 0.7,) * 3 + (k + 1.8,) * 2


@pytest.mark.parametrize("k", [1, 2])
def test_fsi_rates(k):
    (e1, m1, *_), (e2, m2, *_) = (two_layer_errors(h, k) for h in (0.1, 0.05))
    for a, b, bound in zip(e1, e2, BOUNDS(k)):
        assert log2(a / b) > bound
    assert abs(m2) < abs(m1) / 2 ** (k + 1)


def test_fsi_translation_mode():
    """With traction on every outer face in x, the fluid x-translation is fixed only by the
    kinematic row, u_F,x = g_x + alpha etahat_x on Sigma: its error is alpha times the mean of
    eta_x - etahat_x over Sigma (|Sigma| = 1), which dominates the raw fluid velocity error at
    alpha = 1000. Modulo its mean, the fluid velocity error is two orders smaller. Observed
    k = 2, maxh 0.2: |u - u_h| 2.968e-3, mean over D_F of (u - u_h)_x -2.968e-3, alpha times the
    mean over Sigma of (eta - etahat)_x -2.97e-3, |u - u_h - mean| 1.706e-5."""
    k, alpha = 2, 1e3
    lamS, muS, exF, exS, g, gS, uF, eta = fi.problem(alpha=alpha)
    mesh, DF, DS = two_layer_mesh(0.2)
    n = specialcf.normal(2)
    fluid, _, solid, _ = solve_fsi(mesh, k, PHI, DF, DS, fi.MU_F, lamS, muS, exF[2], exS[2], g, gS,
                                   (TABLE_F, uF, exF[0] * n), (TABLE_S, eta, exS[0] * n), alpha=alpha)
    dV = dx(definedonelements=DF, bonus_intorder=4)
    err = uF - fluid[1]
    mean = [Integrate(err[i] * dV, mesh) for i in (0, 1)]                # |D_F| = 1
    raw, rest = (sqrt(Integrate(InnerProduct(v, v) * dV, mesh)) for v in (err, err - CF(tuple(mean))))
    sigma_mean = alpha * Integrate((eta[0] - solid[3][0]) * ds(definedon=mesh.Boundaries("interface"),
                                                               bonus_intorder=4), mesh)
    assert rest < 1e-2 * raw
    assert abs(mean[0] - sigma_mean) < 1e-2 * abs(sigma_mean)

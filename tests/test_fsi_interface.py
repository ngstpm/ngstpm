import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from fsi_interface import (ETA, MU_F, NU_INC, P_F, PHI, U_F, domains, modulo_errors, problem,  # noqa: E402
                           side_errors, solve)

# fluid: divergence-free velocity of degree k, pressure of degree k - 1 (nonzero mean);
# solid: displacement of degree k
POLY = {1: (CF((x + 2 * y, 3 * x - y)), CF(0.3), CF((1 + 2 * x - y, -3 + x + 4 * y))),
        2: (CF((x * x - 2 * x * y + y * y + 1, -2 * x * y + y * y + x)), x - 0.5 + 2 * (y - 0.5),
            CF((x * x - 2 * x * y + y, 3 * y * y + x * y - x)))}


def run(family, h, k, uF=U_F, pF=P_F, eta=ETA):
    lamS, muS, exF, exS, g, gS, uF, eta = problem(uF, pF, eta)
    mesh, DS, DF, _ = domains(family, h, PHI)
    fluid, p_h, solid = solve(mesh, k, PHI, DF, DS, MU_F, lamS, muS, exF[2], exS[2], g, gS, uF)
    return side_errors(mesh, fluid, p_h, solid, DF, DS, uF, pF, exF, eta, exS)


# alpha_S = 1000, nu_S = 0.3. All 11 columns (raw sigma_F and p_F included: the level is fixed
# without a constraint) and the mean of p - p_h are roundoff: observed at most 1.7e-12
# (snapped, k = 2).
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("family", ["fitted", "snapped"])
def test_patch(family, k):
    e, mean = run(family, 0.3, k, *POLY[k])
    assert max(e) < 1e-10 and abs(mean) < 1e-11


# Observed EOC, maxh 0.2 -> 0.1 (fluid sigma raw, p raw, u, gamma, |Pu-uh^|, u*; solid sigma,
# eta, gamma, |Pe-eh^|, eta*; mean of p - p_h):
#   fitted,  k = 1: 2.88, 3.01, 2.11, 2.44, 3.50, 3.44;  2.07, 2.06, 1.94, 2.92, 3.04;  -1.2e-2 -> -1.3e-3
#   fitted,  k = 2: 3.79, 3.81, 3.81, 3.61, 4.09, 4.08;  3.15, 3.13, 3.24, 4.23, 4.26;   1.1e-3 ->  7.5e-5
#   snapped, k = 1: 2.98, 3.17, 2.28, 3.02, 4.01, 4.06;  2.32, 2.30, 2.16, 3.32, 3.33;  -1.8e-2 -> -1.7e-3
#   (snapped, k = 2: 3.72, 3.74, 4.02, 3.63, 4.16, 4.14;  3.14, 3.17, 3.47, 4.12, 4.13, not tested)
# Bounds: k + 0.7 for the L2 fields, k + 1.8 for the projected traces and u*, eta*.
COLUMNS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
BOUNDS = {(f, k): (k + 0.7,) * 4 + (k + 1.8,) * 2 + (k + 0.7,) * 3 + (k + 1.8,) * 2
          for f, k in (("fitted", 1), ("fitted", 2), ("snapped", 1))}
# Observed at maxh 0.1: sigma_F, p_F (raw), u_F, |Pu-uh^|, eta and |mean of p - p_h|; asserted
# within 1.5x. A wrong sign of a tau term in the traction row passes the rate checks but not
# these at k = 1 (tau_S: sigma_F 15x worse; tau_F: |Pu-uh^| 1.75x (fitted), 2.0x (snapped)).
AT_01 = {("fitted", 1): (6.705e-3, 3.954e-3, 1.119e-3, 1.575e-4, 6.172e-2, 1.33e-3),
         ("fitted", 2): (3.452e-4, 2.268e-4, 9.290e-6, 1.733e-5, 1.904e-3, 7.54e-5),
         ("snapped", 1): (1.088e-2, 5.756e-3, 1.248e-3, 3.662e-4, 7.366e-2, 1.71e-3)}


@pytest.mark.parametrize("family, k", list(BOUNDS))
def test_rates(family, k):
    (e1, m1), (e2, m2) = (run(family, h, k) for h in (0.2, 0.1))
    for j, b in zip(COLUMNS, BOUNDS[(family, k)]):
        assert log2(e1[j] / e2[j]) > b
    assert abs(m2) < abs(m1) / 2 ** (k + 1)          # the pressure level converges, unconstrained
    for v, ref in zip((e2[0], e2[1], e2[2], e2[4], e2[7], abs(m2)), AT_01[(family, k)]):
        assert v < 1.5 * ref


def test_nearly_incompressible_modulo():
    """nu_S = 0.49999999, bounded data, fitted, k = 1: the raw stresses carry the shift
    c = O(lam_S h^(k+3)) (decision 18), but modulo c I they converge at k + 1. Observed EOC
    maxh 0.2 -> 0.1: sigma_F, p_F, sigma_S modulo c I 2.47, 2.53, 2.03; c 4.13."""
    lamS, muS, exF, exS, g, gS, uF, eta = problem(nu=NU_INC)
    e = []
    for h in (0.2, 0.1):
        mesh, DS, DF, _ = domains("fitted", h, PHI)
        fluid, p_h, solid = solve(mesh, 1, PHI, DF, DS, MU_F, lamS, muS, exF[2], exS[2], g, gS, uF)
        _, c = side_errors(mesh, fluid, p_h, solid, DF, DS, uF, P_F, exF, eta, exS)
        e.append(modulo_errors(mesh, fluid, p_h, solid, DF, DS, P_F, exF, exS, c))
    for a, b in zip(*e):
        assert log2(a / b) > 1.7

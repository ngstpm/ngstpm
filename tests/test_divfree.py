import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, GridFunction, InnerProduct, Integrate, Mesh, VectorL2, div, dx, specialcf, sqrt, x, y
from ngsolve.meshes import MakeStructured2DMesh

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import oseen_stress  # noqa: E402
import stokes_stress as st  # noqa: E402
from divfree import checks, divergence_free  # noqa: E402

PSI = {1: x ** 3 / 3 + x * x * y - x * y * y + y ** 3, 2: x ** 4 - x ** 3 * y + x * x * y * y + x * y ** 3 + y ** 4}


def l2(mesh, v, D=None):
    return sqrt(Integrate(InnerProduct(v, v) * dx(definedonelements=D, bonus_intorder=4), mesh))


@pytest.mark.parametrize("k", [1, 2])
def test_patch(k):
    """Exact data of a solenoidal u of degree k+1 (u_h = P_k u, uhat = u, A sigma + S(gamma)
    = grad u): u** = u to roundoff (observed relative 1.2e-14)."""
    mesh = MakeStructured2DMesh(quads=False, nx=3, ny=3)
    u = CF((PSI[k].Diff(y), -PSI[k].Diff(x)))
    sigma, gamma, _ = st.manufactured(u, CF(0), 0.5)
    u_h = GridFunction(VectorL2(mesh, order=k))
    u_h.Set(u)
    us = divergence_free(mesh, k, 0.5, sigma, u_h, gamma, u)
    assert l2(mesh, us - u) < 1e-11 * l2(mesh, u)


def fitted(k, h, oseen):
    from netgen.geom2d import unit_square
    mesh = Mesh(unit_square.GenerateMesh(maxh=h))
    if oseen:
        sigma, gamma, f = oseen_stress.manufactured(st.U, st.P, 0.1, oseen_stress.BETA)
        sol = oseen_stress.solve(mesh, k, 0.1, f, st.U, oseen_stress.BETA)[0]
    else:
        sigma, gamma, f = st.manufactured(st.U, st.P, 1.0)
        sol = st.solve(mesh, k, 1.0, f, st.U)[0]
    us = divergence_free(mesh, k, 0.1 if oseen else 1.0, *sol[:4])
    return l2(mesh, us - st.U), checks(mesh, us)


# Unstructured meshes, maxh 0.1 -> 0.05, |u - u**| EOC: Stokes (mu = 1) 3.01 (k = 1), 4.10
# (k = 2); Oseen (mu = 0.1, beta = (1 + y, x)) 2.96 (k = 1), 4.24 (k = 2). div u**, the
# normal jump and |u** (HDiv) - element solution| observed below 2e-12.
@pytest.mark.parametrize("oseen", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_fitted(k, oseen):
    (e1, c1), (e2, c2) = fitted(k, 0.1, oseen), fitted(k, 0.05, oseen)
    assert log2(e1 / e2) > k + 1.8
    assert max(c1 + c2) < 1e-10


@pytest.mark.parametrize("k", [1, 2])
def test_transferred(k):
    """Unfitted disk (stokes_stress_tpm, Example 2), h = 0.2: div u** = -m_h on every element
    (observed |div u** + m_h| below 4e-13, |m_h| 7e-4 and 8e-4), u** in H(div)."""
    from elasticity_tpm import EXAMPLES, at, domain
    from stokes_stress_tpm import solve
    name = "Example 2 (unfitted disk)"
    sigma, gamma, f = st.manufactured(st.U, st.P, 1.0)
    mesh, D, dirichlet, _ = domain(name, 0.2, EXAMPLES[name])
    sol, _, m_h, D = solve(mesh, k, 1.0, f, at(st.U), EXAMPLES[name], D, dirichlet=dirichlet)
    us = divergence_free(mesh, k, 1.0, *sol[:4], D)
    dV = dx(definedonelements=D, bonus_intorder=2)
    assert abs(m_h) > 1e-4
    assert sqrt(Integrate((div(us) + m_h) ** 2 * dV, mesh)) < 1e-10
    assert max(checks(mesh, us, D)[2:]) < 1e-12

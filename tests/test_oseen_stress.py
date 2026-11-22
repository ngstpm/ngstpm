import sys
from math import log2
from pathlib import Path

import pytest
from ngsolve import CF, Mesh, specialcf, x, y
from ngsolve.meshes import MakeStructured2DMesh

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import stokes_stress  # noqa: E402
from oseen_stress import BETA, manufactured, solve, stabilization  # noqa: E402
from stokes_stress import P, U, errors  # noqa: E402

BCS = {"dirichlet": (".*", False), "mixed": ("left|bottom", True)}


def run(mesh, k, mu, u, p, beta=BETA, bc="dirichlet", stab="global"):
    dirichlet, traction = BCS[bc]
    sigma, gamma, f = manufactured(u, p, mu, beta)
    gN = sigma * specialcf.normal(2) if traction else None
    sol, p_h, m_h, S = solve(mesh, k, mu, f, u, beta, gN, dirichlet, stab)
    return errors(mesh, sol, p_h, u, p, sigma, gamma), m_h, sol


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("k", [1, 2])
def test_stokes_limit(k, bc):
    """beta = 0 (S = mu) reproduces stokes_stress.solve dof for dof (test_oseen.m:15-25);
    observed difference exactly 0."""
    mesh = MakeStructured2DMesh(quads=False, nx=4, ny=4)
    dirichlet, traction = BCS[bc]
    sigma, gamma, f = stokes_stress.manufactured(U, P, 1.0)
    gN = sigma * specialcf.normal(2) if traction else None
    a = stokes_stress.solve(mesh, k, 1.0, f, U, gN, dirichlet)[0][1].gfu.vec.FV().NumPy()
    sol, _, _, S = solve(mesh, k, 1.0, f, U, CF((0, 0)), gN, dirichlet)
    b = sol[1].gfu.vec.FV().NumPy()
    assert S == 1.0 and abs(a - b).max() < 1e-12 * abs(a).max()


@pytest.mark.parametrize("stab", ["global", "pointwise"])
@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("k", [1, 2])
def test_patch(k, bc, stab):
    """Divergence-free velocity of degree k, zero-mean pressure of degree k - 1 and
    beta = (1 + y, x): sigma, p, u, gamma to roundoff (observed at most 1.5e-13), m_h = 0;
    in the mixed case only if the convective flux is dropped on the traction faces."""
    u, p = {1: (CF((x + 2 * y, 3 * x - y)), CF(0.0)),
            2: (CF((x * x - 2 * x * y + y * y + 1, -2 * x * y + y * y + x)), x - 0.5 + 2 * (y - 0.5))}[k]
    e, m_h, _ = run(MakeStructured2DMesh(quads=False, nx=3, ny=3), k, 1.0, u, p, bc=bc, stab=stab)
    assert max(e[:4]) < 1e-10
    assert m_h is None or abs(m_h) < 1e-12


@pytest.mark.parametrize("k", [1, 2])
def test_stabilization(k):
    """Constant beta = (1, 0) on a structured mesh: |beta . n| is 0, 1 or 1/sqrt(2) on the
    edges, so S = mu + 1/2 exactly."""
    mesh = MakeStructured2DMesh(quads=False, nx=3, ny=3)
    assert abs(stabilization(mesh, CF((1, 0)), 0.1, k) - 0.6) < 1e-14


def matlab_grid():
    """MATLAB StructuredSimplicialGrid(4, 4): the same vertices and triangles."""
    import netgen.meshing as ngm
    import numpy as np
    t = np.linspace(0, 1, 5)
    co = [(a, b) for a in t for b in t]                    # vertex 5 i + j + 1 at (t_i, t_j)
    m = ngm.Mesh(dim=2)
    pn = [m.Add(ngm.MeshPoint(ngm.Pnt(a, b, 0))) for a, b in co]
    idx = m.AddRegion("d", dim=2)
    for i in range(4):
        for j in range(4):
            v = 5 * i + j
            for tri in ((v + 5, v + 1, v), (v + 1, v + 5, v + 6)):   # diagonal (t_i+1, t_j)-(t_i, t_j+1)
                m.Add(ngm.Element2D(idx, [pn[w] for w in tri[::-1]]))   # counterclockwise
    return Mesh(m)


@pytest.mark.parametrize("k", [1, 2])
def test_stabilization_matlab(k):
    """S = mu + max_bn/2 with max_bn of MATLAB convective_matrix.m for beta = (1 + y, x) on
    StructuredSimplicialGrid(4, 4) (values from Fable's MATLAB run, same Gauss rule)."""
    max_bn = {1: 1.971824583655185, 2: 1.982642038949257}[k]
    assert abs(stabilization(matlab_grid(), BETA, 0.1, k) - (0.1 + max_bn / 2)) < 1e-12


# One structured level (N = 8, k = 1, mu = 0.1, Dirichlet): errors of sigma and u pinned for
# both S, so that S reaches the solve (global and pointwise differ by about 25 % in sigma).
PINNED = {"global": (1.6534726510e-02, 7.7053083024e-03), "pointwise": (1.2966345891e-02, 8.7077579644e-03)}


@pytest.mark.parametrize("stab", PINNED)
def test_pinned(stab):
    e, _, _ = run(MakeStructured2DMesh(quads=False, nx=8, ny=8), 1, 0.1, U, P, stab=stab)
    for a, b in zip((e[0], e[2]), PINNED[stab]):
        assert abs(a / b - 1) < 1e-6


# Unstructured meshes of the unit square, beta = (1 + y, x), maxh 0.1 -> 0.05. Observed EOC
# (sigma, p, u, gamma; |P_M u - uhat|_h, u*), see agentB-tables/goal4-stage1-*.txt:
# k = 1: mu = 1, Dirichlet 2.02, 1.92, 2.07, 1.89; 2.88, 2.97, mixed 2.04, 1.98, 2.07, 1.93;
# 2.97, 3.02; mu = 0.1, Dirichlet 1.96, 2.01, 2.06, 1.88; 2.93, 2.96.
# k = 2: mu = 1, Dirichlet 3.11, 3.25, 3.13, 3.00; 4.07, 4.09; mu = 0.1, mixed 3.20, 3.28,
# 3.23, 3.18; 4.39, 4.29. m_h is roundoff.
@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("mu", [1.0, 0.1])
@pytest.mark.parametrize("k", [1, 2])
def test_rates(k, mu, bc):
    from netgen.geom2d import unit_square
    out = [run(Mesh(unit_square.GenerateMesh(maxh=h)), k, mu, U, P, bc=bc) for h in (0.1, 0.05)]
    (e1, m1, _), (e2, m2, _) = out
    for a, b in zip(e1[:4], e2[:4]):
        assert log2(a / b) > k + 0.7
    for a, b in zip(e1[4:6], e2[4:6]):
        assert log2(a / b) > k + 1.7
    assert m1 is None or (abs(m1) < 1e-12 and abs(m2) < 1e-12)

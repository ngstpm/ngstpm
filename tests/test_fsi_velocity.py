import sys
from math import log2
from pathlib import Path

import numpy as np
import pytest
from ngsolve import BitArray, GridFunction, InnerProduct, Integrate, Mesh, VOL, ds, sqrt

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import fsi_transient as ft  # noqa: E402
import fsi_velocity as fv  # noqa: E402

GAP = 0.05


def gap_mesh(maxh, gap=GAP):
    """two_layer_mesh with Gamma_h^F at y = -gap ("interface_fluid") and Gamma_h^S at y = gap
    ("interface"): two pieces of one mesh; Sigma = {y = 0} is reached by paths of length gap."""
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    p = [geo.AppendPoint(*q) for q in ((0, -1), (1, -1), (1, -gap), (0, -gap), (0, gap), (1, gap), (1, 0.5), (0, 0.5))]
    for a, b, bc, left in ((0, 1, "symmetry", 1), (1, 2, "outflow", 1), (2, 3, "interface_fluid", 1),
                           (3, 0, "inflow", 1), (4, 5, "interface", 2), (5, 6, "wall", 2), (6, 7, "top", 2),
                           (7, 4, "wall", 2)):
        geo.Append(["line", p[a], p[b]], leftdomain=left, rightdomain=0, bc=bc)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    DF = BitArray([e.index == 0 for e in mesh.Elements(VOL)])
    return mesh, DF, ~DF, (mesh.Boundaries("interface_fluid"), mesh.Boundaries("interface"))


def worst(k, data, gap, mat, **kw):
    mesh, DF, DS, facets = gap_mesh(0.3) if gap else (*ft.two_layer_mesh(0.3), (None, None))
    w = []
    fv.run(mesh, DF, DS, k, **data, mat=mat, dt=0.1, nsteps=4, facets=facets,
           monitor=lambda r: w.append(r.errors(r.exact)), **kw)
    return len(w), np.max(np.array(w), axis=0)


# Observed worst column (4 steps, h = 0.3): no gap 1e-14 (k = 1) .. 3e-13 (k = 2); gap 0.05: see
# test output. Crank-Nicolson is exact for u_F, p linear and eta quadratic in t (v, sigma_S'
# linear); with the gap the transfer rows (PathIntegral of A_S sigma_S') are exercised.
@pytest.mark.parametrize("gap", [0.0, GAP])
@pytest.mark.parametrize("data", ["polynomial", "matlab"])
@pytest.mark.parametrize("k", [1, 2])
def test_exact_polynomial(k, data, gap):
    d = ft.polynomial(k) if data == "polynomial" else ft.matlab_polynomial()
    n, W = worst(k, d, gap, ft.material(rhoF=2.0, rhoS=3.0, b0=1.5))
    assert n == 4 and W.max() < 1e-9, dict(zip(fv.COLUMNS, W))


def test_kinematic_row_is_velocity_match():
    """No gap, conforming facets: uhat_F - vhat = P_M g on Sigma at every node t^n (B2 data)."""
    mesh, DF, DS = ft.two_layer_mesh(0.3)
    r = fv.run(mesh, DF, DS, 2, **ft.from_rest(2), mat=ft.material(b0=2.0), dt=0.05, nsteps=3)
    gu = GridFunction(r.fes)
    gu.vec.data = r.U
    G = mesh.Boundaries("interface")
    d = gu.components[5] - gu.components[fv.S0 + 5]
    gP = GridFunction(r.fes.components[5])
    gP.Set(r.exact[0] - r.exact[3][1], dual=True, definedon=G)
    e = d - gP
    assert sqrt(Integrate(InnerProduct(e, e) * ds(definedon=G), mesh)) < 1e-10


# Meddahi 2025 Example 1 (fsi_transient.MEDDAHI), k = 1, dt = h, T = 0.3, two-layer mesh h 0.1 -> 0.05.
# Observed EOC: F (s, p, u, g, Pu-uh^, u-u*) 1.91, 1.78, 1.95, 1.73, 2.73, 2.81; S (s, v, w, e, v-vh^)
# 1.97, 1.91, 1.68, 1.96, 2.01 (h 0.05 -> 0.025: all >= 1.85 but S|w| 1.85, F|s| 1.98, F|p| 1.93).
def test_rates_meddahi():
    data = dict(ft.MEDDAHI)
    mat = data.pop("mat")
    e = []
    for h in (0.1, 0.05):
        mesh, DF, DS = ft.two_layer_mesh(h)
        n = round(ft.T_MEDDAHI / h)
        r = fv.run(mesh, DF, DS, 1, **data, mat=mat, dt=ft.T_MEDDAHI / n, nsteps=n, tables=ft.DIRICHLET_TABLES)
        e.append(r.errors(r.exact))
    eoc = [log2(a / b) for a, b in zip(*e)]
    assert min(eoc) > 1.6, dict(zip(fv.COLUMNS, eoc))

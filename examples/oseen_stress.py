"""Oseen flow in stress-velocity-rotation form, mixed HDG with weakly symmetric stress,
2D, fitted mesh: stokes_stress.py with a given divergence-free convective velocity beta,
kept out of the stress (MATLAB NavierStokesProblem.m:3-9; Solano-Vargas 2022, (3.4)):

    A sigma - grad u + S(gamma) = 0,   (beta . grad) u - div sigma = f   in Omega,
    u = g on Gamma_D,   sigma n = gN on Gamma_N.

The rows of stokes_stress.py with the momentum row and the flux changed to

    (sigma, grad v) - (u x beta, grad v) + <(beta . n) uhat, v> - <sigma n - S (u - uhat), v> = (f, v)
    sigma_hat n = sigma n - S (u - uhat) - (beta . n) uhat,   <sigma_hat n, mu> = 0 (interior faces),
    <sigma n - S (u - uhat), mu> = <gN, mu> on Gamma_N   (no convective flux: do-nothing),

(u x beta, grad v) = sum_ij u_i beta_j d_j v_i (convective_matrix.m:16-24,
IncompressibleElasticityMatrices.m:104-107, 152-157, skip_faces at :82). The stabilisation
is S = mu + max |beta . n|/2, one constant, the maximum over the face quadrature points
of the mesh (convective_matrix.m:89, IncompressibleElasticityMatrices.m:84-86), or,
with stab="pointwise", S = mu + |beta . n|/2 on each face (not in MATLAB). Both give
S - (beta . n)/2 >= mu; (3.5) of Solano-Vargas assumes a constant tau, so the pointwise
S lies outside its analysis.

Pressure mean as in stokes_stress.py: constraint and multiplier m only when the velocity
is prescribed on the whole boundary.

Do-nothing is meant for outflow faces (beta . n >= 0): on an inflow traction face the
energy gains +(1/2)(beta . n)|uhat|^2 < 0 and the scheme degrades for small mu (mu = 0.01,
k = 1, traction on right|top: |u - u_h| = 1.6e-1, 6.1e-2 at h = 0.1, 0.05, against 4.5e-3,
9.6e-4 with traction on left|bottom); MATLAB behaves the same.

Differences from MATLAB: the pressure mean follows the rule above (MATLAB defaults to
"constrained", NavierStokesProblem.m:31); steady only, so no rho_c (= 1) and no
convective_stabilization = false; skip marks all boundary faces where MATLAB uses
neufaces and natural_faces (equivalent, muh vanishes on the Dirichlet faces); the
pointwise S option.
"""
from math import log2

import numpy as np
from ngsolve import *

from stokes_stress import COLUMNS, HEADER, P, U, add_constraint, errors, finish, forms


def stabilization(mesh, beta, mu, order):
    """S = mu + max |beta . n|/2 over the Gauss points of the edges, with MATLAB's rule:
    max(ceil((k_beta + 2 k)/2), 1) + 1 points per edge, k_beta = k for a given field
    (convective_matrix.m:37, 42, 63, get_1d_reference_quadrature.m:15)."""
    s, _ = np.polynomial.legendre.leggauss(max((3 * order + 1) // 2, 1) + 1)
    s = (s + 1) / 2
    ends = np.array([[mesh.vertices[v.nr].point for v in e.vertices] for e in mesh.edges])
    a, t = ends[:, 0], ends[:, 1] - ends[:, 0]
    n = np.stack((t[:, 1], -t[:, 0]), axis=1) / np.linalg.norm(t, axis=1)[:, None]
    pts = a[:, None, :] + s[None, :, None] * t[:, None, :]            # edges x points x 2
    b = beta(mesh(pts[..., 0].ravel(), pts[..., 1].ravel())).reshape(pts.shape)
    return mu + np.abs(np.einsum("epi,ei->ep", b, n)).max() / 2


def convection(lhs, trial, test, beta, skip=None):
    """-(u x beta, grad v)_K + <(beta . n) uhat, v>_dK - <(beta . n) uhat, mu>_dK; the
    last is zero on interior faces when beta . n is single valued (then skip only acts on
    the boundary; it matters on interior faces when beta . n jumps), and is dropped on the
    faces where the facet function skip is 1 (do-nothing, MATLAB skip_faces).
    bonus_intorder=2 for beta of degree <= 2 (the patch and manufactured beta are linear)."""
    n = specialcf.normal(2)
    u, uhat, v, muh = trial[3], trial[5], test[3], test[5]
    bn = beta * n
    keep = 1 if skip is None else 1 - skip
    lhs += -InnerProduct(OuterProduct(u, beta), grad(v)) * dx(bonus_intorder=2)
    lhs += bn * uhat * (v - keep * muh) * dx(element_boundary=True, bonus_intorder=2)


def solve(mesh, order, mu, f, g, beta, gN=None, dirichlet=".*", stab="global"):
    """Returns (sigma_h, u_h, gamma_h, uhat_h, ustar_h), p_h, m_h (None with a traction
    boundary) and S (the constant, or None for stab="pointwise")."""
    constrained = gN is None
    S = stabilization(mesh, beta, mu, order) if stab == "global" else None
    tau = S / mu if S is not None else 1 + Norm(beta * specialcf.normal(2)) / (2 * mu)
    fes, top, lhs, rhs, trial, test, sigmahat_n = forms(mesh, order, mu, f, dirichlet, tau,
                                                        constrained=constrained)
    skip = None
    if gN is not None:      # do-nothing on the boundary faces (muh vanishes on the Dirichlet ones)
        skip = GridFunction(FacetFESpace(mesh, order=0))
        skip.Set(1, definedon=mesh.Boundaries(".*"))
    convection(lhs, trial, test, beta, skip)
    muh = test[5]
    lhs += sigmahat_n * muh * dx(element_boundary=True)
    if gN is not None:
        rhs += gN * muh * ds(skeleton=True)
    if constrained:
        add_constraint(lhs, trial, test)
    sol, p_h, m_h = finish(lhs, rhs, fes, top, (g, BND, mesh.Boundaries(dirichlet)), mesh, order,
                           mu, constrained=constrained)
    return sol, p_h, m_h, S


def manufactured(u, p, mu, beta):
    """sigma = 2 mu eps(u) - p I, gamma = (d_y u_1 - d_x u_2)/2, f = (beta . grad) u - div sigma
    (NavierStokesProblem/get_manufactured_solution.m:46, steady, rho_c = 1)."""
    import stokes_stress
    sigma, gamma, f = stokes_stress.manufactured(u, p, mu)
    gu = CF((u[0].Diff(x), u[0].Diff(y), u[1].Diff(x), u[1].Diff(y)), dims=(2, 2))
    return sigma, gamma, f + gu * beta


BETA = CF((1 + y, x))           # divergence free (test_oseen.m:28, test_transfer_oseen.m:27)


def study(meshes, order, mu=1.0, dirichlet=".*", traction=False, stab="global", u=U, p=P,
          beta=BETA, record=None):
    """Print the EOC table on a list of (h, mesh); record: a study_io.Record."""
    sigma, gamma, f = manufactured(u, p, mu, beta)
    gN = sigma * specialcf.normal(2) if traction else None
    print(f"k = {order}, mu = {mu}, dirichlet = {dirichlet!r}, stab = {stab}")
    print(HEADER + f"  {'S':>8}")
    prev, rows = None, []
    for h, mesh in meshes:
        sol, p_h, m_h, S = solve(mesh, order, mu, f, u, beta, gN, dirichlet, stab)
        e = errors(mesh, sol, p_h, u, p, sigma, gamma)
        line = f"  {h:7.4f}"
        for i, ei in enumerate(e):
            r = f"{log2(prev[1][i] / ei) / log2(prev[0] / h):4.2f}" if prev and ei > 0 else "  - "
            line += f"  {ei:9.3e}  {r}"
        print(line + f"  {'-' if m_h is None else f'{m_h:.2e}':>10}"
              + f"  {'-' if S is None else f'{S:.4f}':>8}", flush=True)
        if record is not None:
            record.add(h, mesh, e, sol[1].gfu, m_h=m_h, S=S)
        prev = (h, e)
        rows.append((e, m_h, S))
    return rows


CASES = {"dirichlet": {}, "mixed": dict(dirichlet="left|bottom", traction=True)}


if __name__ == "__main__":
    from netgen.geom2d import unit_square
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(CASES))
    p.add_argument("--mu", type=float, default=1.0, help="viscosity (default 1)")
    p.add_argument("--stab", choices=("global", "pointwise"), default="global",
                   help="S = mu + max|beta.n|/2 (default) or mu + |beta.n|/2 per face")
    a = parse(p)
    if a.k is None:
        # one process per case; measured peaks (all four levels in one process) 385 MB at
        # k = 1, 1.16 GB at k = 2 (agentB-tables/goal4-stage1-*.txt)
        run_defaults(__file__, [["--case", c, "--k", k, "--h", 0.2, 0.1, 0.05, 0.025, "--mu", mu]
                                for k in (1, 2)
                                for mu in (1.0, 0.1) for c in CASES])
    else:
        record = Record(COLUMNS)
        meshes = [(h, Mesh(unit_square.GenerateMesh(maxh=h))) for h in a.h]
        study(meshes, a.k, a.mu, stab=a.stab, record=record, **CASES[a.case])
        if a.out:                       # S (global) is saved per level with the errors
            save(a.out, record, dict(driver="oseen_stress", case=a.case, k=a.k, mu=a.mu,
                                     stab=a.stab, tau_F=1.0, beta="(1 + y, x)"))

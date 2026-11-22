"""Component-wise boundary conditions for the HDG trace, steady, fitted meshes (stage A of
reports/agentB-hemodynamics-plan.md): on each boundary, each Cartesian component c of the
trace uhat is either prescribed or loaded,

    "dirichlet":  uhat_c = (P_M g)_c                         (dof removed from the free set)
    "neumann":    <(sigma_hat n)_c, mu_c>_F = <gN_c, mu_c>_F,   sigma_hat n = sigma n - tau mu (u - uhat),

given by a table {boundary: (type_x, type_y)}. On axis-aligned faces this expresses the
conditions of Meddahi 2025, (52): inflow (sigma n).n = -p_in, tang(u) = 0 is ("neumann",
"dirichlet") on x = 0 with gN = (p_in, 0); symmetry u.n = 0, tang(sigma n) = 0 is ("neumann",
"dirichlet") on y = 0; the top (sigma n).n = 0, tang(u) = 0 is ("dirichlet", "neumann") on
y = const. MATLAB: boundary_condition.m:1-50 (the specs), componentwise_boundary_conditions.m:30-41
(rows face_dof + (c-1) N_DOF). Here component c of the trace space FacetFESpace ** 2 (or
restrict(FacetFESpace) ** 2) holds its dofs at the offset c n, n the scalar ndof, the same layout.

One mechanism: the spaces are built with the dirichlet pattern NONE (no boundary), every
Dirichlet component goes through the table (free_components clears its dofs from the free
BitArray, as elasticity.free_dofs clears the unused bubbles); the flux row is added on the
faces of the boundaries with a traction component (traction_rows, with boundary_ds), and the
traction data is tested against mu_c of the traction components only.

Differences from MATLAB (followed otherwise):
- Boundaries missing from the table get no row here (the caller adds them, e.g. the interface);
  in MATLAB every face carries the flux row and untagged faces are traction-free
  (componentwise_boundary_conditions.m:24-41 touches the tagged faces only).
- Dirichlet data: NGSolve's Set on the boundary faces (L2 projection face by face), copied
  only into the dofs of the Dirichlet components; MATLAB L2projSkeleton1d
  (componentwise_boundary_conditions.m:37). Same P_M g.
- Stabilisation tau_S = tau mu on both sides (decision 2); MATLAB's solid uses tau = 1 without
  mu (LinearElasticityMatrices.m:45, 105), its fluid mu tau (IncompressibleElasticityMatrices.m:78).
- Stokes (lam = inf): pressure split of stokes_stress.py, no zero-mean constraint and no
  multiplier; the table must prescribe a normal traction somewhere (decision 5, orchestrator
  decision 4: the constraint only when the normal velocity is prescribed on the whole boundary).
  Not checked. MATLAB reaches the same algebra with pressure_mean = "pinned"
  (hemodynamics.m:418, IncompressibleElasticityMatrices.m:164-168).

solve: elasticity or Stokes on one domain; solve_fsi: the steady FSI interface problem of
fsi_interface.py (goal-3 rows) with component conditions on the outer faces, on a fitted
mesh where the transfer paths have length zero.
"""
from math import log2

from ngsolve import *

import elasticity as el
import fsi_interface as fi
import stokes_stress as st
from elasticity import compliance, free_dofs, hdg_rows, postprocess, skew, solve_linear_system, spaces, stress
from ngstpm import boundary_ds, interior_ds, pair, transfer_paths

DIRICHLET, NEUMANN = "dirichlet", "neumann"
NONE = ""                       # a dirichlet pattern that matches no (named) boundary
INF = float("inf")


def component_dofs(V, boundary, c):
    """The dofs of component c of the vector trace space V (F ** 2) on the faces of the
    boundary (a name or regex), numbered in V: offset of component c + the scalar dofs of
    the boundary elements (MATLAB rows face_dof + (c-1) N_DOF,
    componentwise_boundary_conditions.m:33-35)."""
    F, start = V.components[c], V.Range(c).start
    dofs = {d for e in V.mesh.Boundaries(boundary).Elements() for d in F.GetDofNrs(e)}
    if any(d < 0 for d in dofs):
        raise ValueError(f"boundary {boundary!r} has faces outside the trace space")
    return sorted(start + d for d in dofs)


def _global(fes, comp, boundary, c):
    """component_dofs in the numbering of the product space fes (trace = component comp)."""
    offset = fes.Range(comp).start
    return [offset + d for d in component_dofs(fes.components[comp], boundary, c)]


def free_components(fes, table, free, comp=5):
    """free without the dofs of the Dirichlet components of the table (trace = component comp)."""
    for b, types in table.items():
        for c, t in enumerate(types):
            if t == DIRICHLET:
                for d in _global(fes, comp, b, c):
                    free.Clear(d)
    return free


def set_dirichlet(gfu, table, g, comp=5):
    """The dofs of the Dirichlet components of the table := those of P_M g (g: vector CF);
    the other dofs of gfu are left as they are (several calls with different data
    accumulate: NGSolve's Set with definedon zeroes the dofs outside the region)."""
    names = [b for b, types in table.items() if DIRICHLET in types]
    if not names:
        return
    V = gfu.space.components[comp]
    tmp = GridFunction(V)
    tmp.Set(g, BND, definedon=V.mesh.Boundaries("|".join(names)))
    offset = gfu.space.Range(comp).start
    for b in names:
        for c, t in enumerate(table[b]):
            if t == DIRICHLET:
                for d in component_dofs(V, b, c):
                    gfu.vec[offset + d] = tmp.vec[d]


def traction_rows(mesh, lhs, rhs, sigmahat_n, muh, table, gN=None, D=None):
    """The flux row <sigma_hat n, mu> on the faces of the table's boundaries with a traction
    component (seen from the element set D, boundary_ds), and <gN_c, mu_c> for the traction
    components (gN: the vector traction sigma n, None = 0). The rows of the Dirichlet
    components are not free (free_components), so they carry no equation."""
    names = [b for b, types in table.items() if NEUMANN in types]
    if not names:
        return
    lhs += sigmahat_n * muh * boundary_ds(mesh, D, facets=mesh.Boundaries("|".join(names)))
    if gN is None:
        return
    for b in names:
        masked = CF(tuple(gN[c] if t == NEUMANN else 0 for c, t in enumerate(table[b])))
        rhs += masked * muh * ds(skeleton=True, definedon=mesh.Boundaries(b))


def solve(mesh, order, lam, mu, f, table, g, gN=None, tau=1.0):
    """Elasticity (lam finite) or Stokes (lam = inf, pressure split, no constraint) on the
    whole mesh with the component conditions of table; g: the Dirichlet data, gN: the
    traction sigma n (vector CFs, used on the components of their type). Returns
    (sigma_h, u_h, gamma_h, uhat_h, ustar_h) and p_h = -tr(sigma_h)/2."""
    stokes = lam == INF
    fes, top = spaces(mesh, order, NONE, split=stokes)
    trial, test = fes.TnT()
    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    m = 8 if stokes else 6
    sigmahat_n = hdg_rows(lhs, rhs, trial[:m], test[:m], lam, mu, f, tau)
    lhs += sigmahat_n * test[5] * interior_ds(mesh)
    traction_rows(mesh, lhs, rhs, sigmahat_n, test[5], table, gN)
    gfu = GridFunction(fes)
    set_dirichlet(gfu, table, g)
    free = free_components(fes, table, free_dofs(fes, top, True))
    solve_linear_system(lhs, rhs, gfu, free)
    sP_h, w1_h, w2_h, u_h, gam_h, uhat_h = gfu.components[:6]
    u_h.gfu = gfu                                       # the whole solution, for study_io
    sigma_h = stress(sP_h, w1_h, w2_h)
    return (sigma_h, u_h, gam_h, uhat_h, postprocess(mesh, order, lam, mu, sigma_h, u_h, gam_h)), \
        -Trace(sigma_h) / 2


def solve_fsi(mesh, order, phi, DF, DS, muF, lamS, muS, fF, fS, g, gS, bcF, bcS, alpha=1e3, tau=1.0,
              full=True):
    """fsi_interface.solve with component conditions on the outer faces: bcF = (table, g, gN)
    of the fluid trace, bcS = (table, g, gN) of the solid trace (g, gN as in solve). g(X, Y):
    the kinematic data, gS(X, Y): the stress jump sigma_S - sigma_F, at xbar. Fluid with the
    pressure split and no constraint (the normal traction of bcF and the coupling fix the level).
    Same interface rows as fsi_interface.solve (fsi_interface.py:99-117, which is not reused: it
    hard-codes the boundary "outer", :88, :116, and adds the flux row on interior faces only, :96).
    On a fitted mesh with Gamma on mesh lines the paths have length zero. Steady, no inertia: if
    bcF loads the x traction on every outer fluid face, the fluid x-translation is fixed only by
    the kinematic row, so its error is alpha times the mean over Gamma of the error of etahat_x
    (tests/test_component_bc.py::test_fsi_translation_mode).
    Returns the fluid (sigma_h, u_h, gamma_h, uhat_h, ustar_h), p_h, the solid
    (sigma_h, eta_h, gamma_h, etahat_h, etastar_h) and the paths (pF, pS)."""
    pF, pS = (transfer_paths(mesh, phi, D, order=order + 4) for D in (DF, DS))
    I = pair(pF, pS)                                                  # side 1: fluid, side 2: solid
    dF, dS = (I.ds(side=i, bonus_intorder=max(6, 2 * order)) for i in (1, 2))

    spF, spS = spaces(mesh, order, NONE, DF, split=True), spaces(mesh, order, NONE, DS)
    fes = spF[0] * spS[0]
    trial, test = fes.TnT()
    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    for tr, te, lam, mu, f, D, (table, _, gN) in ((trial[:8], test[:8], INF, muF, fF, DF, bcF),
                                                  (trial[8:], test[8:], lamS, muS, fS, DS, bcS)):
        sigmahat_n = hdg_rows(lhs, rhs, tr, te, lam, mu, f, tau, D)
        lhs += sigmahat_n * te[5] * interior_ds(mesh, D)
        traction_rows(mesh, lhs, rhs, sigmahat_n, te[5], table, gN, D)
    (sPF, wF1, wF2, uF, gamF, uhatF), (sPS, wS1, wS2, eta, gamS, etahat) = trial[:6], trial[8:]

    # Gamma_h^F, traction:  sigma_F n_F + sigma_S n_S = g_nu  at xbar, n_S = nubar = -n_F
    xbar, nubar = pF.target, pF.target_normal
    tF = -sPF.Extrapolate(xbar) * nubar - tau * muF * (uF - uhatF)
    tS = sPS.Extrapolate(xbar) * nubar - tau * muS * (eta.Extrapolate() - etahat.Extrapolate())
    lhs += (tF + tS) * test[5] * dF
    rhs += gS(*xbar) * nubar * test[5] * dF

    # Gamma_h^S, kinematic:  u_F - alpha eta = g  at xbar
    xbar = pS.target
    sF = stress(sPF, wF1, wF2) if full else sPF
    sS = stress(sPS, wS1, wS2) if full else sPS
    ubF = uhatF.Transfer(pF, gradient=compliance(sF, INF, muF) + skew(gamF))
    ubS = etahat.Transfer(pS, gradient=compliance(sS, lamS, muS) + skew(gamS))
    lhs += (ubF - alpha * ubS) * test[13] * dS
    rhs += g(*xbar) * test[13] * dS

    gfu = GridFunction(fes)
    free = free_dofs(fes, spF[1], True)
    free = free_dofs(fes, spS[1], True, free=free, first=8)
    for (table, gD, _), comp in ((bcF, 5), (bcS, 13)):
        set_dirichlet(gfu, table, gD, comp)
        free = free_components(fes, table, free, comp)
    solve_linear_system(lhs, rhs, gfu, free)
    c = gfu.components
    c[3].gfu = gfu                                      # the whole solution, for study_io
    sigF, sigS = stress(*c[:3]), stress(*c[8:11])
    fluid = (sigF, c[3], c[4], c[5], postprocess(mesh, order, INF, muF, sigF, c[3], c[4], DF))
    solid = (sigS, c[11], c[12], c[13], postprocess(mesh, order, lamS, muS, sigS, c[11], c[12], DS))
    return fluid, -Trace(sigF) / 2, solid, (pF, pS)


# ---- single domain (A2): the unit square with the four face types of the benchmark ----------

BENCHMARK = {"left": (NEUMANN, DIRICHLET),      # inflow-like:   (sigma n)_x, uhat_y
             "right": (NEUMANN, DIRICHLET),     # outflow-like
             "bottom": (NEUMANN, DIRICHLET),    # symmetry-like
             "top": (DIRICHLET, NEUMANN)}       # top-like:      uhat_x, (sigma n)_y
U_EL = CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y)))


def square_problem(problem, u=None, p=None, E=1.0, nu=0.3, mu=1.0):
    """lam, mu, exact u, p (None for elasticity), sigma, gamma, f. Elasticity: E, nu, u
    (default U_EL); Stokes: mu, u, p (default stokes_stress.U, P)."""
    if problem == "elasticity":
        lam, mu = el.lame(E, nu)
        u = U_EL if u is None else u
        return (lam, mu, u, None, *el.manufactured(u, lam, mu))
    u, p = (st.U if u is None else u), (st.P if p is None else p)
    return (INF, mu, u, p, *st.manufactured(u, p, mu))


def square_errors(mesh, problem, order, table=BENCHMARK, u=None, p=None, **material):
    """Solve on mesh with table and the exact data; elasticity: the 8 columns of
    elasticity.errors; Stokes: the 7 of stokes_stress.errors."""
    lam, mu, u, p, sigma, gamma, f = square_problem(problem, u, p, **material)
    sol, p_h = solve(mesh, order, lam, mu, f, table, u, sigma * specialcf.normal(2))
    if p is None:
        return el.errors(mesh, sol, u, sigma, gamma), sol
    return st.errors(mesh, sol, p_h, u, p, sigma, gamma), sol


# ---- two-layer rectangle (A3): Meddahi Example 1 geometry, fitted, flat interface -----------

PHI = lambda X, Y: -Y                                                  # solid {phi < 0} = {y > 0}
TABLE_F = {"inflow": (NEUMANN, DIRICHLET), "outflow": (NEUMANN, DIRICHLET),
           "symmetry": (NEUMANN, DIRICHLET)}
TABLE_S = {"top": (DIRICHLET, NEUMANN), "wall": (DIRICHLET, DIRICHLET)}


def two_layer_mesh(maxh):
    """Omega_F = (0,1) x (-1,0) (domain index 0), Omega_S = (0,1) x (0,0.5) (index 1), one
    Netgen mesh conforming on Gamma = {y = 0}. Boundaries: fluid inflow (x = 0), outflow
    (x = 1), symmetry (y = -1); solid top (y = 0.5), wall (x = 0 and 1); interface.
    Returns mesh, DF, DS."""
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    p = [geo.AppendPoint(*q) for q in ((0, -1), (1, -1), (1, 0), (1, 0.5), (0, 0.5), (0, 0))]
    for a, b, bc, left in ((0, 1, "symmetry", 1), (1, 2, "outflow", 1), (2, 3, "wall", 2),
                           (3, 4, "top", 2), (4, 5, "wall", 2), (5, 0, "inflow", 1)):
        geo.Append(["line", p[a], p[b]], leftdomain=left, rightdomain=0, bc=bc)
    geo.Append(["line", p[5], p[2]], leftdomain=2, rightdomain=1, bc="interface")
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    DF = BitArray([e.index == 0 for e in mesh.Elements(VOL)])
    return mesh, DF, ~DF


def two_layer_errors(maxh, order, uF=None, pF=None, eta=None, alpha=1e3):
    """Solve the steady FSI on two_layer_mesh(maxh) with the exact data of
    fsi_interface.problem (defaults U_F, P_F, ETA, nu_S = 0.3) and the tables TABLE_F, TABLE_S;
    returns the 11 columns and the mean of p - p_h of fsi_interface.side_errors, and the
    L2(Gamma_h) norm of ell on each side."""
    uF = fi.U_F if uF is None else uF
    pF = fi.P_F if pF is None else pF
    eta = fi.ETA if eta is None else eta
    lamS, muS, exF, exS, g, gS, uF, eta = fi.problem(uF, pF, eta, alpha=alpha)
    mesh, DF, DS = two_layer_mesh(maxh)
    n = specialcf.normal(2)
    fluid, p_h, solid, paths = solve_fsi(mesh, order, PHI, DF, DS, fi.MU_F, lamS, muS, exF[2], exS[2], g, gS,
                                         (TABLE_F, uF, exF[0] * n), (TABLE_S, eta, exS[0] * n), alpha=alpha)
    e, mean = fi.side_errors(mesh, fluid, p_h, solid, DF, DS, uF, pF, exF, eta, exS)
    ell = [path_length(mesh, P, D) for P, D in zip(paths, (DF, DS))]
    return e, mean, ell, (mesh, fluid, solid)


def path_length(mesh, path, D):
    """||ell||_{L2(Gamma_h)} of the paths of D (Gamma_h seen from D)."""
    return sqrt(Integrate(path.ell * path.ell * boundary_ds(mesh, D), mesh))


# ---- driver ----------------------------------------------------------------------------------

def table_line(h, e, prev):
    line = f"  {h:7.4f}"
    for i, ei in enumerate(e):
        r = f"{log2(prev[1][i] / ei) / log2(prev[0] / h):4.2f}" if prev and ei > 0 else "  - "
        line += f"  {ei:9.3e}  {r}"
    return line


COLUMNS = {"elasticity": el.COLUMNS, "stokes": st.COLUMNS, "fsi": fi.COLUMNS}


def study(case, order, hs, alpha=1e3, record=None):
    from netgen.geom2d import unit_square
    print(f"{case}, k = {order}, component conditions "
          + (f"{BENCHMARK}" if case != "fsi" else f"F {TABLE_F}, S {TABLE_S}, alpha = {alpha:g}"))
    print(f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS[case])
          + ("  mean p-ph  |ell|" if case == "fsi" else ""))
    prev = None
    for h in hs:
        if case == "fsi":
            e, mean, ell, (mesh, fluid, _) = two_layer_errors(h, order, alpha=alpha)
            print(table_line(h, e, prev) + f"  {mean:9.2e}  {max(ell):.1e}")
            gfu, extra = fluid[1].gfu, dict(mean=mean, ell=max(ell))
        else:
            mesh = Mesh(unit_square.GenerateMesh(maxh=h))
            e, sol = square_errors(mesh, case, order)
            print(table_line(h, e, prev))
            gfu, extra = sol[1].gfu, {}
        if record is not None:
            record.add(h, mesh, e, gfu, **extra)
        prev = (h, e)


CASES = ("elasticity", "stokes", "fsi")


if __name__ == "__main__":
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(CASES))
    p.add_argument("--alpha", type=float, default=1e3, help="fsi: alpha of the kinematic row (default 1e3)")
    a = parse(p)
    if a.k is None:
        run_defaults(__file__, [["--case", c, "--k", k, "--h", 0.2, 0.1, 0.05] for c in CASES for k in (1, 2)])
    else:
        record = Record(COLUMNS[a.case])
        study(a.case, a.k, a.h, alpha=a.alpha, record=record)
        if a.out:
            save(a.out, record, dict(driver="component_bc", case=a.case, k=a.k, tau=1.0, alpha=a.alpha,
                                     table=str(BENCHMARK if a.case != "fsi" else (TABLE_F, TABLE_S))))

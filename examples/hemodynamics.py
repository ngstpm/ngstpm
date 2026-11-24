"""Linear hemodynamics FSI benchmark (Meddahi 2025, Sec. 4.2, Example 2; stage C of
reports/agentB-hemodynamics-plan.md): the march of fsi_transient.py on the fitted channel

    Omega_F = (0,6) x (0,0.5) (fluid),  Omega_S = (0,6) x (0.5,0.6) (wall),  Sigma = {y = 0.5},

f = 0, zero initial data, driven by the inlet pulse (51)
    p_in(t) = (p_max/2)(1 - cos(2 pi t/t_max)) for t <= t_max = 3e-3, 0 after, p_max = 1.333e4,
with the data (50) (rho_S = 1.1, mu_S = 0.575e6, lam_S = 1.7e6, rho_F = 1, the Winkler spring
b0 = beta_S = 4e6) and mu_F = 0.035 (the benchmark's value; (50) prints 1, a typo:
unfitted-hdg/.claude/JOURNAL.md:2174-2206). Boundary table (52), component by component
(MATLAB hemodynamics.m:425-432, boundary_condition.m):

    inflow   x = 0, fluid:  (sigma n)_x = +p_in (n = (-1, 0)),  uhat_y = 0
    outflow  x = 6, fluid:  (sigma n)_x = 0,                    uhat_y = 0
    symmetry y = 0, fluid:  (sigma n)_x = 0,                    uhat_y = 0
    top      y = 0.6, wall: etahat_x = 0,                       (sigma n)_y = 0
    wall_in, wall_out (x = 0, 6), wall: etahat = 0

The time loop (theta-scheme fluid with one backward-Euler step, Newmark(1/4, 1/2) solid, the
kinematic row on the solid faces) is fsi_transient.FSITransient unchanged; this file only adds
the mesh, the data, the table and MATLAB's three profiles at t = 4, 8, 12 ms
(hemodynamics.m:1094-1135, 656-667):

    flow:         (2/3) uhat_x on the symmetry faces at s = 0.15, 0.5, 0.85 of each face (the trace);
    pressure:     -tr(sigma_F)/2 (sigma_F with the bubble curls) averaged over the 3 vertices of
                  every fluid element with a vertex on y = 0, against the centroid x;
    displacement: etahat_y on Sigma at k + 1 Gauss points per face (MATLAB: the interface
                  quadrature of the solid partition, k + 1 points per patch; at gap 0 the
                  transfer is the identity);
each sorted in x.

Mesh (C1): one Netgen mesh with two subdomains, conforming on Sigma; the interface segment runs
(0, 0.5) -> (6, 0.5) with the wall on its left (with the sides swapped GenerateMesh did not
return, plan C1). D_F, D_S come from the element index, never from the sign of phi at vertices:
the pitfall of MATLAB's carved lane (hemodynamics.m:22-38: vertices ON Sigma classified as fluid
by inpolygon, the solid losing the row touching y = 0.5) cannot occur. phi = 0.5 - y (Omega_S =
{phi < 0}); the transfer paths have length zero.

Differences from MATLAB (followed otherwise; those of fsi_transient.py and component_bc.py apply):
- Mesh: one unstructured Netgen mesh (about 718 + 120 elements at h = 0.1) against MATLAB's two
  mesh2d meshes (720/722 + 120, blood_vessel.m:46-76): the curves are compared by interpolation
  and a metric, never dof by dof; the pressure recipe samples different element rows.
- Enrichment: 2k bubble curls against MATLAB's k + 1 (decision 3; 4 against 3 at k = 2).
- Stabilisation: S_F = tau_F mu_F with tau_F = 1 (MATLAB IncompressibleElasticityMatrices.m:78);
  S_S given directly (--S-solid): 1 = MATLAB (LinearElasticityMatrices.m:45, 105), mu_S = decision 2.
- The boundary data is reassembled per step at the Parameter t (the load of FSITransient), not
  one vector b_in times p_in(t^{n+1}) (plan); same vector.
"""
import json
import os
import time as clock
from math import pi as PI

import numpy as np
from ngsolve import *

from component_bc import DIRICHLET, NEUMANN
from elasticity import compliance, skew
from fsi_transient import FSITransient, rss_mb, t
from ngstpm import SideEval

P_MAX, T_MAX = 1.333e4, 3e-3
PHI = lambda X, Y: 0.5 - Y                                           # wall {phi < 0} = {y > 0.5}
TABLE_F = {"inflow": (NEUMANN, DIRICHLET), "outflow": (NEUMANN, DIRICHLET), "symmetry": (NEUMANN, DIRICHLET)}
TABLE_S = {"top": (DIRICHLET, NEUMANN), "wall_in": (DIRICHLET, DIRICHLET), "wall_out": (DIRICHLET, DIRICHLET)}
MU_F = 0.035
MATERIAL = dict(mu_F=MU_F, rho_F=1.0, lam_S=1.7e6, mu_S=0.575e6, rho_S=1.1, b0=4e6)
SNAPSHOT_TIMES = (4e-3, 8e-3, 12e-3)                                 # the profiles at 4, 8, 12 ms
PROFILES = ("flow", "pressure", "displacement")


def snapshot_steps(dt, times=SNAPSHOT_TIMES):
    """The step numbers of the snapshot times at the time step dt (40, 80, 120 at dt = 1e-4)."""
    return tuple(round(s / dt) for s in times)


def p_in(time=t):
    """The inlet pulse (51) as a CF in the time Parameter (MATLAB hemodynamics.m:409)."""
    return IfPos(T_MAX - time, P_MAX / 2 * (1 - cos(2 * PI * time / T_MAX)), 0)


SPLITS = ("symmetric", "fluid")


def interface_heights(gap, split="symmetric"):
    """(y of Gamma_h^F, y of Gamma_h^S) for the gap delta (MATLAB hemodynamics.m gap_split): the
    total gap is 2 delta in both splits; "symmetric" retreats both sides by delta, "fluid" keeps
    the wall conforming to Sigma (full thickness, solid paths of length 0) and retreats the
    fluid by 2 delta."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    return (0.5 - gap, 0.5 + gap) if split == "symmetric" else (0.5 - 2 * gap, 0.5)


def channel_mesh(maxh, gap=0.0, split="symmetric"):
    """Fluid (0,6) x (0,0.5) (domain index 0), wall (0,6) x (0.5,0.6) (index 1), one Netgen mesh
    conforming on Sigma = {y = 0.5}. Boundaries: inflow, outflow, symmetry (fluid), top, wall_in,
    wall_out (wall), interface. Returns mesh, DF, DS.
    gap > 0: the symmetric gap of MATLAB (hemodynamics.m gap_split = "symmetric"): the fluid
    (0,6) x (0,0.5-gap) and the wall (0,6) x (0.5+gap,0.6), two pieces of one mesh with no
    element in between; Gamma_h^F = "interface_fluid" (y = 0.5-gap), Gamma_h^S = "interface"
    (y = 0.5+gap); Sigma stays y = 0.5, reached by vertical paths of length gap.
    split = "fluid": the fluid (0,6) x (0,0.5-2 gap), the wall (0,6) x (0.5,0.6) (interface_heights)."""
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    if gap == 0:
        p = [geo.AppendPoint(*q) for q in ((0, 0), (6, 0), (6, 0.5), (6, 0.6), (0, 0.6), (0, 0.5))]
        for a, b, bc, left in ((0, 1, "symmetry", 1), (1, 2, "outflow", 1), (2, 3, "wall_out", 2),
                               (3, 4, "top", 2), (4, 5, "wall_in", 2), (5, 0, "inflow", 1)):
            geo.Append(["line", p[a], p[b]], leftdomain=left, rightdomain=0, bc=bc)
        geo.Append(["line", p[5], p[2]], leftdomain=2, rightdomain=1, bc="interface")
    else:
        yF, yS = interface_heights(gap, split)
        p = [geo.AppendPoint(*q) for q in ((0, 0), (6, 0), (6, yF), (0, yF), (0, yS), (6, yS), (6, 0.6),
                                           (0, 0.6))]
        for a, b, bc, left in ((0, 1, "symmetry", 1), (1, 2, "outflow", 1), (2, 3, "interface_fluid", 1),
                               (3, 0, "inflow", 1), (4, 5, "interface", 2), (5, 6, "wall_out", 2),
                               (6, 7, "top", 2), (7, 4, "wall_in", 2)):
            geo.Append(["line", p[a], p[b]], leftdomain=left, rightdomain=0, bc=bc)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    DF = BitArray([e.index == 0 for e in mesh.Elements(VOL)])
    return mesh, DF, ~DF


def boundary_data(mesh):
    """bcF, bcS = (table, g_D, g_N) of component_bc: zero Dirichlet data, traction (p_in(t), 0)
    on the inflow and zero elsewhere. The traction rows integrate over element boundaries (a VOL
    context, where a BoundaryCF cannot be evaluated), so the inflow is selected by its outward
    normal (-1, 0) among the traction boundaries (inflow, outflow (1, 0), symmetry (0, -1))."""
    zero = CF((0, 0))
    n = specialcf.normal(2)
    gN = CF((p_in() * IfPos(-n[0] - 0.5, 1, 0), 0))
    return (TABLE_F, zero, gN), (TABLE_S, zero, None)


def setup(maxh, order=2, dt=1e-4, mat=MATERIAL, S_solid=1.0, solver="umfpack", gap=0.0, split="symmetric"):
    """The FSITransient object of the benchmark on channel_mesh(maxh, gap, split), at rest (zero
    state). S_solid: the solid stabilisation S_S (tau_S = S_solid/mu_S); S_F = mu_F."""
    mesh, DF, DS = channel_mesh(maxh, gap, split)
    bcF, bcS = boundary_data(mesh)
    zero = CF((0, 0))
    g = lambda X, Y: CF((X - X, Y - Y))          # g = 0 (a constant 0 CF leaves the form without a test function)
    run = FSITransient(mesh, order, PHI, DF, DS, mat, zero, zero, g, None, bcF, bcS, dt,
                       tau=(1.0, S_solid / mat["mu_S"]), solver=solver,
                       facets=(None, None) if gap == 0 else (mesh.Boundaries("interface_fluid"),
                                                             mesh.Boundaries("interface")))
    run.gap, run.split = gap, split
    run.ell_S = interface_heights(gap, split)[1] - 0.5 if gap > 0 else 0.0     # solid path length
    return run


# ---- MATLAB's profile recipes (hemodynamics.m:1094-1135) ------------------------------------

def flow_profile(mesh, uhat, s=(0.15, 0.5, 0.85)):
    """(x, (2/3) uhat_x) at s of each symmetry face, sorted in x (eval_trace, :1094-1116)."""
    X, V = [], []
    for e in mesh.Boundaries("symmetry").Elements():
        a, b = (mesh[v].point for v in e.vertices)
        for si in s:
            px, py = a[0] + si * (b[0] - a[0]), a[1] + si * (b[1] - a[1])
            X.append(px)
            V.append(2 / 3 * uhat(mesh(px, py, VOL_or_BND=BND))[0])
    i = np.argsort(X)
    return np.array(X)[i], np.array(V)[i]


VERTEX_RULE = IntegrationRule([(0, 0), (1, 0), (0, 1)], [1 / 6] * 3)


def pressure_profile(mesh, p, DF):
    """(centroid x, mean of p over the 3 vertices) of the fluid elements with a vertex on y = 0,
    sorted in x (bottom_pressure, :1118-1135). The vertex values are taken inside each element
    (an integration rule at the reference vertices, divided by the same rule applied to 1)."""
    on = BitArray([DF[e.nr] and min(mesh[v].point[1] for v in e.vertices) < 1e-10 for e in mesh.Elements(VOL)])
    d = dx(definedonelements=on, intrules={TRIG: VERTEX_RULE})
    pv = np.array(Integrate(p * d, mesh, element_wise=True))
    one = np.array(Integrate(CF(1) * d, mesh, element_wise=True))
    els = [e for e in mesh.Elements(VOL) if on[e.nr]]
    xc = np.array([np.mean([mesh[v].point[0] for v in e.vertices]) for e in els])
    pm = np.array([pv[e.nr] / one[e.nr] for e in els])
    i = np.argsort(xc)
    return xc[i], pm[i]


def displacement_profile(mesh, etahat, order, ell=0.0, G=None, DS=None):
    """(x, d_y) on Sigma at order + 1 Gauss points per interface face, sorted in x.
    ell = 0: d = etahat. ell > 0: the faces of Gamma_h^S at y = 0.5 + ell, and d = etahat
    transferred down the vertical path to Sigma (MATLAB hemodynamics.m:660-667),
    d(xbar) = etahat(x) + int_x^xbar G dl, G = A sigma_S + S(gamma_S) of the solid element of x
    (polynomial extrapolation into the gap, k + 2 Gauss points on the path)."""
    g = (np.polynomial.legendre.leggauss(order + 1)[0] + 1) / 2
    tq, wq = np.polynomial.legendre.leggauss(order + 2)
    X, V = [], []
    for e in mesh.Boundaries("interface").Elements():
        a, b = (mesh[v].point for v in e.vertices)
        for si in g:
            px, py = a[0] + si * (b[0] - a[0]), a[1] + si * (b[1] - a[1])
            X.append(px)
            d = etahat(mesh(px, py, VOL_or_BND=BND))[1]
            if ell > 0:
                inside = mesh(px, py + 1e-9 * ell)                # in the solid element of x
                for tj, wj in zip(tq, wq):
                    yq = py - ell * (tj + 1) / 2
                    d += wj / 2 * (-ell) * SideEval(G[1, 1], CF((px, yq)), mesh, DS)(inside)
            V.append(d)
    i = np.argsort(X)
    return np.array(X)[i], np.array(V)[i]


def profiles(run):
    """The three profiles of the current state of run (dict name -> (x, value))."""
    fluid, p_h, solid, _ = run.fields()
    return dict(flow=flow_profile(run.mesh, fluid[3]), pressure=pressure_profile(run.mesh, p_h, run.DF),
                displacement=solid_displacement(run, solid))


def solid_displacement(run, solid):
    """displacement_profile of the solid fields (sigma, eta, gamma, etahat, ...) of run."""
    G = compliance(solid[0], run.mat["lam_S"], run.mat["mu_S"]) + skew(solid[2])
    return displacement_profile(run.mesh, solid[3], run.order, run.ell_S, G, run.DS)


# ---- the benchmark run ----------------------------------------------------------------------

def benchmark(maxh=0.1, order=2, dt=1e-4, nsteps=120, snapshots=None, mat=MATERIAL, S_solid=1.0,
              solver="umfpack", log=print, gap=0.0, split="symmetric"):
    """March nsteps from rest; at each snapshot step (default snapshot_steps(dt)) the three
    profiles, and the pressure profile of the step before (for the two-step average). Returns
    (run, snaps, rss, timing): snaps[step] = dict of profiles (+ "pressure_prev"), rss = RSS (MB)
    after every step."""
    if snapshots is None:
        snapshots = snapshot_steps(dt)
    t0 = clock.time()
    run = setup(maxh, order, dt, mat, S_solid, solver, gap, split)
    t_setup = clock.time() - t0
    nF = sum(1 for e in run.mesh.Elements(VOL) if run.DF[e.nr])
    log(f"channel h = {maxh}, gap = {gap:g} ({split}): fluid {nF} elements, wall {run.mesh.ne - nF}; ndof {run.fes.ndof}, "
        f"free {run.free.NumSet()}; setup {t_setup:.1f} s, RSS {rss_mb():.0f} MB", flush=True)
    snaps, rss, prev = {}, [], {}

    def monitor(r):
        rss.append(rss_mb())
        if r.steps + 1 in snapshots:
            fluid, p_h, _, _ = r.fields()
            prev[r.steps + 1] = pressure_profile(r.mesh, p_h, r.DF)
        if r.steps in snapshots:
            snaps[r.steps] = dict(profiles(r), pressure_prev=prev.get(r.steps), U=r.U.FV().NumPy().copy())
            P = snaps[r.steps]
            log(f"  step {r.steps:4d} (t = {1e3 * r.time:.1f} ms): peaks flow {P['flow'][1].max():.6g}, "
                f"pressure {P['pressure'][1].max():.6g}, d_y {P['displacement'][1].max():.6g}; RSS {rss[-1]:.0f} MB",
                flush=True)
        elif r.steps % 10 == 0:
            d = solid_displacement(r, r.fields()[2])[1]
            log(f"  step {r.steps:4d} (t = {1e3 * r.time:.1f} ms), max |d_y| on Sigma = {abs(d).max():.4g}, "
                f"RSS {rss[-1]:.0f} MB", flush=True)
    t1 = clock.time()
    run.march(nsteps, start="euler", monitor=monitor)
    t_march = clock.time() - t1
    log(f"march {nsteps} steps {t_march:.1f} s, factorizations {run.factorizations}, peak step RSS "
        f"{max(rss):.0f} MB", flush=True)
    return run, snaps, rss, dict(setup=t_setup, march=t_march)


def print_table(snaps, dt):
    """One row per snapshot of snaps (as returned by benchmark): for each profile its largest
    value and where (x) it is reached, and the smallest d_y."""
    print(f"  {'t (ms)':>6}" + "".join(f"  {c:>11}  {'at x':>6}" for c in ("max flow", "max p", "max d_y"))
          + f"  {'min d_y':>11}")
    for n in sorted(snaps):
        line = f"  {1e3 * n * dt:6g}"
        for name in PROFILES:
            X, V = snaps[n][name]
            i = int(np.argmax(V))
            line += f"  {V[i]:11.5g}  {X[i]:6.3f}"
        print(line + f"  {snaps[n]['displacement'][1].min():11.5g}")


def save_run(path, run, snaps, rss, timing, params):
    """PATH.npz: the profiles (double precision) per snapshot step, the snapshot steps and times,
    the parameters (JSON), the dof vector of each snapshot state, the RSS per step; the mesh as
    PATH_mesh.vol.gz."""
    from study_io import commit
    stem = path[:-4] if path.endswith(".npz") else path
    data = dict(params=json.dumps({**params, "commit": commit(), "timing": timing}), rss=np.array(rss),
                snapshots=np.array(sorted(snaps)), snapshot_times=run.dt * np.array(sorted(snaps)))
    for n, P in snaps.items():
        for name in PROFILES:
            data[f"{name}_{n}"] = np.vstack(P[name])
        if P["pressure_prev"] is not None:
            data[f"pressure_prev_{n}"] = np.vstack(P["pressure_prev"])
        data[f"dofs_{n}"] = P["U"]
    run.mesh.ngmesh.Save(stem + "_mesh.vol.gz")
    data["mesh"] = np.array(os.path.basename(stem + "_mesh.vol.gz"))
    np.savez(stem + ".npz", **data)


if __name__ == "__main__":
    from study_io import parse, parser

    p = parser(__doc__, ["benchmark"])
    p.add_argument("--dt", type=float, default=1e-4, help="time step (default 1e-4)")
    p.add_argument("--T", type=float, default=12e-3, help="final time (default 12e-3)")
    p.add_argument("--steps", type=int, help="number of steps (default T/dt; overrides --T)")
    p.add_argument("--mu-f", type=float, default=MU_F, help=f"fluid viscosity (default {MU_F})")
    p.add_argument("--S-solid", type=float, default=1.0,
                   help="solid stabilisation S_S (default 1 = MATLAB; mu_S = decision 2)")
    p.add_argument("--solver", choices=("umfpack", "superlu"), default="umfpack")
    p.add_argument("--gap", type=float, default=0.0,
                   help="gap delta (default 0); total gap 2 delta, placed by --split")
    p.add_argument("--split", choices=SPLITS, default="symmetric",
                   help="symmetric: fluid up to 0.5 - delta, wall from 0.5 + delta; fluid: fluid up to "
                        "0.5 - 2 delta, wall conforming to Sigma")
    a = parse(p)
    if a.k is None:
        a.k, a.h = 2, [0.1]
    nsteps = a.steps if a.steps is not None else round(a.T / a.dt)
    mat = dict(MATERIAL, mu_F=a.mu_f)
    print(f"hemodynamics benchmark, k = {a.k}, h = {a.h[0]}, dt = {a.dt}, {nsteps} steps, gap {a.gap:g} ({a.split}), material {mat}, "
          f"S_F = mu_F, S_S = {a.S_solid:g}, solver {a.solver}", flush=True)
    snapshots = snapshot_steps(a.dt)
    run, snaps, rss, timing = benchmark(a.h[0], a.k, a.dt, nsteps, snapshots, mat, a.S_solid, a.solver,
                                        gap=a.gap, split=a.split)
    print_table(snaps, a.dt)
    if a.out:
        save_run(a.out, run, snaps, rss, timing,
                 dict(driver="hemodynamics", k=a.k, h=a.h[0], dt=a.dt, steps=nsteps, gap=a.gap, split=a.split, material=mat,
                      S_F=a.mu_f, S_S=a.S_solid, solver=a.solver, start="euler", snapshots=list(snapshots)))

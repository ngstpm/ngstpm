"""Carved stenosis, Gamma_h at distance O(h): the velocity-displacement FSI of fsi_transient.py
(Newmark solid, as the preprints and MATLAB) against the velocity-velocity FSI of
fsi_velocity.py (Crank-Nicolson, Meddahi 2025) on the same carving (applied paper,
experiments.tex:185-202: R = 0.10, structured background grid h = 0.025, cut elements discarded).

    Sigma = {y = 0.5 - R exp(-4 (x - 3)^2)},  phi = 0.5 - R exp(-4 (x - 3)^2) - y,

fluid below Sigma (phi > 0), wall between Sigma and y = 0.6 (phi < 0). The data, materials,
pressure pulse and boundary table are those of hemodynamics.py (zero initial state).
Mesh: structured triangles of (0,6) x (0,0.6), nx = 6/h, ny = 0.6/h; boundaries symmetry (y = 0),
top (y = 0.6), inflow / wall_in (x = 0, faces of D_F / D_S), outflow / wall_out (x = 6).
D_F, D_S: the elements with phi > 0 (phi < 0) at all vertices (a vertex with phi = 0 belongs to
neither), each pruned as in MATLAB split.m (elasticity_tpm.prune).
Stabilisation: each form as written: velocity-displacement tau_S mu_S = 1 (MATLAB, hemodynamics.py),
velocity-velocity tau_S = sqrt(rho_S mu_S) on v - vhat (fsi_velocity.py).
Recorded: ||U|| after every step (the growth monitor; the two state vectors hold different
unknowns, so only their histories are compared) and, at 4, 8, 12 ms, the flow and pressure
profiles of hemodynamics.py and the vertical displacement of the outer wall y = 0.6 (eta_y at
order + 1 points per top face; Sigma is curved, the top line is common to both forms).

    python stenosis_comparison.py --form vd|vv [--h 0.025 --k 2 --R 0.1 --dt 1e-4 --T 12e-3 --grid right|combo] --out F.mat
    python stenosis_comparison.py --compare vd.mat vv.mat
"""
import time as clock

import numpy as np
from ngsolve import *

import hemodynamics as hd
from elasticity_tpm import prune
from fsi_transient import FSITransient
from fsi_velocity import FSIVelocity
from ngstpm import unfitted_domain

L, H = 6.0, 0.6


def matlab_rows(h, R):
    """Row count of MATLAB's carve_grid (hemodynamics.m): the first of round(H/h) .. + 3 that puts
    the flat wall y_Sigma(0) at least 0.15 of a row off every grid line (25 at h = 0.025)."""
    y0, n = 0.5 - R * np.exp(-36.0), round(H / h)
    return next((c for c in range(n, n + 4) if abs(y0 * c / H - round(y0 * c / H)) > 0.15), n)


def background_mesh(h, DF=None, DS=None, grid="right", R=0.0):
    """Structured triangles of (0,L) x (0,H) with the boundaries of hemodynamics.py; a face on
    x = 0 (x = L) is inflow (outflow) if its element is in DF, wall_in (wall_out) if in DS, and
    carved otherwise (all carved when DF, DS are None). Element numbers: 2 (i ny + j) and
    2 (i ny + j) + 1 for the two triangles of the cell (i, j).
    grid "right": ny = H/h rows, every diagonal "/" (the ngstpm grid of c670b3d).
    grid "combo": MATLAB's carve_grid(grid_style = "combo"): ny = matlab_rows(h, R), diagonal "\\"
    where Sigma descends (x <= 3, slope <= 0) and "/" where it climbs."""
    from netgen.csg import Pnt
    from netgen.meshing import Element1D, Element2D, FaceDescriptor, MeshPoint
    from netgen.meshing import Mesh as NGMesh
    nx = round(L / h)
    ny = matlab_rows(h, R) if grid == "combo" else round(H / h)
    m = NGMesh(dim=2)
    p = [[m.Add(MeshPoint(Pnt(L * i / nx, H * j / ny, 0))) for j in range(ny + 1)] for i in range(nx + 1)]
    m.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    m.SetMaterial(1, "background")
    def descends(i):     # MATLAB's lattice_grid test: centred difference, slope <= 0 (so "\" where it rounds to 0)
        wall = lambda x: 0.5 - R * np.exp(-((x - 3) / 0.5) ** 2)
        xc = L * (i + 0.5) / nx
        xl, xr = max(xc - 1e-4, 0.0), min(xc + 1e-4, L)
        return (wall(xr) - wall(xl)) / (xr - xl) <= 0
    back = [grid == "combo" and descends(i) for i in range(nx)]   # "\" in column i
    for i in range(nx):
        for j in range(ny):
            if back[i]:          # first triangle holds the face x = x_i, the second x = x_{i+1}
                m.Add(Element2D(1, [p[i][j], p[i + 1][j], p[i][j + 1]]))
                m.Add(Element2D(1, [p[i + 1][j], p[i + 1][j + 1], p[i][j + 1]]))
            else:                # first triangle holds x = x_{i+1}, the second x = x_i
                m.Add(Element2D(1, [p[i][j], p[i + 1][j], p[i + 1][j + 1]]))
                m.Add(Element2D(1, [p[i][j], p[i + 1][j + 1], p[i][j + 1]]))
    names = ("symmetry", "top", "inflow", "wall_in", "outflow", "wall_out", "carved")
    for k, name in enumerate(names):
        m.SetBCName(k, name)
    index = {name: k + 1 for k, name in enumerate(names)}
    for i in range(nx):
        m.Add(Element1D([p[i][0], p[i + 1][0]], index=index["symmetry"]))
        m.Add(Element1D([p[i + 1][ny], p[i][ny]], index=index["top"]))
    side = lambda el, fluid, wall: fluid if DF is not None and DF[el] else wall if DS is not None and DS[el] \
        else "carved"
    for j in range(ny):
        m.Add(Element1D([p[0][j + 1], p[0][j]], index=index[side(2 * j + (0 if back[0] else 1),
                                                                  "inflow", "wall_in")]))
        m.Add(Element1D([p[nx][j], p[nx][j + 1]],
                        index=index[side(2 * ((nx - 1) * ny + j) + (1 if back[-1] else 0), "outflow", "wall_out")]))
    return Mesh(m)


def stenosis_phi(R):
    return lambda X, Y: 0.5 - R * exp(-4 * (X - 3) ** 2) - Y


def carve(h, R, grid="right"):
    """mesh, phi, D_F, D_S of the carved stenosis."""
    mesh, phi = background_mesh(h, grid=grid, R=R), stenosis_phi(R)
    DF = prune(mesh, phi, unfitted_domain(mesh, phi, side=2)[0])
    DS = prune(mesh, phi, unfitted_domain(mesh, phi, side=1)[0])
    return background_mesh(h, DF, DS, grid, R), phi, DF, DS          # the same elements, the sides labelled


def setup(form, h, order, dt, R, mat=hd.MATERIAL, grid="right"):
    mesh, phi, DF, DS = carve(h, R, grid)
    bcF, bcS = hd.boundary_data(mesh)
    zero = CF((0, 0))
    g = lambda X, Y: CF((X - X, Y - Y))          # g = 0 (a constant 0 CF leaves the form without a test function)
    if form == "vd":
        return FSITransient(mesh, order, phi, DF, DS, mat, zero, zero, g, None, bcF, bcS, dt,
                            tau=(1.0, 1.0 / mat["mu_S"]))
    return FSIVelocity(mesh, order, phi, DF, DS, mat, zero, zero, g, None, bcF, bcS, dt)


def displacement(run):
    """eta of the current state (the eta block of U, or the displacement E of FSIVelocity)."""
    _, _, solid, *_ = run.fields()
    return solid[1] if isinstance(run, FSITransient) else solid[4]


def wall_profile(run):
    """(x, eta_y) at order + 1 Gauss points of each top face, inside its element, sorted in x."""
    mesh, eta = run.mesh, displacement(run)
    s = (np.polynomial.legendre.leggauss(run.order + 1)[0] + 1) / 2
    X, V = [], []
    for e in mesh.Boundaries("top").Elements():
        a, b = (mesh[v].point for v in e.vertices)
        for si in s:
            px = a[0] + si * (b[0] - a[0])
            X.append(px)
            V.append(eta(mesh(px, H - 1e-9))[1])
    i = np.argsort(X)
    return np.array(X)[i], np.array(V)[i]


def profiles(run):
    fluid, p_h, *_ = run.fields()
    return dict(flow=hd.flow_profile(run.mesh, fluid[3]), pressure=hd.pressure_profile(run.mesh, p_h, run.DF),
                wall=wall_profile(run))


def march(form, h, order, dt, nsteps, R, log=print, grid="right"):
    t0 = clock.time()
    run = setup(form, h, order, dt, R, grid=grid)
    nF, nS = run.DF.NumSet(), run.DS.NumSet()
    log(f"{form}: h = {h}, k = {order}, R = {R}, grid {grid}, dt = {dt}, {nsteps} steps; fluid {nF}, wall {nS}, "
        f"discarded {run.mesh.ne - nF - nS} elements; ndof {run.fes.ndof}; setup {clock.time() - t0:.1f} s",
        flush=True)
    snapshots, norms, snaps = hd.snapshot_steps(dt), [], {}

    def monitor(r):
        norms.append(np.linalg.norm(r.U.FV().NumPy()))
        if not np.isfinite(norms[-1]):
            raise FloatingPointError(f"step {r.steps}: the state is not finite")
        if r.steps in snapshots:
            snaps[r.steps] = dict(profiles(r), U=r.U.FV().NumPy().copy())
            P = snaps[r.steps]
            log(f"  step {r.steps:4d}: ||U|| {norms[-1]:.4e}, max flow {P['flow'][1].max():.5g}, "
                f"max p {P['pressure'][1].max():.5g}, max |wall d_y| {abs(P['wall'][1]).max():.5g}", flush=True)
        elif r.steps % 10 == 0:
            log(f"  step {r.steps:4d}: ||U|| {norms[-1]:.4e}", flush=True)

    t1 = clock.time()
    status = "completed"
    try:
        if isinstance(run, FSITransient):
            run.march(nsteps, start="euler", monitor=monitor)
        else:
            run.march(nsteps, monitor=monitor)
    except FloatingPointError as err:
        status = str(err)
        log(f"  stopped: {status}", flush=True)
    log(f"march {len(norms)} steps {clock.time() - t1:.1f} s ({status})", flush=True)
    return run, norms, snaps, status


def save(path, run, form, norms, snaps, status, params):
    from scipy.io import savemat
    from study_io import commit
    data = dict(form=form, status=status, norms=np.array(norms), snapshots=np.array(sorted(snaps), dtype=float),
                commit=commit(), **{k: v for k, v in params.items()})
    for n, P in snaps.items():
        for name in ("flow", "pressure", "wall"):
            data[f"{name}_{n}"] = np.vstack(P[name])
        data[f"U_{n}"] = P["U"]
    savemat(path, data)
    run.mesh.ngmesh.Save(path[:-4] + "_mesh.vol.gz")


def compare(a_path, b_path):
    """One row per snapshot: relative l2 difference ||a - b|| / ||b|| of each profile (same mesh,
    same sample points); for each run ||U|| at the end of the pulse (T_MAX), its maximum and its
    last value."""
    from scipy.io import loadmat
    a, b = loadmat(a_path), loadmat(b_path)
    print(f"{a_path} vs {b_path}")
    for r in (a, b):
        N = r["norms"].ravel()
        n0 = min(round(hd.T_MAX / float(r["dt"].ravel()[0])), len(N))
        print(f"  {str(r['form'][0]):>2}: {str(r['status'][0])}, {len(N)} steps; ||U|| at {1e3 * hd.T_MAX:g} ms "
              f"{N[n0 - 1]:.4e}, max {N.max():.4e}, last {N[-1]:.4e}")
    print(f"  {'step':>5}  {'flow':>9}  {'pressure':>9}  {'wall d_y':>9}")
    common = sorted(set(a["snapshots"].ravel().astype(int)) & set(b["snapshots"].ravel().astype(int)))
    for n in common:
        row = []
        for name in ("flow", "pressure", "wall"):
            va, vb = a[f"{name}_{n}"][1], b[f"{name}_{n}"][1]
            row.append(np.linalg.norm(va - vb) / np.linalg.norm(vb))
        print(f"  {n:5d}" + "".join(f"  {v:9.2e}" for v in row))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--form", choices=("vd", "vv"), help="vd: fsi_transient (velocity-displacement), "
                   "vv: fsi_velocity (velocity-velocity)")
    p.add_argument("--h", type=float, default=0.025)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--R", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--T", type=float, default=12e-3)
    p.add_argument("--grid", choices=("right", "combo"), default="right",
                   help="background grid: right (every diagonal /) or MATLAB's combo carve_grid")
    p.add_argument("--out", help="the .mat file")
    p.add_argument("--compare", nargs=2, metavar=("A.mat", "B.mat"))
    a = p.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        nsteps = round(a.T / a.dt)
        run, norms, snaps, status = march(a.form, a.h, a.k, a.dt, nsteps, a.R, grid=a.grid)
        if a.out:
            save(a.out, run, a.form, norms, snaps, status, dict(h=a.h, k=a.k, R=a.R, dt=a.dt, steps=nsteps, grid=a.grid))

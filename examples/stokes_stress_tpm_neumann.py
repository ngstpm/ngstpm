"""Stokes flow in stress form (stokes_stress.py) on Omega = {phi < 0} with a traction
condition on the unfitted part Gamma = {phi = 0} of its boundary, sigma nu = gN nu, and
u = g on the fitted part (the boundary of the box); here the box (-1.3, 1.3)^2 minus a
disk of radius 0.6.

As stokes_stress_tpm.py, but the equation of uhat on Gamma_h is the traction transferred
to the end xbar of the path, with the P_k part of the stress (decision of goal 1, as
elasticity_tpm_neumann.py):

    sP_h(xbar) nubar - tau mu (u_h - uhat)(x) = gN(xbar) nubar.

Pressure mean: the traction fixes the level of p, so there is no constraint and no
multiplier (rule of stokes_stress.py); p and sigma are compared raw and the mean of
p_h - p over D_h is reported. As in poisson_tpm_neumann.py, the flux transferred over
paths of length O(h) is less robust than the transferred trace (known issue D21).
"""
from math import log2

from ngsolve import *

from elasticity_tpm import at, gamma_ds, path_ratio, prune as prune_domain
from ngstpm import interior_ds, transfer_paths, unfitted_domain
from poisson_tpm import background_mesh
import stokes_stress as st
from stokes_stress_tpm import eoc_line, mean_difference


def solve(mesh, order, mu, f, g, gN, phi, tau=1.0, prune=False, convect=None, factor=None):
    """g: the velocity on the box (CF); gN(X, Y): the stress at xbar (2x2 CF);
    convect(lhs, trial, test, D): added terms (oseen_stress_tpm.py); factor: as in
    stokes_stress.finish.
    Returns (sigma_h, u_h, gamma_h, uhat_h, ustar_h), p_h and D."""
    D = unfitted_domain(mesh, phi)[0]
    if prune:
        D = prune_domain(mesh, phi, D)
    p = transfer_paths(mesh, phi, D, order=order + 4)                 # x -> p.target on Gamma
    xbar, nubar = p.target, p.target_normal
    dB = gamma_ds(mesh, D, bonus_intorder=max(6, 2 * order))         # Gamma_h

    fes, top, lhs, rhs, trial, test, sigmahat_n = st.forms(mesh, order, mu, f, ".*", tau, D,
                                                           constrained=False)
    if convect is not None:
        convect(lhs, trial, test, D)
    sP, u, uhat = trial[0], trial[3], trial[5]
    muh = test[5]
    lhs += sigmahat_n * muh * interior_ds(mesh, D)
    lhs += (sP.Extrapolate(xbar) * nubar - tau * mu * (u - uhat)) * muh * dB
    rhs += gN(*xbar) * nubar * muh * dB
    sol, p_h, _ = st.finish(lhs, rhs, fes, top, (g, BND, mesh.Boundaries(".*")), mesh, order, mu,
                            D, constrained=False, factor=factor)
    return sol, p_h, D


PHI = lambda X, Y: 0.6 - sqrt(X * X + Y * Y)                         # (-1.3, 1.3)^2 minus a disk
COLUMNS = ("|s-sh|", "|p-ph|", "|u-uh|", "|g-gh|", "|Pu-uh^|h", "|u-u*|")
HEADER = (f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS)
          + f"  {'mean(ph-p)':>10}      R   #D_h")


def study(order, hs, mu=1.0, prune=False, record=None):
    """EOC table; record: a study_io.Record that collects the levels."""
    sigma, gamma, f = st.manufactured(st.U, st.P, mu)
    print(f"traction on the hole, k = {order}, mu = {mu}" + (", pruned" if prune else ", not pruned"))
    print(HEADER)
    prev, rows = None, []
    for h in hs:
        mesh = background_mesh(h)
        sol, p_h, D = solve(mesh, order, mu, f, st.U, at(sigma), PHI, prune=prune)
        e = st.errors(mesh, sol, p_h, st.U, st.P, sigma, gamma, D)[:6]
        c = -mean_difference(mesh, st.P, p_h, D)
        R = path_ratio(mesh, PHI, D)
        print(eoc_line(h, e, prev) + f"  {c:10.2e}  {R:5.3f}  {D.NumSet():5d}")
        if record is not None:
            record.add(h, mesh, e, sol[1].gfu, mean_ph_minus_p=c, R=R, nD=D.NumSet())
        prev = (h, e)
        rows.append((e, c, R, D.NumSet()))
    return rows


if __name__ == "__main__":
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, ["traction"])
    p.add_argument("--mu", type=float, default=1.0, help="viscosity (default 1)")
    p.add_argument("--prune", action="store_true", help="prune D_h (MATLAB rules)")
    a = parse(p)
    if a.k is None:
        # one process per case, within 2 GB each (k = 1 at h = 0.025: about 2.2 GB)
        run_defaults(__file__, [["--k", k, "--h", 0.2, 0.1, 0.05] for k in (1, 2)])
    else:
        record = Record(COLUMNS)
        study(a.k, a.h, mu=a.mu, prune=a.prune, record=record)
        if a.out:
            save(a.out, record, dict(driver="stokes_stress_tpm_neumann", case=a.case, k=a.k, mu=a.mu,
                                     tau=1.0, prune=a.prune))

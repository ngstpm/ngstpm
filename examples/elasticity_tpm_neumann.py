"""Linear elasticity on Omega = {phi < 0} with a traction condition on the unfitted
part Gamma = {phi = 0} of its boundary, sigma nu = gN nu (nu the outward normal), and
u = g on the fitted part (the boundary of the box).

As elasticity_tpm.py, but the equation of uhat on Gamma_h is the traction
transferred to the end xbar of the path,

    (E sigma_h)(xbar) nubar - tau_S (u_h - uhat)(x) = gN(xbar) nubar,

with sigma_h the P_k part of the stress (the bubbles have no normal trace on Gamma_h;
full=True also extrapolates their curl, which is less accurate: for k = 1 the errors
then stagnate on h = 0.2, 0.1, 0.05). Displacements of degree k are reproduced to
roundoff; as in poisson_tpm_neumann.py, the flux transferred over paths of length
O(h) is less robust than the transferred trace, and the rates are irregular.
"""

from ngsolve import *

from elasticity import HEADER, eoc_line, errors, finish, forms, lame, manufactured, spaces, stress
from elasticity_tpm import at, gamma_ds, path_ratio, prune as prune_domain
from ngstpm import interior_ds, transfer_paths, unfitted_domain
from poisson_tpm import background_mesh


def solve(mesh, order, lam, mu, f, g, gN, phi, tau=1.0, full=False, prune=False):
    """g: the displacement on the box (CF); gN(X, Y): the stress at xbar (2x2 CF).
    With prune, D_h is pruned as in elasticity_tpm.prune.
    Returns (sigma_h, u_h, gamma_h, uhat_h, ustar_h) and D."""
    D = unfitted_domain(mesh, phi)[0]
    if prune:
        D = prune_domain(mesh, phi, D)
    p = transfer_paths(mesh, phi, D, order=order + 4)                 # x -> p.target on Gamma
    xbar, nubar = p.target, p.target_normal
    dB = gamma_ds(mesh, D, bonus_intorder=max(6, 2 * order))         # Gamma_h

    fes, top = spaces(mesh, order, ".*", D)
    lhs, rhs, (sP, w1, w2, u, gam, uhat), muh, sigmahat_n = forms(fes, lam, mu, f, tau, D)
    lhs += sigmahat_n * muh * interior_ds(mesh, D)
    sigma_bar = stress(sP, w1, w2, at=xbar) if full else sP.Extrapolate(xbar)
    lhs += (sigma_bar * nubar - tau * mu * (u - uhat)) * muh * dB
    rhs += gN(*xbar) * nubar * muh * dB
    return finish(lhs, rhs, fes, top, (g, BND, mesh.Boundaries(".*")), mesh, order, lam, mu, D), D


PHI = lambda X, Y: 0.6 - sqrt(X * X + Y * Y)                         # (-1.3, 1.3)^2 minus a disk


def study(order, hs, nu=0.3, full=False, prune=False, record=None):
    """EOC table; record: a study_io.Record that collects the levels."""
    lam, mu = lame(1.0, nu)
    u = CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y)))
    sigma, gamma, f = manufactured(u, lam, mu)
    print(f"traction on the hole, k = {order}, nu = {nu}" + (", whole sigma" if full else "")
          + (", pruned" if prune else ", not pruned"))
    print(HEADER + "      R   #D_h  removed")
    prev, rows = None, []
    for h in hs:
        mesh = background_mesh(h)
        sol, D = solve(mesh, order, lam, mu, f, u, at(sigma), PHI, full=full, prune=prune)
        e = errors(mesh, sol, u, sigma, gamma, D)
        R = path_ratio(mesh, PHI, D)
        removed = unfitted_domain(mesh, PHI)[0].NumSet() - D.NumSet()
        print(eoc_line(h, e, prev) + f"  {R:5.3f}  {D.NumSet():5d}  {removed:7d}")
        if record is not None:
            record.add(h, mesh, e, sol[1].gfu, R=R, nD=D.NumSet(), removed=removed)
        prev = (h, e)
        rows.append((e, R, D.NumSet(), removed))
    return rows


if __name__ == "__main__":
    from elasticity import COLUMNS
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, ["traction"])
    p.add_argument("--nu", type=float, default=0.3, help="Poisson ratio (default 0.3)")
    p.add_argument("--prune", action="store_true", help="prune D_h (MATLAB rules)")
    p.add_argument("--full", action="store_true", help="extrapolate the whole stress (default: P_k part)")
    a = parse(p)
    if a.k is None:
        # one process per case, within 2 GB each (k = 2 at h = 0.025 needs about 4 GB: not a default)
        run_defaults(__file__, [["--k", k, "--h", *hs]
                                for k, hs in ((1, (0.2, 0.1, 0.05, 0.025)), (2, (0.2, 0.1, 0.05)))])
    else:
        record = Record(COLUMNS)
        study(a.k, a.h, nu=a.nu, full=a.full, prune=a.prune, record=record)
        if a.out:
            save(a.out, record, dict(driver="elasticity_tpm_neumann", case=a.case, k=a.k, E=1.0,
                                     nu=a.nu, tau=1.0, full=a.full, prune=a.prune))

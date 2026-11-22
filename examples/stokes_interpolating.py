"""Stokes flow on interpolating meshes (the meshes of darcy_interpolating.py: Omega_h
bounded by the polygon through points of Gamma, gap O(h^2)), standalone. Cases:

    disk, flower   u = g on Gamma: the formulation of stokes_tpm.py (same exact solution,
                   pressure p_h = ptilde_h + alpha with the multiplier rho and the level
                   alpha fixed by RegionIntegral over the gap);
    traction       sigma nu = gN nu on the circle r = 0.6, u = g on the box: the stress form
                   of stokes_stress_tpm_neumann.py (P_k part of the stress, no multiplier,
                   its exact solution).

Variants (the suffix of the case key):

    -tpm      the conditions transferred to xbar on Gamma along closest-point paths, as in
              stokes_tpm.solve and stokes_stress_tpm_neumann.solve;
    -closest  the data of Gamma at xbar imposed at x with no transfer: uhat(x) = g(xbar),
              or sigma_hat n (the HDG traction, n the normal of Gamma_h) = gN(xbar) nubar.

solve_dirichlet and solve_traction take the element set D and the measure of Gamma_h,
so that stokes_snapped.py uses them on snapped meshes.
"""

from ngsolve import *

from convergence import coefficients, main
from darcy_interpolating import CURVES, check_polygon, interpolating_mesh
from elasticity_tpm import at
from poisson import solve_linear_system
from stokes import multiplier_inverse, shifted, stokes_data
from ngstpm import ParametricCurve, boundary_ds, interior_ds, restrict, transfer_paths
import stokes_stress as st


def solve_dirichlet(mesh, D, dB, facets, dirichlet, order, f, g, curve, nu=1.0, tau=1.0, transfer=True):
    """stokes_tpm.solve on the element set D with Gamma_h measured by dB (facets: its facet
    set or boundary Region, for transfer_paths), uhat = g on the mesh boundaries `dirichlet`;
    transfer=False: uhat(x) = g(xbar) on Gamma_h."""
    p = transfer_paths(mesh, curve, D, kind="closest", order=order + 4, facets=facets)
    X = (x, y)
    n = specialcf.normal(2)
    dV = dx(definedonelements=D)
    dS = dx(element_boundary=True, definedonelements=D)
    dI = interior_ds(mesh, D)

    S = restrict(MatrixValued(L2(mesh, order=order)), D)
    V = restrict(VectorL2(mesh, order=order), D)
    W = restrict(L2(mesh, order=order, lowest_order_wb=True), D)
    M = restrict(FacetFESpace(mesh, order=order, dirichlet=dirichlet) ** 2, D)
    R = NumberSpace(mesh)
    A = NumberSpace(mesh)
    fes = S * V * W * M * R * A

    (L, u, ph, uhat, rho, alpha), (G, v, q, mu, rho_, alpha_) = fes.TnT()
    div_G = fem.Einsum("ijj->i", grad(G))
    sigmahat_n = nu * L * n - ph * n - tau * nu * (u - uhat)

    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    lhs += InnerProduct(L, G) * dV + u * div_G * dV - uhat * (G * n) * dS
    lhs += nu * InnerProduct(L, grad(v)) * dV - ph * Trace(grad(v)) * dV - sigmahat_n * v * dS
    rhs += f * v * dV
    lhs += -u * grad(q) * dV + (uhat * n) * q * dS
    lhs += sigmahat_n * mu * dI
    lhs += (uhat.Transfer(p, gradient=L) if transfer else uhat) * mu * dB
    rhs += g(*p.target) * mu * dB
    lhs += ph * rho_ * dV + rho * q * dV
    lhs += alpha * alpha_ * dV + (ph + alpha).RegionIntegral(p) * alpha_ * dB

    gfu = GridFunction(fes)
    gfu.components[3].Set(g(*X), definedon=mesh.Boundaries(".*"))   # only the Dirichlet dofs count
    solve_linear_system(lhs, rhs, gfu, inverse=multiplier_inverse(fes))
    Lh, uh, pt, uhath, rhoh, alphah = gfu.components
    a = alphah.vec[0]
    return (Lh, uh, shifted(pt, a), uhath, rhoh), a


def solve_traction(mesh, D, dB, facets, dirichlet, order, f, g, gN, curve, mu=1.0, tau=1.0, transfer=True):
    """stokes_stress_tpm_neumann.solve on D, Gamma_h measured by dB, u = g on `dirichlet`;
    transfer=False: sigma_hat n = gN(xbar) nubar on Gamma_h. Returns the solution and p_h."""
    p = transfer_paths(mesh, curve, D, kind="closest", order=order + 4, facets=facets)
    xbar, nubar = p.target, p.target_normal
    n = specialcf.normal(2)
    assert Integrate(nubar * n * dB, mesh) > 0.9 * Integrate(CF(1) * dB, mesh)   # nubar out of D_h
    fes, top, lhs, rhs, trial, test, sigmahat_n = st.forms(mesh, order, mu, f, dirichlet, tau, D,
                                                           constrained=False)
    sP, u, uhat = trial[0], trial[3], trial[5]
    muh = test[5]
    lhs += sigmahat_n * muh * interior_ds(mesh, D)
    if transfer:
        lhs += (sP.Extrapolate(xbar) * nubar - tau * mu * (u - uhat)) * muh * dB
    else:
        lhs += sigmahat_n * muh * dB
    rhs += gN(*xbar) * nubar * muh * dB
    sol, p_h, _ = st.finish(lhs, rhs, fes, top, (g, BND, mesh.Boundaries(dirichlet)), mesh, order, mu,
                            D, constrained=False)
    return sol, p_h


def dirichlet_errors(mesh, D, k, sol, alpha, rho, uex, pex, Lex):
    Lh, uh, ph, _, _ = sol
    dV = dx(definedonelements=D, bonus_intorder=4)
    e = [sqrt(Integrate(InnerProduct(a - b, a - b) * dV, mesh))
         for a, b in ((Lh, Lex), (uh, uex), (ph, pex), (ph - alpha, pex))]
    return dict(zip(("|L-Lh|", "|u-uh|", "|p-ph|", "unshifted"), e), **{"|rho|": abs(rho.vec[0]), "alpha": alpha})


def traction_errors(mesh, D, sol, p_h):
    sigma, gamma, _ = st.manufactured(st.U, st.P, 1.0)
    e = st.errors(mesh, sol, p_h, st.U, st.P, sigma, gamma, D)
    dV = dx(definedonelements=D, bonus_intorder=4)
    c = Integrate((p_h - st.P) * dV, mesh) / Integrate(CF(1) * dV, mesh)
    return {"|s-sh|": e[0], "|u-uh|": e[2], "|p-ph|": e[1], "mean(ph-p)": c}


def dirichlet_data():
    """The solution of stokes_tpm.row: g (callable), u, p, L, f."""
    g = lambda X, Y: CF((sin(pi * X) * sin(pi * Y), cos(pi * X) * cos(pi * Y)))
    uex, pex = g(x, y), sin(2 * pi * x) * sin(2 * pi * y)
    Lex, f = stokes_data(uex, pex, 1.0)
    return g, uex, pex, Lex, f


def run(case, k, mesh, D, dB, facets, dirichlet, curve, pex=None, uex=None):
    """One row: case = geometry-variant ("disk", "flower" or "traction"); pex, uex: a
    polynomial solution instead (Dirichlet cases, for the exactness check)."""
    geometry, variant = case.rsplit("-", 1)
    transfer = variant == "tpm"
    if geometry == "traction":
        sigma, _, f = st.manufactured(st.U, st.P, 1.0)
        sol, p_h = solve_traction(mesh, D, dB, facets, dirichlet, k, f, st.U, at(sigma), -curve, transfer=transfer)
        return traction_errors(mesh, D, sol, p_h)
    g, u0, p0, L0, f = dirichlet_data()
    if pex is not None:
        g = lambda X, Y: uex.Replace({x: X, y: Y})
        u0, p0 = uex, pex
        L0, f = stokes_data(u0, p0, 1.0)
    sol, alpha = solve_dirichlet(mesh, D, dB, facets, dirichlet, k, f, g, curve, transfer=transfer)
    return dirichlet_errors(mesh, D, k, sol, alpha, sol[4], u0, p0, L0)


GEOMETRY = {"disk": "disk", "flower": "flower", "traction": "neumann"}           # darcy_interpolating meshes
CASES = {f"interpolating-{g}-{v}": {k: (0.2, 0.1, 0.05, 0.025) for k in (1, 2, 3)}
         for g in GEOMETRY for v in ("tpm", "closest")}


def row(case, k, maxh, solution=False, **kw):
    name = case[len("interpolating-"):]
    geometry = GEOMETRY[name.rsplit("-", 1)[0]]
    mesh, P = interpolating_mesh(geometry, maxh)
    check_polygon(mesh, P, geometry)
    D = BitArray(mesh.ne)
    D.Set()
    facets = mesh.Boundaries("gamma")
    dB = boundary_ds(mesh, D, facets=facets, bonus_intorder=max(6, 2 * k))
    return run(name, k, mesh, D, dB, facets, "box", ParametricCurve(CURVES[geometry]), **kw)


if __name__ == "__main__":
    import sys
    if any("traction" in a for a in sys.argv):
        main(row, CASES, ("|s-sh|", "|u-uh|", "|p-ph|"), {"mean(ph-p)": ".2e"}, hfmt="6.4f")
    else:
        main(row, CASES, ("|L-Lh|", "|u-uh|", "|p-ph|", "unshifted"), {"|rho|": ".1e", "alpha": ".1e"}, hfmt="6.4f")

"""Transient linear FSI in velocity on both sides: the fluid of fsi_transient.py and the solid
of elastodynamics.VelocityStress, Crank-Nicolson on every row (Meddahi 2025, (40)-(41)),

    rho_F u_F' - div sigma_F = f_F,   A_F sigma_F = grad u_F - S(gamma_F)          in Omega_F,
    rho_S v' + b0 eta - div sigma_S = f_S,   A_S sigma_S' = grad v - S(omega)      in Omega_S,
    u_F - v = g,   sigma_F n_F + sigma_S n_S = g_nu                                  on Sigma,

v = eta' the solid VELOCITY, omega = gamma_S'. Unknowns: (sigma_F, u_F, gamma_F, uhat_F, xi,
rho) as in fsi_transient.py (pressure split) and (sigma_S, v, omega, vhat) as in
VelocityStress (tau_S sqrt(rho_S mu_S) on v - vhat); eta by the trapezoid rule.

One step solves for the midpoint values y = (y^n + y^{n+1})/2 (data averaged over t^n,
t^{n+1}), then y^{n+1} = 2 y - y^n for every dof:
  fluid: the HDG rows with (2 rho_F/dt) (u, v) and (2 rho_F/dt) (u^n, v) on the right;
  solid: the rows of VelocityStress (masses 2/dt (A sigma, tau), (2 rho_S/dt + b0 dt/2) (v, w));
  Gamma_h^F, traction (as fsi_transient.py, the solid flux on v - vhat):
      -sigma_F nubar - tau_F mu_F (u_F - uhat_F) + sigma_S nubar - tau_S (v - vhat) = g_nu;
  Gamma_h^S, kinematic: T(uhat_F; A_F sigma_F + S(gamma_F)) - T(vhat; A_S sigma_S' + S(omega))
      = g(xbar), sigma_S' = (2/dt)(sigma_S - sigma_S^n) (no 2/dt scaling of the row, no history
      but the old solid stress).
Start (Meddahi (41)): L2 projections of the exact fields at t = 0; sigma_F and sigma_S onto the
whole stress space (P_k + curl bubbles), rho onto P_0 (the element mean of tr sigma_F), xi = 0,
uhat_F and vhat by P_M. The fluid stress, pressure and the solid omega are algebraic: their
nodal values carry the start error with alternating sign.
Departs from MATLAB (displacement solid, Newmark; fsi_transient.py follows it).
"""
from math import log2

from ngsolve import *

import elasticity as el
import fsi_transient as ft
import stokes_stress as st
from component_bc import INF, NONE, TABLE_F, TABLE_S, free_components, set_dirichlet, traction_rows
from elastodynamics import velocity_rows
from elasticity import compliance, free_dofs, hdg_rows, postprocess, skew, spaces, stress
from fsi_transient import facet_projection, t
from ngstpm import interior_ds, pair, transfer_paths

S0 = 8                                              # the first solid component


class FSIVelocity:
    """The forms of one run (mesh, k, dt fixed), the state U (all dofs at t^n), the solid
    displacement E (its v block) and the previous load F. Arguments as ft.FSITransient, except
    bcS = (table, g_D, g_N) with g_D the solid VELOCITY."""

    def __init__(self, mesh, order, phi, DF, DS, mat, fF, fS, g, gS, bcF, bcS, dt, tau=(1.0, 1.0), full=True,
                 facets=(None, None)):
        self.mesh, self.order, self.DF, self.DS, self.mat, self.dt = mesh, order, DF, DS, mat, dt
        self.bc = (bcF, bcS)
        muF, rhoF, lamS, muS, rhoS, b0 = (mat[k] for k in ("mu_F", "rho_F", "lam_S", "mu_S", "rho_S", "b0"))
        tauS = tau[1] * (rhoS * muS) ** 0.5
        cs = 2 / dt

        pF, pS = (transfer_paths(mesh, phi, D, order=order + 4, facets=F) for D, F in zip((DF, DS), facets))
        I = pair(pF, pS)                                              # side 1: fluid, side 2: solid
        dF, dS = (I.ds(side=i, bonus_intorder=max(6, 2 * order)) for i in (1, 2))
        spF, spS = spaces(mesh, order, NONE, DF, split=True), spaces(mesh, order, NONE, DS)
        self.tops = (spF[1], spS[1])
        self.fes = fes = spF[0] * spS[0]
        trial, test = self.trial, self.test = fes.TnT()
        dVF, dVS = dx(definedonelements=DF), dx(definedonelements=DS)
        lhs = BilinearForm(fes, condense=True)
        load = LinearForm(fes)

        sigmahat_nF = hdg_rows(lhs, load, trial[:S0], test[:S0], INF, muF, fF, tau[0], DF)
        lhs += 2 * rhoF / dt * trial[3] * test[3] * dVF
        sigmahat_nS = velocity_rows(lhs, trial[S0:], test[S0:], lamS, muS, tauS, cs, 2 * rhoS / dt + b0 * dt / 2, DS)
        load += fS * test[S0 + 3] * dVS
        for sh, muh, D, (table, _, gN) in ((sigmahat_nF, test[5], DF, bcF), (sigmahat_nS, test[S0 + 5], DS, bcS)):
            lhs += sh * muh * interior_ds(mesh, D)
            traction_rows(mesh, lhs, load, sh, muh, table, gN, D)
        (sPF, wF1, wF2, uF, gamF, uhatF), (sPS, wS1, wS2, v, om, vhat) = trial[:6], trial[S0:S0 + 6]

        # Gamma_h^F, traction:  sigma_F n_F + sigma_S n_S = g_nu  at xbar, n_S = nubar = -n_F
        xbar, nubar = pF.target, pF.target_normal
        tF = -sPF.Extrapolate(xbar) * nubar - tau[0] * muF * (uF - uhatF)
        tS = sPS.Extrapolate(xbar) * nubar - tauS * (v.Extrapolate() - vhat.Extrapolate())
        lhs += (tF + tS) * test[5] * dF
        if gS is not None:
            load += gS(*xbar) * nubar * test[5] * dF

        # Gamma_h^S, kinematic:  T(uhat_F) - T(vhat) = g  at xbar
        sF = stress(sPF, wF1, wF2) if full else sPF
        sS = stress(sPS, wS1, wS2) if full else sPS
        ubF = uhatF.Transfer(pF, gradient=compliance(sF, INF, muF) + skew(gamF))
        ubS = vhat.Transfer(pS, gradient=cs * compliance(sS, lamS, muS) + skew(om))
        lhs += (ubF - ubS) * test[S0 + 5] * dS
        load += g(*pS.target) * test[S0 + 5] * dS

        self.H = H = BilinearForm(fes)                               # the old-state terms
        H += 2 * rhoF / dt * uF * test[3] * dVF
        H += cs * (InnerProduct(compliance(stress(sPS, wS1, wS2), lamS, muS), stress(*test[S0:S0 + 3]))
                   + rhoS * v * test[S0 + 3]) * dVS
        H += -cs * compliance(sS, lamS, muS).PathIntegral(pS) * test[S0 + 5] * dS
        self.B0 = BilinearForm(fes)
        self.B0 += b0 * v * test[S0 + 3] * dVS
        for a in (H, self.B0):
            a.Assemble()
        self.lhs, self.load = lhs, load

        free = free_dofs(fes, spF[1], True)
        free = free_dofs(fes, spS[1], True, free=free, first=S0)
        for (table, _, _), comp in ((bcF, 5), (bcS, S0 + 5)):
            free = free_components(fes, table, free, comp)
        self.free = free
        lhs.Assemble()
        Id = IdentityMatrix(lhs.mat.height)
        Ainv = lhs.mat.Inverse(free, inverse="umfpack")
        self.Ainv = (Id + lhs.harmonic_extension) @ (Ainv + lhs.inner_solve) @ (Id + lhs.harmonic_extension_trans)
        self.gfu = GridFunction(fes)
        self.U, self.E, self.F = (self.gfu.vec.CreateVector() for _ in range(3))
        for vec in (self.U, self.E, self.F):
            vec[:] = 0
        self.time, self.steps = 0.0, 0

    def step(self):
        """One Crank-Nicolson step to t^{n+1}."""
        dt = self.dt
        t.Set(self.time + dt)
        self.load.Assemble()
        b = self.load.vec.CreateVector()
        b.data = 0.5 * (self.load.vec + self.F) + self.H.mat * self.U - self.B0.mat * self.E
        self.F.data = self.load.vec
        gfu = self.gfu
        gfu.vec[:] = 0
        for (table, gD, _), comp in zip(self.bc, (5, S0 + 5)):
            set_dirichlet(gfu, table, 0.5 * (gD + gD.Replace({t: t - dt})), comp)
        b.data -= self.lhs.mat * gfu.vec
        gfu.vec.data += self.lhs.harmonic_extension * gfu.vec
        gfu.vec.data += self.Ainv * b
        self.U.data *= -1                  # U.data = 2 y - U would alias
        self.U.data += 2 * gfu.vec
        self.E.data += dt * gfu.vec        # only the v block is read
        self.time += dt
        self.steps += 1

    def march(self, nsteps, monitor=None):
        for _ in range(nsteps):
            self.step()
            if monitor is not None:
                monitor(self)

    def initial_state(self, exF, exS):
        """Meddahi (41) at t = 0: exF = (sigma, gamma, u) of the fluid, exS = (sigma, v, omega,
        eta) of the solid (CFs at the Parameter t)."""
        t.Set(0.0)
        gf = GridFunction(self.fes)
        gf.vec[:] = 0
        c = gf.components
        for first, D, (sigma, u, gamma), top in ((0, self.DF, exF, self.tops[0]),
                                                 (S0, self.DS, exS[:3], self.tops[1])):
            self._project_stress(gf, first, sigma, D, top)
            c[first + 3].Set(u, definedonelements=D)
            c[first + 4].Set(gamma, definedonelements=D)
            facet_projection(c[first + 5], u, D)
        c[7].Set(Trace(exF[0]), definedonelements=self.DF)
        self.U.data = gf.vec
        gf.vec[:] = 0
        c[S0 + 3].Set(exS[3], definedonelements=self.DS)
        self.E.data = gf.vec
        self.load.Assemble()
        self.F.data = self.load.vec
        self.time, self.steps = 0.0, 0

    def _project_stress(self, gf, first, sigma, D, top):
        """Components first..first+2 of gf := the L2 projection of sigma onto P_k + curl(B_K)."""
        fes, (sP, w1, w2), tests = self.fes, self.trial[first:first + 3], stress(*self.test[first:first + 3])
        dV = dx(definedonelements=D)
        mass = BilinearForm(InnerProduct(stress(sP, w1, w2), tests) * dV).Assemble()
        load = LinearForm(InnerProduct(sigma, tests) * dV).Assemble()
        mask = BitArray(fes.ndof)
        mask.Clear()
        for comp in range(first, first + 3):
            for d in fes.Range(comp):
                mask.Set(d)
        free = free_dofs(fes, top, False, free=fes.FreeDofs() & mask, first=first)
        tmp = gf.vec.CreateVector()
        tmp.data = mass.mat.Inverse(free, inverse="umfpack") * load.vec
        gf.vec.data += tmp

    def fields(self):
        """(fluid (sigma, u, gamma, uhat, u*), p, solid (sigma, v, omega, vhat, eta)) of U, E."""
        gf, ge = GridFunction(self.fes), GridFunction(self.fes)
        gf.vec.data = self.U
        ge.vec.data = self.E
        c, m, k = gf.components, self.mat, self.order
        sigF = stress(*c[:3])
        fluid = (sigF, c[3], c[4], c[5], postprocess(self.mesh, k, INF, m["mu_F"], sigF, c[3], c[4], self.DF))
        c[3].gfu = gf
        return fluid, -Trace(sigF) / 2, (stress(*c[S0:S0 + 3]), c[S0 + 3], c[S0 + 4], c[S0 + 5], ge.components[S0 + 3])

    def errors(self, exact):
        """Fluid: stokes_stress.errors (|s-sh|, |p-ph|, |u-uh|, |g-gh|, |Pu-uh^|h, |u-u*|); solid:
        |s-sh|, |v-vh|, |w-wh|, |e-eh|, |v-vh^|h; exact = (uF, pF, (sigF, gamF), (sigS, v, omega,
        eta)) at the Parameter t."""
        uF, pF, (sF, gF), (sS, v, om, eta) = exact
        fluid, p_h, solid = self.fields()
        eF = st.errors(self.mesh, fluid, p_h, uF, pF, sF, gF, self.DF)[:6]
        dV = dx(definedonelements=self.DS, bonus_intorder=4)
        dS = dx(element_boundary=True, definedonelements=self.DS, bonus_intorder=4)
        l2 = lambda a: sqrt(Integrate(InnerProduct(a, a) * dV, self.mesh))
        h = specialcf.mesh_size
        eS = (l2(solid[0] - sS), l2(solid[1] - v), l2(solid[2] - om), l2(solid[4] - eta),
              sqrt(Integrate(h * InnerProduct(v - solid[3], v - solid[3]) * dS, self.mesh)))
        return list(eF) + list(eS)


COLUMNS = ("F|s-sh|", "F|p-ph|", "F|u-uh|", "F|g-gh|", "F|Pu-uh^|", "F|u-u*|",
           "S|s-sh|", "S|v-vh|", "S|w-wh|", "S|e-eh|", "S|v-vh^|h")


def run(mesh, DF, DS, order, uF, pF, eta, mat, dt, nsteps, tables=(TABLE_F, TABLE_S), monitor=None, tau=(1.0, 1.0),
        facets=(None, None)):
    """Set up, start from Meddahi (41) and march nsteps; returns the FSIVelocity object, with
    exact = the argument of errors."""
    d = ft.manufactured(uF, pF, eta, mat)
    n = specialcf.normal(2)
    (sF, gF, fF), (sS, gS_, fS) = d["exF"], d["sS"]
    v = ft.dt_(eta)
    r = FSIVelocity(mesh, order, ft.PHI, DF, DS, mat, fF, fS, d["g"], d["gS"], (tables[0], uF, sF * n),
                    (tables[1], v, sS * n), dt, tau=tau, facets=facets)
    r.exact = (uF, pF, (sF, gF), (sS, v, ft.dt_(gS_), eta))
    r.initial_state((sF, uF, gF), r.exact[3])
    r.march(nsteps, monitor)
    return r


def study_h(order, hs, T, data, mat, tables=(TABLE_F, TABLE_S), dt_of=lambda h: h):
    """Space-time EOC table at t = T, dt = T/round(T/dt_of(h)), with the FSITransient
    (Newmark solid, Euler start) errors of its columns S|s-sh|, S|e-eh|, S|v-vh| alongside."""
    print(f"k = {order}, T = {T}, {mat}")
    print(f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS) + "   Newmark S|s|, S|e|, S|v|")
    prev = None
    for h in hs:
        n = max(1, round(T / dt_of(h)))
        mesh, DF, DS = ft.two_layer_mesh(h)
        r = run(mesh, DF, DS, order, **data, mat=mat, dt=T / n, nsteps=n, tables=tables)
        e = r.errors(r.exact)
        nm = ft.run(mesh, DF, DS, order, **data, mat=mat, dt=T / n, nsteps=n, tables=tables)
        en = nm.errors(nm.exact)
        line = f"  {h:7.4f}"
        for j, ej in enumerate(e):
            r = f"{log2(prev[1][j] / ej) / log2(prev[0] / h):4.2f}" if prev else "  - "
            line += f"  {ej:9.3e}  {r}"
        print(line + "   " + "  ".join(f"{en[j]:.3e}" for j in (6, 7, 11)))
        prev = (h, e)



if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=("meddahi", "from-rest"), default="meddahi")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--h", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    p.add_argument("--T", type=float, help="final time (default: meddahi 0.3, from-rest 0.5)")
    a = p.parse_args()
    if a.case == "meddahi":
        m = dict(ft.MEDDAHI)
        mat = m.pop("mat")
        study_h(a.k, a.h, a.T or ft.T_MEDDAHI, m, mat, ft.DIRICHLET_TABLES)
    else:
        study_h(a.k, a.h, a.T or 0.5, ft.from_rest(a.k), ft.material())

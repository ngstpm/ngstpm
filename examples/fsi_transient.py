"""Transient linear FSI on a fitted mesh (stage B of reports/agentB-hemodynamics-plan.md): the
rows of fsi_interface.py / component_bc.solve_fsi with inertia, a Winkler spring and the time
loop of MATLAB's FluidStructureCoupling (hemodynamics.m:500-545),

    rho_F du_F/dt - div sigma_F = f_F,   A_F sigma_F = grad u_F - S(gamma_F)          in Omega_F,
    rho_S eta'' + b0 eta - div sigma_S = f_S,   A_S sigma_S = grad eta - S(gamma_S)    in Omega_S,
    u_F - eta' = g,   sigma_F n_F + sigma_S n_S = g_nu                                  on Sigma,

eta the solid DISPLACEMENT. One step t^n -> t^{n+1} = (n+1) dt, all data at t^{n+1}:

  fluid, theta-scheme in acceleration form (theta = 1 at step 1, 1/2 after; MATLAB
  IncompressibleElasticityMatrices.m:110-126, get_theta.m:11-18):
      (rho_F/(theta dt)) (u, v) + ... = (f, v) + M_F (u^n/(theta dt) + ((1-theta)/theta) a^n),
      a^{n+1} = (u^{n+1} - u^n)/(theta dt) - ((1-theta)/theta) a^n   (get_kinematic_state.m:43);
  solid, Newmark(beta = 1/4, gamma = 1/2) (LinearElasticityMatrices.m:106-109, 120-127):
      (rho_S/(beta dt^2) + b0) (eta, v) + ... = (f, v) + M_S ((eta^n + dt v^n)/(beta dt^2)
                                                              + ((1/2-beta)/beta) a^n),
      a^{n+1} = (x^{n+1} - x^n - dt v^n)/(beta dt^2) - ((1/2-beta)/beta) a^n,
      v^{n+1} = v^n + dt ((1-gamma) a^n + gamma a^{n+1}), for every solid dof x (sigma_S, eta,
      gamma_S, etahat alike; @LinearElasticityProblem/get_kinematic_state.m:41-72);
  interface (MATLAB sides): traction row on Gamma_h^F as in component_bc.solve_fsi; kinematic
  row on Gamma_h^S, MATLAB's cdt [T(uhat_F)] - [T(etahat)] = cdt g - T(h0) (InterfaceProblem.m:
  340-354, solve.m:958-976, 1124) divided by cdt = (beta/gamma) dt:
      T(uhat_F; A_F sigma_F + S(gamma_F)) - alpha T(etahat; A_S sigma_S + S(gamma_S))
          = g - alpha K_2 (U + c_v V + c_a A),
      alpha = gamma/(beta dt) = 2/dt, c_v = dt (1 - beta/gamma) = dt/2,
      c_a = dt^2 ((1/2 - beta) - beta (1 - gamma)/gamma) = 0.
  With the Newmark update this is uhat_F^{n+1} - v^{n+1} = g on Sigma (g = (u_F - eta')(t^{n+1})).

Every history term is (form assembled once) x (state vector), as MATLAB's element matrix x dofs:
M_F = rho_F (u, v)_{D_F}, M_S = rho_S (eta, v)_{D_S} and K_2 = <T(etahat; ...), mu_2>_{Gamma_h^S}
(the solid half of the kinematic row) are non-condensed BilinearForms on the product space, U, V,
A the state vectors (value, velocity, acceleration) of all dofs; the fluid uses the U and A
blocks, the solid all three. The condensed left-hand side has theta as a Parameter; it is
assembled and factorised once per theta (Euler start: twice), each step is the algebra of
elasticity.solve_linear_system without reassembly. Only the load (f, g_N, g, g_nu at the
Parameter t) is reassembled per step.

Differences from MATLAB (followed otherwise):
- Initial state: L2 projections of the exact fields at t = 0 (sigma into its P_k part, the bubble
  part zero; uhat by P_M) for the value, velocity and acceleration of every solid dof and the
  fluid velocity (and acceleration with start="cn"); MATLAB: elliptic projections and a
  consistent acceleration (@LinearElasticityProblem/initial_state.m:45-48,
  +FluidStructureCoupling/initial_state.m:105-109). Identical (zero) in the benchmark.
- start="cn" (control): Crank-Nicolson from step 1 from the projected exact fluid acceleration
  (as tests/test_fsi_transient_polynomial.m:51-54, whose a_F^0 is computed); start="euler-all":
  backward Euler at every step (control).
  Observed consequences of the scheme (not of the port): with the Euler start and u_F''(0) != 0
  the O(dt) pressure error of the Euler step is carried by the trapezoid undamped with
  alternating sign, so sigma_F and p are first order in dt (the rest second order); the CN start
  from the exact a(0) gives 2 everywhere (tests/test_fsi_transient.py, B2). L2-projected initial
  accelerations do not satisfy the discrete equations; Newmark and the trapezoid carry that
  undamped, and the dt self-convergence is irregular where the space error is large (B3).
- Stabilisation tau_F mu_F, tau_S mu_S (decision 2; tau_S = 1/mu_S gives MATLAB's solid tau = 1,
  LinearElasticityMatrices.m:45, 105).
- The rest as component_bc.solve_fsi (enrichment, P_k traction row, interface quadrature).
- Solver: UMFPACK (default) or SciPy SuperLU (solver="superlu"); MATLAB a sparse direct solve.
"""
import subprocess
from math import log2, pi as PI
import os

import numpy as np
from ngsolve import *

import elasticity as el
import fsi_interface as fi
import stokes_stress as st
from component_bc import (DIRICHLET, INF, NONE, PHI, TABLE_F, TABLE_S, free_components, set_dirichlet,
                          traction_rows, two_layer_mesh)
from elasticity import compliance, free_dofs, hdg_rows, postprocess, skew, spaces, stress
from elasticity_tpm import at
from ngstpm import interior_ds, pair, transfer_paths

BETA, GAMMA = 0.25, 0.5
t = Parameter(0.0)                                  # the time of the data
STARTS = ("euler", "cn", "euler-all")


def rss_mb():
    """Resident set size of this process in MB (ps; macOS RSS)."""
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
    return int(r.stdout.strip()) / 1024


class FSITransient:
    """The forms of one run (mesh, k, dt fixed) and the state vectors U, V, A.

    mat: dict mu_F, rho_F, lam_S, mu_S, rho_S, b0; fF, fS: loads (vector CFs in x, y, t);
    g(X, Y): kinematic data at xbar; gS(X, Y): the stress jump sigma_S - sigma_F at xbar
    (g_nu = gS(xbar) n_S), None = 0; bcF, bcS = (table, g_D, g_N) of component_bc.
    facets = (Gamma_h^F, Gamma_h^S): boundary Regions or facet BitArrays, for D_F, D_S not
    sharing a face (a gap); default the faces between D_F and D_S (transfer_paths); kind: the
    path kind of transfer_paths ("vertical" = MATLAB's SmoothCurveHorizontal). With full, the
    traction row extrapolates the whole stress sigma_P + curl(w), as the preprint and MATLAB."""

    def __init__(self, mesh, order, phi, DF, DS, mat, fF, fS, g, gS, bcF, bcS, dt, tau=(1.0, 1.0),
                 full=True, solver="umfpack", facets=(None, None), kind=None):
        self.mesh, self.order, self.DF, self.DS, self.mat, self.dt = mesh, order, DF, DS, mat, dt
        self.solver, self.bc = solver, (bcF, bcS)
        muF, rhoF, lamS, muS, rhoS, b0 = (mat[k] for k in ("mu_F", "rho_F", "lam_S", "mu_S", "rho_S", "b0"))
        self.alpha = alpha = GAMMA / (BETA * dt)
        self.c_v = dt * (1 - BETA / GAMMA)
        self.c_a = dt ** 2 * ((0.5 - BETA) - BETA * (1 - GAMMA) / GAMMA)
        self.theta = Parameter(1.0)

        pF, pS = (transfer_paths(mesh, phi, D, kind=kind, order=order + 4, facets=F) for D, F in zip((DF, DS), facets))
        self.paths = (pF, pS)
        I = pair(pF, pS)                                              # side 1: fluid, side 2: solid
        dF, dS = (I.ds(side=i, bonus_intorder=max(6, 2 * order)) for i in (1, 2))
        spF, spS = spaces(mesh, order, NONE, DF, split=True), spaces(mesh, order, NONE, DS)
        self.fes = fes = spF[0] * spS[0]
        trial, test = fes.TnT()
        lhs = BilinearForm(fes, condense=True)
        load = LinearForm(fes)
        for tr, te, lam, mu, f, D, (table, _, gN), ta in ((trial[:8], test[:8], INF, muF, fF, DF, bcF, tau[0]),
                                                          (trial[8:], test[8:], lamS, muS, fS, DS, bcS, tau[1])):
            sigmahat_n = hdg_rows(lhs, load, tr, te, lam, mu, f, ta, D)
            lhs += sigmahat_n * te[5] * interior_ds(mesh, D)
            traction_rows(mesh, lhs, load, sigmahat_n, te[5], table, gN, D)
        (sPF, wF1, wF2, uF, gamF, uhatF), (sPS, wS1, wS2, eta, gamS, etahat) = trial[:6], trial[8:]
        dVF, dVS = dx(definedonelements=DF), dx(definedonelements=DS)
        lhs += rhoF / (self.theta * dt) * uF * test[3] * dVF
        lhs += (rhoS / (BETA * dt ** 2) + b0) * eta * test[11] * dVS

        # Gamma_h^F, traction:  sigma_F n_F + sigma_S n_S = g_nu  at xbar, n_S = nubar = -n_F
        xbar, nubar = pF.target, pF.target_normal
        tF = -(stress(sPF, wF1, wF2, at=xbar) if full else sPF.Extrapolate(xbar)) * nubar - tau[0] * muF * (uF - uhatF)
        tS = (stress(sPS, wS1, wS2, at=xbar) if full else sPS.Extrapolate(xbar)) * nubar - tau[1] * muS * (eta.Extrapolate() - etahat.Extrapolate())
        lhs += (tF + tS) * test[5] * dF
        if gS is not None:
            load += gS(*xbar) * nubar * test[5] * dF

        # Gamma_h^S, kinematic:  T(uhat_F) - alpha T(etahat) = g - alpha K_2 (U + c_v V + c_a A)
        xbar = pS.target
        sF = stress(sPF, wF1, wF2) if full else sPF
        sS = stress(sPS, wS1, wS2) if full else sPS
        ubF = uhatF.Transfer(pF, gradient=compliance(sF, INF, muF) + skew(gamF))
        ubS = etahat.Transfer(pS, gradient=compliance(sS, lamS, muS) + skew(gamS))
        lhs += (ubF - alpha * ubS) * test[13] * dS
        load += g(*xbar) * test[13] * dS

        self.MF = BilinearForm(fes)
        self.MF += rhoF * uF * test[3] * dVF
        self.MS = BilinearForm(fes)
        self.MS += rhoS * eta * test[11] * dVS
        self.K2 = BilinearForm(fes)
        self.K2 += ubS * test[13] * dS
        for a in (self.MF, self.MS, self.K2):
            a.Assemble()
        self.lhs, self.load = lhs, load

        free = free_dofs(fes, spF[1], True)
        free = free_dofs(fes, spS[1], True, free=free, first=8)
        for (table, _, _), comp in ((bcF, 5), (bcS, 13)):
            free = free_components(fes, table, free, comp)
        self.free = free
        self.gfu = GridFunction(fes)
        self.U, self.V, self.A = (self.gfu.vec.CreateVector() for _ in range(3))
        for v in (self.U, self.V, self.A):
            v[:] = 0
        n = fes.ndof
        self.F = np.zeros(n, dtype=bool)
        for c in range(8):
            self.F[np.array(list(fes.Range(c)))] = True
        self.S = ~self.F
        self.factorizations, self.thetas, self.time, self.steps = 0, [], 0.0, 0
        self._theta = None

    # ---- linear algebra ----------------------------------------------------------------------
    def factor(self, theta):
        """Assemble the condensed matrix at theta and factorise it (the previous factor is released)."""
        self.Ainv = self.Ainv_full = None
        self.theta.Set(theta)
        self.lhs.Assemble()
        if self.solver == "superlu":
            import superlu
            Ainv = superlu.inverse(self.lhs.mat, self.free)
        else:
            Ainv = self.lhs.mat.Inverse(self.free, inverse="umfpack")
        Id = IdentityMatrix(self.lhs.mat.height)
        H, Ht = self.lhs.harmonic_extension, self.lhs.harmonic_extension_trans
        self.Ainv = Ainv
        self.Ainv_full = (Id + H) @ (Ainv + self.lhs.inner_solve) @ (Id + Ht)
        self._theta = theta
        self.factorizations += 1

    def rhs(self):
        """The right-hand side at t^{n+1} (t already set): load + history terms."""
        th, dt = self._theta, self.dt
        self.load.Assemble()
        b = self.load.vec.CreateVector()
        tmp = self.load.vec.CreateVector()
        b.data = self.load.vec
        tmp.data = (1 / (th * dt)) * self.U + ((1 - th) / th) * self.A
        b.data += self.MF.mat * tmp
        tmp.data = (1 / (BETA * dt ** 2)) * self.U + (1 / (BETA * dt)) * self.V + ((0.5 - BETA) / BETA) * self.A
        b.data += self.MS.mat * tmp
        tmp.data = self.U + self.c_v * self.V + self.c_a * self.A
        b.data -= self.alpha * (self.K2.mat * tmp)
        return b

    def step(self, theta):
        """One step to t^{n+1} with the fluid theta (refactorises if theta changes)."""
        if theta != self._theta:
            self.factor(theta)
        t.Set(self.time + self.dt)
        b = self.rhs()
        gfu = self.gfu
        gfu.vec[:] = 0
        for (table, gD, _), comp in zip(self.bc, (5, 13)):
            set_dirichlet(gfu, table, gD, comp)
        b.data -= self.lhs.mat * gfu.vec
        gfu.vec.data += self.lhs.harmonic_extension * gfu.vec
        gfu.vec.data += self.Ainv_full * b
        self.update(gfu.vec, theta)
        self.time += self.dt
        self.steps += 1
        self.thetas.append(theta)

    def update(self, x, theta):
        """The state after a step: fluid acceleration (theta-scheme), solid Newmark."""
        dt = self.dt
        X, U, V, A = (v.FV().NumPy() for v in (x, self.U, self.V, self.A))
        F, S = self.F, self.S
        A[F] = (X[F] - U[F]) / (theta * dt) - ((1 - theta) / theta) * A[F]
        a1 = (X[S] - U[S] - dt * V[S]) / (BETA * dt ** 2) - ((0.5 - BETA) / BETA) * A[S]
        V[S] += dt * ((1 - GAMMA) * A[S] + GAMMA * a1)
        A[S] = a1
        U[:] = X

    def march(self, nsteps, start="euler", monitor=None):
        """nsteps steps; start: "euler" (theta = 1 at step 1, then 1/2), "cn" (1/2 from step 1,
        needs the fluid acceleration in A), "euler-all" (theta = 1 throughout). monitor(self)
        after each step."""
        for n in range(nsteps):
            theta = 1.0 if start == "euler-all" or (start == "euler" and self.steps == 0) else 0.5
            self.step(theta)
            if monitor is not None:
                monitor(self)

    # ---- state and errors ----------------------------------------------------------------------
    def initial_state(self, exS, uF_t, start="euler"):
        """L2 projections at t = 0 of the exact fields: U, V, A of the solid from
        exS(order) = (sigma, eta, gamma) of the displacement, velocity and acceleration; the
        fluid velocity u_F into U and, for start="cn", du_F/dt into A."""
        t.Set(0.0)
        for vec, s in ((self.U, 0), (self.V, 1), (self.A, 2)):
            self.project(vec, 8, *exS(s))
        self.project(self.U, 0, None, uF_t(0), None, fluid_only_u=True)
        if start == "cn":
            self.project(self.A, 0, None, uF_t(1), None, fluid_only_u=True)
        self.time, self.steps = 0.0, 0

    def project(self, vec, first, sigma, u, gamma, fluid_only_u=False):
        """vec[components first..first+5] := L2 projections of (sigma -> sP, u, gamma, u -> uhat
        by P_M); the bubble parts are zero. fluid_only_u: only u."""
        gf = GridFunction(self.fes)
        gf.vec[:] = 0
        c, D = gf.components, (self.DF if first == 0 else self.DS)
        c[first + 3].Set(u, definedonelements=D)
        if not fluid_only_u:
            c[first].Set(sigma, definedonelements=D)
            c[first + 4].Set(gamma, definedonelements=D)
            facet_projection(c[first + 5], u, D)
        X, G = vec.FV().NumPy(), gf.vec.FV().NumPy()
        block = self.F if first == 0 else self.S
        X[block] = G[block]

    def fields(self, vec=None):
        """(fluid (sigma, u, gamma, uhat, u*), p, solid (sigma, eta, gamma, etahat, eta*)) of vec
        (default U = the last solution)."""
        gf = GridFunction(self.fes)
        gf.vec.data = self.U if vec is None else vec
        c, m, k = gf.components, self.mat, self.order
        sigF, sigS = stress(*c[:3]), stress(*c[8:11])
        fluid = (sigF, c[3], c[4], c[5], postprocess(self.mesh, k, INF, m["mu_F"], sigF, c[3], c[4], self.DF))
        solid = (sigS, c[11], c[12], c[13],
                 postprocess(self.mesh, k, m["lam_S"], m["mu_S"], sigS, c[11], c[12], self.DS))
        c[3].gfu = gf
        return fluid, -Trace(sigF) / 2, solid, gf

    def errors(self, exact):
        """The 11 columns of fsi_interface.side_errors at the current time, plus |eta' - v_h|
        (v_h the eta block of V); exact = (uF, pF, exF, eta, exS, eta_dot) at the Parameter t."""
        uF, pF, exF, eta, exS, eta_dot = exact
        fluid, p_h, solid, _ = self.fields()
        e, _ = fi.side_errors(self.mesh, fluid, p_h, solid, self.DF, self.DS, uF, pF, exF, eta, exS)
        gv = GridFunction(self.fes)
        gv.vec.data = self.V
        dV = dx(definedonelements=self.DS, bonus_intorder=4)
        ev = eta_dot - gv.components[11]
        return e + [sqrt(Integrate(InnerProduct(ev, ev) * dV, self.mesh))]


def facet_projection(gf, u, D):
    """gf := P_M u on the element boundaries of D (the L2 projection of elasticity.errors)."""
    M = gf.space
    p, q = M.TnT()
    dS = dx(element_boundary=True, definedonelements=D, bonus_intorder=4)
    mass, load = BilinearForm(p * q * dS).Assemble(), LinearForm(u * q * dS).Assemble()
    gf.vec.data = mass.mat.Inverse(inverse="sparsecholesky") * load.vec


COLUMNS = fi.COLUMNS + ("S|v-vh|",)


# ---- manufactured problems -------------------------------------------------------------------

def dt_(cf, s=1):
    """The s-th time derivative of cf (a CF in t)."""
    for _ in range(s):
        cf = cf.Diff(t)
    return cf


def manufactured(uF, pF, eta, mat):
    """Exact data of the transient problem for u_F, p_F, eta (CFs in x, y, t): exF = (sigma, gamma,
    f_F) with f_F = rho_F du_F/dt - div sigma_F; exS(s) = (sigma, eta, gamma) of the s-th time
    derivative of eta; f_S = rho_S eta'' + b0 eta - div sigma_S; g = u_F - eta', gS = sigma_S -
    sigma_F (callables at xbar)."""
    sF, gF, fF = st.manufactured(uF, pF, mat["mu_F"])
    fF = fF + mat["rho_F"] * dt_(uF)
    sS, gS_, fS = el.manufactured(eta, mat["lam_S"], mat["mu_S"])
    fS = fS + mat["rho_S"] * dt_(eta, 2) + mat["b0"] * eta

    def exS(s):
        e = dt_(eta, s)
        sig, gam, _ = el.manufactured(e, mat["lam_S"], mat["mu_S"])
        return sig, e, gam
    return dict(exF=(sF, gF, fF), sS=(sS, gS_, fS), exS=exS, g=at(uF - dt_(eta)), gS=at(sS - sF),
                uF_t=lambda s: dt_(uF, s), exact=(uF, pF, (sF, gF, fF), eta, (sS, gS_, fS), dt_(eta)))


def material(muF=0.4, rhoF=1.0, E=1.0, nu=0.25, rhoS=1.0, b0=0.0):
    """Default: MATLAB tests/test_fsi_transient_polynomial.m:24-28 (visc 0.4, E = 1, nu = 0.25,
    rho_F = rho_S = 1)."""
    lamS, muS = el.lame(E, nu)
    return dict(mu_F=muF, rho_F=rhoF, lam_S=lamS, mu_S=muS, rho_S=rhoS, b0=b0)


def run(mesh, DF, DS, order, uF, pF, eta, mat, dt, tables=(TABLE_F, TABLE_S), start="euler", solver="umfpack",
        monitor=None, nsteps=None, T=None, tau=(1.0, 1.0)):
    """Set up the problem on the two-layer mesh, start from the projected exact state and march
    nsteps (or T/dt) steps; returns the FSITransient object (its errors() at the final time)."""
    d = manufactured(uF, pF, eta, mat)
    n = specialcf.normal(2)
    bcF = (tables[0], uF, d["exF"][0] * n)
    bcS = (tables[1], eta, d["sS"][0] * n)
    run_ = FSITransient(mesh, order, PHI, DF, DS, mat, d["exF"][2], d["sS"][2], d["g"], d["gS"], bcF, bcS, dt,
                        tau=tau, solver=solver)
    run_.exact = d["exact"]
    run_.initial_state(d["exS"], d["uF_t"], start=start)
    run_.march(nsteps if nsteps is not None else round(T / dt), start=start, monitor=monitor)
    return run_


def times(v, s):
    """The vector CF (v_0 s, v_1 s) of the scalar CFs v_i, s (a product of a vector CF with a
    time factor gives subtensor nodes that the data at xbar (CF.Replace) cannot transform)."""
    return CF(tuple(c * s for c in v))


# Meddahi 2025, (45) with (47), exactly incompressible fluid; the y coefficient is -3/2 (the
# divergence-free one; pdftotext prints 2/3)
Y1 = y + 1
FIELD = (sin(2 * PI * x) ** 2 * sin(8 * PI * Y1 / 3), -1.5 * sin(4 * PI * x) * sin(4 * PI * Y1 / 3) ** 2)
MEDDAHI = dict(uF=times(FIELD, sin(2 * t)), pF=sin(2 * PI * x) * sin(2 * PI * y) * sin(t), eta=times(FIELD, sin(t) ** 2),
               mat=dict(mu_F=1.0, rho_F=1.0, lam_S=1.0, mu_S=1.0, rho_S=1.0, b0=0.0))
D2 = (DIRICHLET, DIRICHLET)
DIRICHLET_TABLES = ({"inflow": D2, "outflow": D2, "symmetry": D2}, {"top": D2, "wall": D2})
T_MEDDAHI = 0.3

# space-polynomial fields (degree <= k; p of degree k - 1) for B1 and B2 (tests/test_component_bc.py)
POLY_U = {1: (1 + 2 * x - y, -3 + x + 4 * y), 2: (x * x - 2 * x * y + y, 3 * y * y + x * y - x)}
POLY_F = {1: ((x + 2 * y, 3 * x - y), CF(0.3)),
          2: ((x * x - 2 * x * y + y * y + 1, -2 * x * y + y * y + x), x - 0.5 + 2 * (y - 0.5))}


def from_rest(k):
    """B2 data: degree <= k in space, time factors with f(0) = f'(0) = 0, f''(0) != 0, different
    for u_F (sin^2 t), p (2 sin^2 t cos t) and eta (1 - cos t): g, g_nu non-zero."""
    return dict(uF=times(POLY_F[k][0], sin(t) ** 2), pF=POLY_F[k][1] * sin(t) * sin(2 * t),
                eta=times(POLY_U[k], 1 - cos(t)))


def matlab_polynomial():
    """B1 data of MATLAB tests/test_fsi_transient_polynomial.m:33-35: u_F = curl(x y (1 + t)),
    p = (x + y)(1 + t), eta = (x, y)(1 + t + t^2/2)."""
    psi = x * y * (1 + t)
    return dict(uF=CF((psi.Diff(y), -psi.Diff(x))), pF=(x + y) * (1 + t), eta=times((x, y), 1 + t + t * t / 2))


def polynomial(k):
    """B1 data with the degree-k fields: u_F, p linear in t, eta quadratic in t."""
    return dict(uF=times(POLY_F[k][0], 1 + t), pF=POLY_F[k][1] * (1 - 2 * t),
                eta=times(POLY_U[k], 1 + t + t * t / 2))


# ---- self-convergence ------------------------------------------------------------------------

DIFF_COLUMNS = ("F|ds|", "F|dp|", "F|du|", "F|dg|", "F|du^|h", "S|ds|", "S|de|", "S|dg|", "S|de^|h", "S|dv|")


def differences(r, U1, V1, U2, V2):
    """Norms of the differences of two states on the same mesh: L2(D) of sigma, p, u, gamma,
    |.|_h of the trace on the element boundaries (fluid; solid with the velocity)."""
    mesh = r.mesh
    gu, gv = GridFunction(r.fes), GridFunction(r.fes)
    gu.vec.data = U1 - U2
    gv.vec.data = V1 - V2
    c = gu.components
    h = specialcf.mesh_size
    out = []
    for first, D in ((0, r.DF), (8, r.DS)):
        dV = dx(definedonelements=D, bonus_intorder=4)
        dS = dx(element_boundary=True, definedonelements=D, bonus_intorder=4)
        l2 = lambda a: sqrt(Integrate(InnerProduct(a, a) * dV, mesh))
        s = stress(*c[first:first + 3])
        out += [l2(s)] + ([l2(Trace(s) / 2)] if first == 0 else []) + [l2(c[first + 3]), l2(c[first + 4]),
                                                                        sqrt(Integrate(h * InnerProduct(c[first + 5], c[first + 5]) * dS, mesh))]
    out.append(sqrt(Integrate(InnerProduct(gv.components[11], gv.components[11])
                              * dx(definedonelements=r.DS, bonus_intorder=4), mesh)))
    return out


# ---- driver ----------------------------------------------------------------------------------

def line(a, e, prev, fmt="{:7.4f}"):
    s = "  " + fmt.format(a)
    for i, ei in enumerate(e):
        r = f"{log2(prev[1][i] / ei) / log2(prev[0] / a):5.2f}" if prev and ei > 0 and prev[1][i] > 0 else "   - "
        s += f"  {ei:9.3e} {r}"
    return s


def header(first, columns):
    return f"  {first:>7}" + "".join(f"  {c:>9} {'eoc':>5}" for c in columns)


def study_b1(order, h, dt, nsteps, data, b0, start="euler", solver="umfpack", record=None):
    mat = material(b0=b0)
    mesh, DF, DS = two_layer_mesh(h)
    print(f"B1 exactness, k = {order}, h = {h}, dt = {dt}, b0 = {b0}, start = {start}")
    print(header("t", COLUMNS) + "  RSS MB")
    worst = []

    def monitor(r):
        e = r.errors(r.exact)
        worst.append(max(e))
        print(line(r.time, e, None, "{:7.3f}") + f"  {rss_mb():6.0f}")
    r = run(mesh, DF, DS, order, mat=mat, dt=dt, nsteps=nsteps, start=start, solver=solver, monitor=monitor, **data)
    print(f"  factorizations {r.factorizations}, thetas {r.thetas}, worst {max(worst):.2e}")
    return r, worst


def study_dt(order, h, T, steps, data, mat, tables, start="euler", solver="umfpack", record=None):
    """EOC in dt of the errors at T (fixed mesh)."""
    mesh, DF, DS = two_layer_mesh(h)
    print(header("dt", COLUMNS) + "  RSS MB")
    prev = None
    for N in steps:
        dt = T / N
        r = run(mesh, DF, DS, order, mat=mat, dt=dt, nsteps=N, start=start, solver=solver, tables=tables, **data)
        e = r.errors(r.exact)
        print(line(dt, e, prev, "{:7.5f}") + f"  {rss_mb():6.0f}", flush=True)
        if record is not None:
            record.add(dt, mesh, e, r.gfu, h_mesh=h, steps=N)       # (dt in the h slot: EOC in dt)
        prev = (dt, e)
        del r


def study_self(order, h, T, steps, data, mat, tables, start="euler", solver="umfpack", record=None):
    """Self-convergence in dt: |X_dt - X_{dt/2}| at T (fixed mesh), EOC from consecutive differences."""
    mesh, DF, DS = two_layer_mesh(h)
    print(header("dt", DIFF_COLUMNS) + "  RSS MB")
    prev, last = None, None
    for N in steps:
        dt = T / N
        r = run(mesh, DF, DS, order, mat=mat, dt=dt, nsteps=N, start=start, solver=solver, tables=tables, **data)
        U, V = r.U.CreateVector(), r.V.CreateVector()
        U.data, V.data = r.U, r.V
        if last is not None:
            d = differences(r, last[1], last[2], U, V)
            print(line(last[0], d, prev, "{:7.5f}") + f"  {rss_mb():6.0f}", flush=True)
            if record is not None:
                record.add(last[0], mesh, d, r.gfu, h_mesh=h, steps=N)
            prev = (last[0], d)
        last = (dt, U, V)
        del r


def study_h(order, hs, T, steps, data, mat, tables, start="euler", solver="umfpack", record=None):
    """Errors at T against h, N steps per level (dt = T/N)."""
    print(header("h", COLUMNS) + "  steps  RSS MB")
    prev = None
    for h, N in zip(hs, steps):
        mesh, DF, DS = two_layer_mesh(h)
        r = run(mesh, DF, DS, order, mat=mat, dt=T / N, nsteps=N, start=start, solver=solver, tables=tables, **data)
        e = r.errors(r.exact)
        print(line(h, e, prev) + f"  {N:5d}  {rss_mb():6.0f}", flush=True)
        if record is not None:
            record.add(h, mesh, e, r.gfu, dt=T / N, steps=N)
        prev = (h, e)
        del r


CASES = ("b1", "b1-matlab", "b2", "b3-dt", "b3-h")


if __name__ == "__main__":
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(CASES))
    p.add_argument("--steps", type=int, nargs="+", help="numbers of steps (b2, b3-dt: one run each; b3-h: one per h)")
    p.add_argument("--T", type=float, help="final time (b2: 1, b3: 0.3)")
    p.add_argument("--dt", type=float, default=0.1, help="b1: time step (default 0.1, 4 steps)")
    p.add_argument("--b0", type=float, default=None, help="spring (b1, b2; default b1 0, b2 2)")
    p.add_argument("--start", choices=STARTS, default="euler", help="fluid start (default euler)")
    p.add_argument("--solver", choices=("umfpack", "superlu"), default="umfpack")
    a = parse(p)
    if a.k is None:
        run_defaults(__file__, [["--case", "b2", "--k", k, "--h", 0.3, "--steps", 8, 16, 32, 64] for k in (1, 2)])
    else:
        cols = DIFF_COLUMNS if a.case == "b3-dt" else COLUMNS
        record = Record(cols)
        if a.case in ("b1", "b1-matlab"):
            data = matlab_polynomial() if a.case == "b1-matlab" else polynomial(a.k)
            study_b1(a.k, a.h[0], a.dt, a.steps[0] if a.steps else 4, data, a.b0 or 0.0, a.start, a.solver)
        elif a.case == "b2":
            b0 = 2.0 if a.b0 is None else a.b0
            T = a.T or 1.0
            print(f"B2 time order, k = {a.k}, h = {a.h[0]}, T = {T}, b0 = {b0}, start = {a.start}, "
                  f"data from rest (u_F sin^2 t, p 2 sin^2 t cos t, eta 1 - cos t), tables F {TABLE_F} S {TABLE_S}")
            study_dt(a.k, a.h[0], T, a.steps, from_rest(a.k), material(b0=b0), (TABLE_F, TABLE_S), a.start,
                     a.solver, record)
        elif a.case == "b3-dt":
            T = a.T or T_MEDDAHI
            print(f"B3 Meddahi Example 1, self-convergence in dt, k = {a.k}, h = {a.h[0]}, T = {T}, start = {a.start}")
            m = dict(MEDDAHI)
            study_self(a.k, a.h[0], T, a.steps, dict(uF=m["uF"], pF=m["pF"], eta=m["eta"]), m["mat"],
                       DIRICHLET_TABLES, a.start, a.solver, record)
        else:
            T = a.T or T_MEDDAHI
            print(f"B3 Meddahi Example 1, h-rates, k = {a.k}, T = {T}, steps {a.steps}, start = {a.start}")
            m = dict(MEDDAHI)
            study_h(a.k, a.h, T, a.steps, dict(uF=m["uF"], pF=m["pF"], eta=m["eta"]), m["mat"],
                    DIRICHLET_TABLES, a.start, a.solver, record)
        if a.out:
            save(a.out, record, dict(driver="fsi_transient", case=a.case, k=a.k, steps=a.steps, T=a.T, start=a.start,
                                     solver=a.solver, b0=a.b0, tau=1.0))

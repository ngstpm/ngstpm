"""Linear elastodynamics: the HDG of elasticity.py marched by Newmark, on a fitted mesh or,
with phi, on an unfitted one by the Transfer Path Method (the rows of elasticity_tpm.solve).

    rho eta'' + b0 eta - div sigma = f,   A sigma = grad eta - S(gamma)   in Omega x (0, T],
    eta = g on Gamma,   eta(0), eta'(0) given,

with Gamma the Dirichlet part of the mesh boundary (fitted) or {phi = 0} reached by the
transfer paths from Gamma_h (TPM; there the fitted part of the mesh boundary keeps
etahat = P_M g, as in elasticity_tpm.py).

Port of the MATLAB solid (LinearElasticityProblem, time_scheme "newmark"):
- One step t^n -> t^{n+1}, all data at t^{n+1}, Newmark(beta = 1/4, gamma = 1/2)
  (LinearElasticityMatrices.m:106-114):
      (rho/(beta dt^2) + b0) (eta, v) + [static HDG rows]
          = (f, v) + M ((U + dt V)/(beta dt^2) + ((1/2 - beta)/beta) A),
  M = rho (eta, v)_{D_h} on the eta block; then for every dof x (sigma, eta, gamma, etahat)
      a^{n+1} = (x^{n+1} - x^n - dt v^n)/(beta dt^2) - ((1/2 - beta)/beta) a^n,
      v^{n+1} = v^n + dt ((1 - gamma) a^n + gamma a^{n+1})      (get_kinematic_state.m:41-72).
  The condensed matrix is factorised once; only the load is reassembled per step.
- Initial state (initial_state.m, consistent_acceleration.m): U and V are the HDG elliptic
  projections of eta(0) and eta'(0), i.e. the static problem (the same rows with b0, no mass)
  with load -div C eps(w) + b0 w and datum w. The eta block of A solves the local momentum
  row, rho (a, v)_K = (f(0), v)_K - [static eta row of U](v), so a = Pi eta''(0) when the data
  satisfy the equation; its etahat block is the face L2 projection of the mean of the
  one-sided traces of a, and its (sigma, gamma) blocks are lifted from (a, ahat) by the first
  two rows.
Difference from MATLAB: tau_S = tau mu, as in elasticity.py (tau = 1/mu gives MATLAB's 1).

VelocityStress: the stress-velocity-rotation form (no MATLAB counterpart), unknowns sigma,
v = eta', omega = gamma', vhat in the spaces of elasticity.py,

    rho v' + b0 eta - div sigma = f,   A sigma' = grad v - S(omega),   v = g' on Gamma,

with the HDG rows of elasticity.py for (A sigma', v, omega, vhat) and the flux
sigma_hat n = sigma n - tau_S (v - vhat), tau_S = tau sqrt(rho mu) (an impedance; it damps).
Time: Crank-Nicolson on every row (Meddahi 2025, (40)), solved for the midpoint values
y = (y^n + y^{n+1})/2 with the masses 2/dt (A sigma, tau) and (2 rho/dt + b0 dt/2) (v, w):

    (2/dt) (A sigma, tau) + [rows of tau] = (2/dt) (A sigma^n, tau),
    (2 rho/dt + b0 dt/2) (v, w) + [rows of w] = (f^{n+1/2}, w) + (2 rho/dt) (v^n, w) - b0 (eta^n, w),
    TPM: vhat + ((2/dt) A sigma + S(omega)).PathIntegral = g'^{n+1/2}(xbar) + (2/dt) (A sigma^n).PathIntegral,

data averaged over t^n, t^{n+1}; then y^{n+1} = 2 y - y^n for every dof, and the displacement
by the trapezoid rule eta^{n+1} = eta^n + dt v (eta^{n+1/2} = eta^n + dt/2 v in the b0 term).
Start (Meddahi (41)): L2 projections sigma^0 = Pi sigma(0) (onto the whole stress space
P_k^{2x2} + curl(B_K); its P_k part alone gives O(h^{k+1}/dt) in the first step, omega of order k),
v^0 = Pi eta'(0), omega^0 = Pi gamma'(0), vhat^0 = P_M eta'(0), eta^0 = Pi eta(0).
"""
from ngsolve import *

import elasticity as el
from elasticity import compliance, free_dofs, hdg_rows, postprocess, skew, spaces, stress
from elasticity_tpm import at, gamma_ds
from fsi_transient import facet_projection
from ngstpm import interior_ds, transfer_paths, unfitted_domain

BETA, GAMMA = 0.25, 0.5
t = Parameter(0.0)                                  # the time of the data


class Elastodynamics:
    """The forms of one run (mesh, k, dt fixed) and the state vectors U, V, A.

    mat: dict lam, mu, rho, b0; f: the load, g: the displacement on Gamma (vector CFs in x, y,
    t); phi: the level set of Omega (TPM) or None (fitted); D: the elements of D_h (TPM;
    default unfitted_domain); dirichlet: the fitted Dirichlet boundary of the mesh."""

    def __init__(self, mesh, order, mat, f, g, dt, phi=None, D=None, dirichlet=".*", tau=1.0, full=True):
        self.mesh, self.order, self.mat, self.f, self.g, self.dt = mesh, order, mat, f, g, dt
        self.phi, self.dirichlet, self.tau, self.full = phi, dirichlet, tau, full
        if phi is not None:
            D = unfitted_domain(mesh, phi)[0] if D is None else D
            self.paths = transfer_paths(mesh, phi, D, order=order + 4)
        self.D = D
        self.fes, self.top = spaces(mesh, order, dirichlet, D)
        self.trial, self.test = self.fes.TnT()
        self.dV = dx(definedonelements=D)
        lam, mu, rho, b0 = (mat[k] for k in ("lam", "mu", "rho", "b0"))

        self.lhs = self.bilinear(True, rho / (BETA * dt ** 2) + b0)
        self.load = self.linear(f, g)
        self.M = BilinearForm(self.fes)
        self.M += rho * self.trial[3] * self.test[3] * self.dV
        self.M.Assemble()
        self.free = free_dofs(self.fes, self.top, True)
        self.Ainv = self.factor(self.lhs)

        self.gfu = GridFunction(self.fes)
        self.U, self.V, self.A = (self.gfu.vec.CreateVector() for _ in range(3))
        for v in (self.U, self.V, self.A):
            v[:] = 0
        self.time, self.steps = 0.0, 0

    # ---- forms and linear algebra ------------------------------------------------------------
    def bilinear(self, condense, mass):
        """The static HDG rows with the equation of etahat (fitted, or transferred on Gamma_h)
        plus mass (eta, v)_{D_h}."""
        mesh, D, (lam, mu) = self.mesh, self.D, (self.mat["lam"], self.mat["mu"])
        (sP, w1, w2, eta, gam, etahat), muh = self.trial[:6], self.test[5]
        lhs = BilinearForm(self.fes, condense=condense)
        sigmahat_n = hdg_rows(lhs, LinearForm(self.fes), self.trial[:6], self.test[:6], lam, mu, CF((0, 0)),
                              self.tau, D)
        if self.phi is None:
            lhs += sigmahat_n * muh * dx(element_boundary=True)      # muh vanishes on Dirichlet faces
        else:
            lhs += sigmahat_n * muh * interior_ds(mesh, D)
            sigma = stress(sP, w1, w2) if self.full else sP
            lhs += etahat.Transfer(self.paths, gradient=compliance(sigma, lam, mu) + skew(gam)) * muh * self._dB()
        if mass:
            lhs += mass * eta * self.test[3] * self.dV
        return lhs

    def linear(self, f, g):
        """(f, v)_{D_h} and, TPM, <g(xbar), muh>_{Gamma_h}."""
        load = LinearForm(self.fes)
        load += f * self.test[3] * self.dV
        if self.phi is not None:
            load += at(g)(*self.paths.target) * self.test[5] * self._dB()
        return load

    def _dB(self):
        return gamma_ds(self.mesh, self.D, bonus_intorder=max(6, 2 * self.order))

    def factor(self, lhs):
        """The inverse of the condensed lhs, extended to all dofs."""
        lhs.Assemble()
        Id = IdentityMatrix(lhs.mat.height)
        Ainv = lhs.mat.Inverse(self.free, inverse="umfpack")
        return (Id + lhs.harmonic_extension) @ (Ainv + lhs.inner_solve) @ (Id + lhs.harmonic_extension_trans)

    def solve(self, lhs, Ainv, b, g, x):
        """x := the solution with etahat = P_M g on the fitted Dirichlet faces (b is overwritten)."""
        gf = GridFunction(self.fes)
        gf.vec[:] = 0
        gf.components[5].Set(g, BND, definedon=self.mesh.Boundaries(self.dirichlet))
        b.data -= lhs.mat * gf.vec
        gf.vec.data += lhs.harmonic_extension * gf.vec
        gf.vec.data += Ainv * b
        x.data = gf.vec

    # ---- time stepping -----------------------------------------------------------------------
    def step(self):
        """One Newmark step to t^{n+1}."""
        dt = self.dt
        t.Set(self.time + dt)
        self.load.Assemble()
        b, tmp = self.load.vec.CreateVector(), self.load.vec.CreateVector()
        tmp.data = (1 / (BETA * dt ** 2)) * self.U + (1 / (BETA * dt)) * self.V + ((0.5 - BETA) / BETA) * self.A
        b.data = self.load.vec + self.M.mat * tmp
        x = self.U.CreateVector()
        self.solve(self.lhs, self.Ainv, b, self.g, x)
        X, U, V, A = (v.FV().NumPy() for v in (x, self.U, self.V, self.A))
        a1 = (X - U - dt * V) / (BETA * dt ** 2) - ((0.5 - BETA) / BETA) * A
        V += dt * ((1 - GAMMA) * A + GAMMA * a1)
        A[:] = a1
        U[:] = X
        self.time += dt
        self.steps += 1

    def march(self, nsteps, monitor=None):
        """nsteps steps; monitor(self) after each step."""
        for _ in range(nsteps):
            self.step()
            if monitor is not None:
                monitor(self)

    # ---- initial state -----------------------------------------------------------------------
    def initial_state(self, eta0, eta1):
        """U, V := the HDG elliptic projections of eta0, eta1 (vector CFs in x, y, at t = 0);
        A := the consistent acceleration (load f at t = 0)."""
        t.Set(0.0)
        lam, mu, b0, rho = (self.mat[k] for k in ("lam", "mu", "b0", "rho"))
        static = self.bilinear(True, b0)
        Sinv = self.factor(static)
        for vec, w in ((self.U, eta0), (self.V, eta1)):
            load = self.linear(el.manufactured(w, lam, mu)[2] + b0 * w, w).Assemble()
            b = load.vec.CreateVector()
            b.data = load.vec
            self.solve(static, Sinv, b, w, vec)
        self.A.data = self.consistent_acceleration()
        self.time, self.steps = 0.0, 0

    def consistent_acceleration(self):
        """The acceleration of the state U at the current t: its eta block from the local
        momentum row, etahat the face mean of its traces, (sigma, gamma) lifted."""
        K = self.bilinear(False, self.mat["b0"])
        K.Assemble()
        load = self.linear(self.f, self.g).Assemble()
        r = load.vec.CreateVector()
        r.data = load.vec - K.mat * self.U
        gf = GridFunction(self.fes)
        gf.vec.data = self.M.mat.Inverse(self._block(3), inverse="sparsecholesky") * r
        c = gf.components
        facet_projection(c[5], c[3], self.D)
        free = self._block(0, 1, 2, 4)
        free = free_dofs(self.fes, self.top, False, free=free)
        r.data = -1 * (K.mat * gf.vec)
        gf.vec.data += K.mat.Inverse(free, inverse="umfpack") * r
        return gf.vec

    def _block(self, *components):
        """The free dofs (uncondensed) of the given components."""
        free = self.fes.FreeDofs()
        mask = BitArray(self.fes.ndof)
        mask.Clear()
        for c in components:
            for d in self.fes.Range(c):
                mask.Set(d)
        return free & mask

    # ---- fields and errors -------------------------------------------------------------------
    def fields(self, vec=None):
        """(sigma_h, eta_h, gamma_h, etahat_h, eta*_h) of vec (default U, the last solution)."""
        gf = GridFunction(self.fes)
        gf.vec.data = self.U if vec is None else vec
        c = gf.components
        sigma_h = stress(*c[:3])
        c[3].gfu = gf
        return sigma_h, c[3], c[4], c[5], postprocess(self.mesh, self.order, self.mat["lam"], self.mat["mu"],
                                                     sigma_h, c[3], c[4], self.D)

    def errors(self, eta):
        """elasticity.errors of U against eta at the current t, plus |eta' - v_h| (v_h the eta
        block of V)."""
        sigma, gamma, _ = el.manufactured(eta, self.mat["lam"], self.mat["mu"])
        e = el.errors(self.mesh, self.fields(), eta, sigma, gamma, self.D)
        gv = GridFunction(self.fes)
        gv.vec.data = self.V
        ev = eta.Diff(t) - gv.components[3]
        dV = dx(definedonelements=self.D, bonus_intorder=4)
        return e + (sqrt(Integrate(InnerProduct(ev, ev) * dV, self.mesh)),)


COLUMNS = el.COLUMNS + ("|v-vh|",)


def material(E=1.0, nu=0.3, rho=1.0, b0=0.0):
    lam, mu = el.lame(E, nu)
    return dict(lam=lam, mu=mu, rho=rho, b0=b0)


def manufactured(eta, mat):
    """The load f = rho eta'' + b0 eta - div sigma(eta) of a displacement eta (CF in x, y, t)."""
    _, _, f = el.manufactured(eta, mat["lam"], mat["mu"])
    return f + mat["rho"] * eta.Diff(t).Diff(t) + mat["b0"] * eta


def run(mesh, order, eta, mat, dt, nsteps, monitor=None, **kw):
    """Newmark march of nsteps from the MATLAB initial state for the exact displacement eta;
    returns the Elastodynamics object (its errors(eta) at the final time)."""
    r = Elastodynamics(mesh, order, mat, manufactured(eta, mat), eta, dt, **kw)
    r.initial_state(eta, eta.Diff(t))
    r.march(nsteps, monitor)
    return r


class VelocityStress(Elastodynamics):
    """The stress-velocity-rotation form by Crank-Nicolson: the state U = (sigma, v, omega,
    vhat) at t^n and the displacement eta^n in the v block of E. Arguments as Elastodynamics
    (g: the displacement on Gamma; the datum is g')."""

    def __init__(self, mesh, order, mat, f, g, dt, phi=None, D=None, dirichlet=".*", tau=1.0, full=True):
        self.mesh, self.order, self.mat, self.f, self.dt = mesh, order, mat, f, dt
        self.g = g.Diff(t)                                           # the velocity datum
        self.phi, self.dirichlet, self.tau, self.full = phi, dirichlet, tau, full
        if phi is not None:
            D = unfitted_domain(mesh, phi)[0] if D is None else D
            self.paths = transfer_paths(mesh, phi, D, order=order + 4)
        self.D = D
        self.fes, self.top = spaces(mesh, order, dirichlet, D)
        self.trial, self.test = self.fes.TnT()
        self.dV = dx(definedonelements=D)
        lam, mu, rho, b0 = (mat[k] for k in ("lam", "mu", "rho", "b0"))
        self.tauS = tau * (rho * mu) ** 0.5

        self.lhs = self.bilinear(2 / dt, 2 * rho / dt + b0 * dt / 2)
        self.load = self.linear(f, self.g)
        sP, w1, w2, v = self.trial[:4]
        sigma = stress(sP, w1, w2)
        self.H = BilinearForm(self.fes)                              # the old-state terms
        self.H += (2 / dt) * (InnerProduct(compliance(sigma, lam, mu), stress(*self.test[:3]))
                              + rho * v * self.test[3]) * self.dV
        if phi is not None:
            s = sigma if full else sP
            self.H += (2 / dt) * compliance(s, lam, mu).PathIntegral(self.paths) * self.test[5] * self._dB()
        self.B0 = BilinearForm(self.fes)
        self.B0 += b0 * v * self.test[3] * self.dV
        for a in (self.H, self.B0):
            a.Assemble()
        self.free = free_dofs(self.fes, self.top, True)
        self.Ainv = self.factor(self.lhs)

        self.gfu = GridFunction(self.fes)
        self.U, self.E, self.F = (self.gfu.vec.CreateVector() for _ in range(3))
        for v in (self.U, self.E, self.F):
            v[:] = 0
        self.time, self.steps = 0.0, 0

    def bilinear(self, cs, cm):
        """The HDG rows of (sigma, v, omega, vhat) with cs (A sigma, tau) and cm (v, w)."""
        mesh, D, (lam, mu) = self.mesh, self.D, (self.mat["lam"], self.mat["mu"])
        (sP, w1, w2, v, om, vhat), (tP, z1, z2, w, xi, muh) = self.trial[:6], self.test[:6]
        n = specialcf.normal(2)
        dV, dS = self.dV, dx(element_boundary=True, definedonelements=D)
        sigma, tau_ = stress(sP, w1, w2), stress(tP, z1, z2)
        sigmahat_n = sigma * n - self.tauS * (v - vhat)
        lhs = BilinearForm(self.fes, condense=True)
        lhs += (cs * InnerProduct(compliance(sigma, lam, mu), tau_) - InnerProduct(grad(v), tau_)
                + InnerProduct(skew(om), tau_)) * dV + (v - vhat) * (tau_ * n) * dS
        lhs += InnerProduct(sigma, skew(xi)) * dV
        lhs += cm * v * w * dV + InnerProduct(sigma, grad(w)) * dV - sigmahat_n * w * dS
        if self.phi is None:
            lhs += sigmahat_n * muh * dx(element_boundary=True)      # muh vanishes on Dirichlet faces
        else:
            lhs += sigmahat_n * muh * interior_ds(mesh, D)
            s = sigma if self.full else sP
            lhs += vhat.Transfer(self.paths, gradient=cs * compliance(s, lam, mu) + skew(om)) * muh * self._dB()
        return lhs

    def step(self):
        """One Crank-Nicolson step to t^{n+1}."""
        dt = self.dt
        t.Set(self.time + dt)
        self.load.Assemble()
        b = self.load.vec.CreateVector()
        b.data = 0.5 * (self.load.vec + self.F) + self.H.mat * self.U - self.B0.mat * self.E
        self.F.data = self.load.vec
        y = self.U.CreateVector()
        self.solve(self.lhs, self.Ainv, b, 0.5 * (self.g + self.g.Replace({t: t - dt})), y)
        self.U.data *= -1                  # U.data = 2 y - U would alias (U := 2 y first)
        self.U.data += 2 * y
        self.E.data += dt * y              # only the v block is read
        self.time += dt
        self.steps += 1

    def initial_state(self, eta):
        """Meddahi (41): L2 projections at t = 0 of the exact fields of the displacement eta."""
        t.Set(0.0)
        sigma, gamma, _ = el.manufactured(eta, self.mat["lam"], self.mat["mu"])
        gf = GridFunction(self.fes)
        gf.vec[:] = 0
        c = gf.components
        (sP, w1, w2), tests = self.trial[:3], stress(*self.test[:3])
        mass = BilinearForm(InnerProduct(stress(sP, w1, w2), tests) * self.dV).Assemble()
        load = LinearForm(InnerProduct(sigma, tests) * self.dV).Assemble()
        free = free_dofs(self.fes, self.top, False, free=self._block(0, 1, 2))
        gf.vec.data = mass.mat.Inverse(free, inverse="umfpack") * load.vec    # onto P_k + curl(B_K)
        c[3].Set(eta.Diff(t), definedonelements=self.D)
        c[4].Set(gamma.Diff(t), definedonelements=self.D)
        facet_projection(c[5], eta.Diff(t), self.D)
        self.U.data = gf.vec
        gf.vec[:] = 0
        c[3].Set(eta, definedonelements=self.D)
        self.E.data = gf.vec
        self.load.Assemble()
        self.F.data = self.load.vec
        self.time, self.steps = 0.0, 0

    def fields(self, vec=None):
        """(sigma_h, v_h, omega_h, vhat_h, eta_h) of vec (default U) and E."""
        gf, ge = GridFunction(self.fes), GridFunction(self.fes)
        gf.vec.data = self.U if vec is None else vec
        ge.vec.data = self.E
        c = gf.components
        c[3].gfu = gf
        return stress(*c[:3]), c[3], c[4], c[5], ge.components[3]

    def errors(self, eta):
        """L2(D_h) errors of sigma, v, omega, eta and |v - vhat|_h at the current t."""
        sigma, gamma, _ = el.manufactured(eta, self.mat["lam"], self.mat["mu"])
        sigma_h, v_h, om_h, vhat_h, eta_h = self.fields()
        v = eta.Diff(t)
        dV = dx(definedonelements=self.D, bonus_intorder=4)
        dS = dx(element_boundary=True, definedonelements=self.D, bonus_intorder=4)
        l2 = lambda a: sqrt(Integrate(InnerProduct(a, a) * dV, self.mesh))
        h = specialcf.mesh_size
        return (l2(sigma_h - sigma), l2(v_h - v), l2(om_h - gamma.Diff(t)), l2(eta_h - eta),
                sqrt(Integrate(h * InnerProduct(v - vhat_h, v - vhat_h) * dS, self.mesh)))

    def energy(self):
        """(1/2) ((A sigma, sigma) + rho |v|^2 + b0 |eta|^2) on D_h."""
        sigma_h, v_h, _, _, eta_h = self.fields()
        m = self.mat
        dV = dx(definedonelements=self.D, bonus_intorder=4)
        return 0.5 * Integrate((InnerProduct(compliance(sigma_h, m["lam"], m["mu"]), sigma_h)
                                + m["rho"] * v_h * v_h + m["b0"] * eta_h * eta_h) * dV, self.mesh)


VCOLUMNS = ("|s-sh|", "|v-vh|", "|w-wh|", "|e-eh|", "|v-vh^|h")


def run_velocity(mesh, order, eta, mat, dt, nsteps, monitor=None, **kw):
    """VelocityStress march of nsteps from the L2-projected start for the exact displacement eta."""
    r = VelocityStress(mesh, order, mat, manufactured(eta, mat), eta, dt, **kw)
    r.initial_state(eta)
    r.march(nsteps, monitor)
    return r


SMOOTH = CF((sin(pi * (x + t)) * cos(pi * y), cos(pi * x) * sin(pi * (y - t))))
CASES = ("fitted", "Example 2 (unfitted disk)", "Example 3 (kidney)")


def study(case, order, hs, T=0.5, mat=None, record=None, form="displacement"):
    """Space-time EOC table at t = T for SMOOTH, dt = T/ceil(T/h); form: "displacement"
    (Newmark, Elastodynamics) or "velocity" (Crank-Nicolson, VelocityStress)."""
    from math import ceil
    from netgen.geom2d import unit_square
    from elasticity import eoc_line
    from elasticity_tpm import EXAMPLES, domain

    mat = material() if mat is None else mat
    columns, march = (COLUMNS, run) if form == "displacement" else (VCOLUMNS, run_velocity)
    print(f"{case}, {form} form, k = {order}, T = {T}, {mat}")
    print(f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in columns) + "      dt")
    prev, rows = None, []
    for h in hs:
        nsteps = ceil(T / h)
        if case == "fitted":
            mesh, kw = Mesh(unit_square.GenerateMesh(maxh=h)), {}
        else:
            phi = EXAMPLES[case]
            mesh, D, dirichlet, _ = domain(case, h, phi)
            kw = dict(phi=phi, D=D, dirichlet=dirichlet)
        r = march(mesh, order, SMOOTH, mat, T / nsteps, nsteps, **kw)
        e = r.errors(SMOOTH)
        print(eoc_line(h, e, prev) + f"  {T / nsteps:6.4f}")
        if record is not None:
            record.add(h, mesh, e, r.fields()[1].gfu, dt=T / nsteps)
        prev = (h, e)
        rows.append(e)
    return rows


if __name__ == "__main__":
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(CASES))
    p.add_argument("--T", type=float, default=0.5, help="final time (default 0.5)")
    p.add_argument("--form", choices=("displacement", "velocity"), default="displacement",
                   help="displacement (Newmark, default) or velocity (Crank-Nicolson)")
    a = parse(p)
    if a.k is None:
        run_defaults(__file__, [["--case", c, "--form", f, "--k", k, "--h", *hs] for c in CASES
                                for f in ("displacement", "velocity")
                                for k, hs in ((1, (0.2, 0.1, 0.05)), (2, (0.2, 0.1, 0.05)))])
    else:
        record = Record(COLUMNS if a.form == "displacement" else VCOLUMNS)
        study(a.case, a.k, a.h, T=a.T, record=record, form=a.form)
        if a.out:
            save(a.out, record, dict(driver="elastodynamics", case=a.case, form=a.form, k=a.k, T=a.T, **material()))

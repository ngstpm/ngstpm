"""Stokes flow in stress-velocity-rotation form, mixed HDG with weakly symmetric stress,
2D, fitted mesh: the solid of elasticity.py in the limit lam = infinity.

Find the Cauchy stress sigma, velocity u, rotation gamma (scalar) and trace uhat with

    A sigma - grad u + S(gamma) = 0,   -div sigma = f   in Omega,
    u = g on Gamma_D,   sigma n = gN on Gamma_N,

A sigma = dev(sigma)/(2 mu) = sigma/(2 mu) - tr(sigma) I/(4 mu), pressure p = -tr(sigma)/2.
A is singular on I: the element mean of tr(sigma) is kept as the global unknown rho
(the split=True rows of elasticity.hdg_rows with 1/(2 (lam + mu)) = 0), and, without
a traction boundary, a multiplier m fixes the mean of rho
(IncompressibleElasticityMatrices.m, pressure_mean = "constrained"):

    (A sigma, tau) - (grad u, tau) + <u - uhat, tau n> + (S(gamma), tau) + (xi, tr tau) = 0
    (sigma, S(eta)) = 0
    (sigma, grad v) - <sigma_hat n, v> = (f, v),   sigma_hat n = sigma n - tau mu (u - uhat)
    (tr sigma - rho, chi)_K = 0                       xi, chi in P_0(K), local
    <uhat . n, psi>_dK + (m, psi)_K = 0               rho, psi in P_0(K), global
    (rho, 1) = 0                                      m in R (only if Gamma_N is empty)
    <sigma_hat n, mu> = <gN, mu>_{Gamma_N},  uhat = P_M g on Gamma_D.

Spaces as in elasticity.py. On a compatible problem (int g . n = 0) m_h = 0.

Rule for the pressure mean: the zero-mean constraint with the global multiplier m is
used only when the mode p + c is undetermined, i.e. the velocity is prescribed on the
whole boundary; with a traction condition anywhere there is no constraint and no
multiplier ("pinned" is not used).
"""
from math import log2

from ngsolve import *

from elasticity import free_dofs, hdg_rows, postprocess, solve_linear_system, spaces, stress

INF = float("inf")


def solve(mesh, order, mu, f, g, gN=None, dirichlet=".*", tau=1.0):
    """Returns (sigma_h, u_h, gamma_h, uhat_h, ustar_h), p_h = -tr(sigma_h)/2 and m_h
    (None with a traction boundary)."""
    constrained = gN is None
    fes, top, lhs, rhs, trial, test, sigmahat_n = forms(mesh, order, mu, f, dirichlet, tau,
                                                        constrained=constrained)
    muh = test[5]
    lhs += sigmahat_n * muh * dx(element_boundary=True)
    if gN is not None:      # muh vanishes on the Dirichlet faces
        rhs += gN * muh * ds(skeleton=True)
    if constrained:
        add_constraint(lhs, trial, test)
    return finish(lhs, rhs, fes, top, (g, BND, mesh.Boundaries(dirichlet)), mesh, order, mu,
                  constrained=constrained)


def forms(mesh, order, mu, f, dirichlet=".*", tau=1.0, D=None, constrained=True):
    """Space (with the NumberSpace of m if constrained), forms and the rows on the elements
    of D except the equation of uhat and the constraint (see add_constraint). Returns
    fes, top, lhs, rhs, trial, test, sigma_hat n."""
    fes, top = spaces(mesh, order, dirichlet, D, split=True)
    if constrained:
        fes = fes * NumberSpace(mesh)
    trial, test = fes.TnT()
    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    sigmahat_n = hdg_rows(lhs, rhs, trial[:8], test[:8], INF, mu, f, tau, D)
    return fes, top, lhs, rhs, trial, test, sigmahat_n


def add_constraint(lhs, trial, test, D=None):
    """(m, psi) in the row of rho and (rho, 1) = 0."""
    (rho, m), (psi, q) = trial[7:], test[7:]
    lhs += (m * psi + rho * q) * dx(definedonelements=D)


def finish(lhs, rhs, fes, top, uhat_bc, mesh, order, mu, D=None, constrained=True):
    """Solve with uhat = uhat_bc on the Dirichlet faces; returns (sigma_h, u_h, gamma_h,
    uhat_h, ustar_h), p_h and m_h (None if not constrained)."""
    gfu = GridFunction(fes)
    gfu.components[5].Set(*uhat_bc)
    inverse = multiplier_inverse(fes, 7, 8) if constrained else None
    solve_linear_system(lhs, rhs, gfu, free_dofs(fes, top, True), inverse)
    sP_h, w1_h, w2_h, u_h, gam_h, uhat_h = gfu.components[:6]
    u_h.gfu = gfu                                       # the whole solution, for study_io
    sigma_h = stress(sP_h, w1_h, w2_h)
    ustar_h = postprocess(mesh, order, INF, mu, sigma_h, u_h, gam_h, D)
    m_h = gfu.components[8].vec[0] if constrained else None
    return (sigma_h, u_h, gam_h, uhat_h, ustar_h), -Trace(sigma_h) / 2, m_h


def multiplier_inverse(fes, rho, m):
    """inverse(mat, free) for solve_linear_system when the zero-mean multiplier m (first dof
    of component m) is in the system: the sparse factorization is of the condensed matrix
    without m and without one element mean of tr(sigma) (first free dof of component rho,
    which pins the constant pressure mode); these two unknowns are recovered from their
    2 x 2 Schur complement (BorderedInverse). The dense row and column of m then never enter
    the factorization; the solution is that of the whole system."""
    def inverse(mat, free):
        rho0 = next(d for d in range(fes.Range(rho).start, fes.Range(rho).stop) if free[d])
        removed = (fes.Range(m).start, rho0)
        reduced = BitArray(free)
        for d in removed:
            reduced[d] = False
        return BorderedInverse(mat, mat.Inverse(reduced), removed)
    return inverse


class BorderedInverse(BaseMatrix):
    """Inverse of mat on the free dofs from Ainv, the inverse of mat without the dofs
    removed, and the Schur complement of those (after examples/stokes.py of the ngstpm-stokes
    worktree, without its border vector)."""

    def __init__(self, mat, Ainv, removed):
        super().__init__()
        self.mat, self.Ainv, self.removed = mat, Ainv, removed
        e = mat.CreateColVector()
        self.w = []
        self.S = Matrix(len(removed), len(removed))                 # Schur complement
        for j, d in enumerate(removed):
            e[:] = 0
            e[d] = 1
            col = (mat * e).Evaluate()
            Z = self.rows(e)
            self.w.append((Ainv * col).Evaluate())
            for i, r in enumerate(self.rows(self.w[-1])):
                self.S[i, j] = Z[i] - r

    def rows(self, v):
        """The rows of the removed unknowns applied to v."""
        Av = (self.mat * v).Evaluate()
        return [Av[d] for d in self.removed]

    def Height(self):
        return self.mat.height

    def Width(self):
        return self.mat.width

    def CreateColVector(self):
        return self.mat.CreateColVector()

    def CreateRowVector(self):
        return self.mat.CreateRowVector()

    def Mult(self, b, x):
        x.data = self.Ainv * b
        xr = self.S.I * Vector([b[d] - r for d, r in zip(self.removed, self.rows(x))])
        for w, xi in zip(self.w, xr):
            x.data -= xi * w
        for d, xi in zip(self.removed, xr):
            x[d] = xi


def manufactured(u, p, mu):
    """sigma = 2 mu eps(u) - p I, gamma = (d_y u_1 - d_x u_2)/2 and f = -div sigma."""
    gu = CF((u[0].Diff(x), u[0].Diff(y), u[1].Diff(x), u[1].Diff(y)), dims=(2, 2))
    sigma = mu * (gu + gu.trans) - p * Id(2)
    gamma = 0.5 * (u[0].Diff(y) - u[1].Diff(x))
    f = -CF((sigma[0, 0].Diff(x) + sigma[0, 1].Diff(y), sigma[1, 0].Diff(x) + sigma[1, 1].Diff(y)))
    return sigma, gamma, f


def errors(mesh, sol, p_h, u, p, sigma, gamma, D=None):
    """|sigma - sigma_h|, |p - p_h|, |u - u_h|, |gamma - gamma_h|, |P_M u - uhat|_h,
    |u - u*|, |div u_h| (broken), on the elements of D."""
    import elasticity
    e = elasticity.errors(mesh, sol, u, sigma, gamma, D)
    dV = dx(definedonelements=D, bonus_intorder=4)
    ep = sqrt(Integrate((p_h - p) ** 2 * dV, mesh))
    div = sqrt(Integrate(div_h(sol[1]) ** 2 * dV, mesh))
    return (e[0], ep, e[1], e[2], e[4], e[5], div)


def div_h(u_h):
    return grad(u_h)[0, 0] + grad(u_h)[1, 1]


COLUMNS = ("|s-sh|", "|p-ph|", "|u-uh|", "|g-gh|", "|Pu-uh^|h", "|u-u*|", "|div uh|")
HEADER = f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS) + f"  {'m_h':>10}"
U = CF((sin(pi * x) * cos(pi * y), -cos(pi * x) * sin(pi * y)))        # curl of sin(pi x) sin(pi y)/pi
P = sin(pi * x) * cos(pi * y)                                          # zero mean on the unit square


def study(meshes, order, mu=1.0, dirichlet=".*", traction=False, u=U, p=P, record=None):
    """Print the EOC table on a list of (h, mesh); record: a study_io.Record that collects
    the levels."""
    sigma, gamma, f = manufactured(u, p, mu)
    gN = sigma * specialcf.normal(2) if traction else None
    print(f"k = {order}, mu = {mu}, dirichlet = {dirichlet!r}")
    print(HEADER)
    prev, rows = None, []
    for h, mesh in meshes:
        sol, p_h, m_h = solve(mesh, order, mu, f, u, gN, dirichlet)
        e = errors(mesh, sol, p_h, u, p, sigma, gamma)
        line = f"  {h:7.4f}"
        for i, ei in enumerate(e):
            r = f"{log2(prev[1][i] / ei) / log2(prev[0] / h):4.2f}" if prev and ei > 0 else "  - "
            line += f"  {ei:9.3e}  {r}"
        print(line + ("" if m_h is None else f"  {m_h:10.2e}"))
        if record is not None:
            record.add(h, mesh, e, sol[1].gfu, m_h=m_h)
        prev = (h, e)
        rows.append((e, m_h))
    return rows


CASES = {"dirichlet": {}, "traction": dict(dirichlet="left|bottom", traction=True)}


if __name__ == "__main__":
    from netgen.geom2d import unit_square
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(CASES))
    p.add_argument("--mu", type=float, default=1.0, help="viscosity (default 1)")
    a = parse(p)
    if a.k is None:
        # one process per case, within 2 GB each (Dirichlet: k = 1 at h = 0.0125 about 5 GB,
        # k = 2 at h = 0.025 about 1.8 GB: not defaults)
        run_defaults(__file__, [["--case", c, "--k", k, "--h", *hs, "--mu", mu]
                                for k, hs in ((1, (0.2, 0.1, 0.05, 0.025)), (2, (0.2, 0.1, 0.05)))
                                for mu in (1.0, 1e-3) for c in CASES])
    else:
        record = Record(COLUMNS)
        meshes = [(h, Mesh(unit_square.GenerateMesh(maxh=h))) for h in a.h]
        study(meshes, a.k, a.mu, record=record, **CASES[a.case])
        if a.out:
            save(a.out, record, dict(driver="stokes_stress", case=a.case, k=a.k, mu=a.mu, tau=1.0))

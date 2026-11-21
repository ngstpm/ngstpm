from math import log2

from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh

from poisson import solve_linear_system


def solve(mesh, order, f, g, nu=1.0, tau=1.0, condense=True):
    d = mesh.dim
    n = specialcf.normal(d)
    dS = dx(element_boundary=True)

    S = MatrixValued(L2(mesh, order=order))
    V = VectorL2(mesh, order=order)
    W = L2(mesh, order=order, lowest_order_wb=True)  # element means of p stay global
    M = FacetFESpace(mesh, order=order, dirichlet=".*") ** d
    R = NumberSpace(mesh)
    fes = S * V * W * M * R

    (L, u, p, uhat, rho), (G, v, q, mu, rho_) = fes.TnT()
    div_G = fem.Einsum("ijj->i", grad(G))  # row-wise: (div G)_i = d_j G_ij
    sigmahat_n = nu * L * n - p * n - tau * nu * (u - uhat)

    lhs = BilinearForm(fes, condense=condense)
    rhs = LinearForm(fes)

    lhs += InnerProduct(L, G) * dx + u * div_G * dx - uhat * (G * n) * dS

    lhs += nu * InnerProduct(L, grad(v)) * dx - p * Trace(grad(v)) * dx - sigmahat_n * v * dS
    rhs += f * v * dx

    lhs += -u * grad(q) * dx + (uhat * n) * q * dS

    lhs += sigmahat_n * mu * dS

    lhs += p * rho_ * dx + rho * q * dx  # (p, 1) = 0

    gfu = GridFunction(fes)
    gfu.components[3].Set(g, definedon=mesh.Boundaries(".*"))

    solve_linear_system(lhs, rhs, gfu, inverse=multiplier_inverse(fes))
    return gfu.components


def multiplier_inverse(fes, border=None):
    """Inverse of the (condensed) saddle point matrix with the zero-mean multiplier.

    UMFPACK on the whole matrix is slow and memory-hungry: the multiplier row
    and column are dense. Instead, UMFPACK factors the matrix without the
    multiplier and without one element mean of p (this pins the constant
    pressure), and the removed unknowns are recovered from their small Schur
    complement. fes: a product space with the pressure (element means on the
    wirebasket) as component 2 and the multiplier as component 4, or, with
    border = c, without multiplier: the matrix is then bordered, [[A, c], [c^T, 0]],
    and the multiplier is left in inverse.op.rho after the solve.
    """
    def inverse(mat, free):
        p0 = next(d for d in range(fes.Range(2).start, fes.Range(2).stop)
                  if free[d] and fes.CouplingType(d) == COUPLING_TYPE.WIREBASKET_DOF)
        removed = (p0,) if border is not None else (fes.Range(4).start, p0)
        reduced = BitArray(free)
        for d in removed:
            reduced[d] = False
        inverse.op = BorderedInverse(mat, mat.Inverse(reduced, inverse="umfpack"), removed, border)
        return inverse.op
    return inverse


class BorderedInverse(BaseMatrix):
    """Inverse of mat on the free dofs (bordered by c with a multiplier rho if c is
    given), from Ainv, the inverse of mat without the dofs removed."""

    def __init__(self, mat, Ainv, removed, c=None):
        super().__init__()
        self.mat, self.Ainv, self.removed, self.c, self.rho = mat, Ainv, removed, c, None
        e = mat.CreateColVector()
        cols, Z = [], []                    # columns of the removed unknowns; their block of the matrix
        for d in removed:
            e[:] = 0
            e[d] = 1
            cols.append((mat * e).Evaluate())
            Z.append(self.rows(e))
        if c is not None:
            cols.append(c)
            Z.append([c[d] for d in removed] + [0])
        self.w = [(Ainv * col).Evaluate() for col in cols]
        m = len(cols)
        self.S = Matrix(m, m)                                       # Schur complement
        for j, w in enumerate(self.w):
            for i, r in enumerate(self.rows(w)):
                self.S[i, j] = Z[j][i] - r

    def rows(self, v):
        """The rows of the removed unknowns (then of the multiplier, if bordered) applied to v."""
        Av = (self.mat * v).Evaluate()
        return [Av[d] for d in self.removed] + ([InnerProduct(self.c, v)] if self.c is not None else [])

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
        rhs = [b[d] for d in self.removed] + ([0] if self.c is not None else [])
        xr = self.S.I * Vector([bi - r for bi, r in zip(rhs, self.rows(x))])
        for w, xi in zip(self.w, xr):
            x.data -= xi * w
        for d, xi in zip(self.removed, xr):
            x[d] = xi
        if self.c is not None:
            self.rho = xr[-1]


def stokes_data(u, p, nu):
    """L = grad u and f = -div(nu L - p I) for a 2D velocity u and pressure p."""
    L = CF((u.Diff(x), u.Diff(y)), dims=(2, 2)).trans
    sigma = nu * L - p * Id(2)
    f = -CF(tuple(sigma[i, 0].Diff(x) + sigma[i, 1].Diff(y) for i in range(2)))
    return L, f


if __name__ == "__main__":
    uex = CF((sin(pi * x) * sin(pi * y), cos(pi * x) * cos(pi * y)))
    pex = sin(2 * pi * x) * sin(2 * pi * y)

    for nu in (1.0, 1e-6):
        print(f"nu = {nu:g}")
        Lex, f = stokes_data(uex, pex, nu)
        for k in (1, 2, 3):
            print(f"  k = {k}")
            prev = None
            for N in (4, 8, 16, 32):
                mesh = MakeStructured2DMesh(quads=False, nx=N, ny=N)
                Lh, uh, ph, *_ = solve(mesh, k, f, uex, nu=nu)
                eL = sqrt(Integrate(InnerProduct(Lh - Lex, Lh - Lex), mesh, order=2 * k + 6))
                eu = sqrt(Integrate((uh - uex) ** 2, mesh, order=2 * k + 6))
                ep = sqrt(Integrate((ph - pex) ** 2, mesh, order=2 * k + 6))
                if prev:
                    rL, ru, rp = log2(prev[0] / eL), log2(prev[1] / eu), log2(prev[2] / ep)
                    print(f"    N={N:3d}  |L-Lh|={eL:.3e} ({rL:.2f})  |u-uh|={eu:.3e} ({ru:.2f})"
                          f"  |p-ph|={ep:.3e} ({rp:.2f})")
                else:
                    print(f"    N={N:3d}  |L-Lh|={eL:.3e}         |u-uh|={eu:.3e}"
                          f"         |p-ph|={ep:.3e}")
                prev = (eL, eu, ep)

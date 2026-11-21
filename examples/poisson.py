from math import log2

from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh

def solve_linear_system(lhs, rhs, gfu, inverse=""):
    lhs.Assemble()
    free_dofs = gfu.space.FreeDofs(coupling=lhs.condense)
    Ainv = lhs.mat.Inverse(free_dofs, inverse=inverse)

    rhs.Assemble()
    b = rhs.vec.CreateVector()
    b.data = rhs.vec - lhs.mat * gfu.vec
    if lhs.condense:
        Id = IdentityMatrix(lhs.mat.height)
        A_IS = lhs.harmonic_extension # skeleton -> interior
        A_SI = lhs.harmonic_extension_trans # interior -> skeleton
        A_II = lhs.inner_solve # interior -> interior
        Ainv_full = (Id + A_IS) @ (Ainv + A_II) @ (Id + A_SI)

        gfu.vec.data += A_IS * gfu.vec
    else:
        Ainv_full = Ainv
    gfu.vec.data += Ainv_full * b

def solve(mesh, order, f, g_dirichlet, tau=1.0):
    n = specialcf.normal(mesh.dim)
    dS = dx(element_boundary=True)

    V = VectorL2(mesh, order=order)
    W = L2(mesh, order=order)
    M = FacetFESpace(mesh, order=order, dirichlet=".*")
    fes = V * W * M

    (q, u, uhat), (r, v, mu) = fes.TnT()
    qhat_n = q*n + tau*(u - uhat)

    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)

    lhs += q*r*dx - u*div(r)*dx + uhat*(r*n)*dS
    # rhs += 0 * r * dx

    lhs += -q*grad(v)*dx + qhat_n*v*dS
    rhs += f * v * dx

    lhs += qhat_n * mu * dS
    # lhs += 0 * mu * dS

    gfu = GridFunction(fes)
    gfu.components[2].Set(g_dirichlet, definedon=mesh.Boundaries(".*"))

    solve_linear_system(lhs, rhs, gfu)
    return gfu.components


if __name__ == "__main__":
    uex = sin(pi * x) * cos(pi * y)
    qex = CF((-pi * cos(pi * x) * cos(pi * y), pi * sin(pi * x) * sin(pi * y)))
    f = 2 * pi**2 * uex

    for k in (1, 2, 3):
        print(f"k = {k}")
        prev = None
        for N in (4, 8, 16, 32):
            mesh = MakeStructured2DMesh(quads=False, nx=N, ny=N)
            qh, uh, _ = solve(mesh, k, f, uex)
            eu = sqrt(Integrate((uh - uex) ** 2 * dx(bonus_intorder=6), mesh))
            eq = sqrt(Integrate((qh - qex) ** 2 * dx(bonus_intorder=6), mesh))
            if prev:
                ru, rq = log2(prev[0] / eu), log2(prev[1] / eq)
                print(f"  N={N:3d}  |u-uh|={eu:.3e} ({ru:.2f})  |q-qh|={eq:.3e} ({rq:.2f})")
            else:
                print(f"  N={N:3d}  |u-uh|={eu:.3e}         |q-qh|={eq:.3e}")
            prev = (eu, eq)

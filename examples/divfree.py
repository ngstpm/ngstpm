"""Divergence-free H(div) postprocessing of the HDG velocity of the stress-form Stokes and
Oseen solvers: the port of MATLAB postprocess_velocity.m (Cockburn, Gopalakrishnan,
Nguyen, Peraire, Sayas 2011, (2.9), 2D). On each K, u** in P_{k+1}(K)^2 with

    <(u** - uhat) . n, mu>_F = 0                         mu in P_k(F)
    <d_t(u** . n) - t . ({L}^T n), d_t l_F>_F = 0         l_F the degree-(k+1) Legendre mode
    (u** - u_h, grad w)_K = 0                             w in P_k(K) / R
    (curl u** - (L_21 - L_12), b)_K = 0                   b in b_K P_{k-1}(K)

with L = P_k(A sigma_h + S(gamma_h)) the elementwise L2 projection of the velocity gradient
(gradient_from_stress.m), {L} its average from both sides of F (own side only on the
boundary of D_h), t = (-n_2, n_1), curl u = d_x u_2 - d_y u_1 and b_K the cubic bubble
(b_K P_{k-1} = the interior bubbles of H1(k+2)). The first two rows fix u** . n on F from
single-valued data, so u** is H(div)-conforming, and returned in HDiv(k+1).
The element problems are solved together as one block-diagonal system written in saddle
form [[0, B^T], [B, 0]], B the rows above (no symmetry needed; the multiplier is zero).
"""
import numpy as np
from ngsolve import *

from elasticity import compliance, skew
from ngstpm import interior_ds, restrict

INF = float("inf")


def face_coordinate(mesh):
    """s in [-1, 1] along each edge (orientation of the element normal), as a CF on
    element boundaries: s = 2 (x - m_F) . t / |F|."""
    F0 = FacetFESpace(mesh, order=0)
    mx, my, ln = (GridFunction(F0) for _ in range(3))
    for e in mesh.edges:
        a, b = (np.array(mesh.vertices[v.nr].point) for v in e.vertices)
        d = F0.GetDofNrs(e)[0]
        mx.vec[d], my.vec[d] = (a + b) / 2
        ln.vec[d] = np.linalg.norm(b - a)
    n = specialcf.normal(2)
    return 2 * ((x - mx) * (-n[1]) + (y - my) * n[0]) / ln


def divergence_free(mesh, order, mu, sigma_h, u_h, gamma_h, uhat_h, D=None):
    """u** (HDiv(k+1) GridFunction on the elements of D) from the HDG solution."""
    on = (lambda V: V) if D is None else (lambda V: restrict(V, D))
    k = order
    n = specialcf.normal(2)
    t = CF((-n[1], n[0]))
    dV = dx(definedonelements=D)
    dS = dx(element_boundary=True, definedonelements=D)

    L = GridFunction(on(MatrixValued(L2(mesh, order=k), dim=2)))
    L.Set(compliance(sigma_h, INF, mu) + skew(gamma_h), definedonelements=D)
    Fk = on(FacetFESpace(mesh, order=k))             # t . {L}^T n, single valued
    p, q = Fk.TnT()
    tLn = GridFunction(Fk)
    mass = BilinearForm(p * q * dS).Assemble()
    tLn.vec.data = mass.mat.Inverse(Fk.FreeDofs(), inverse="umfpack") * \
        LinearForm(n * (L * t) * q * dS).Assemble().vec

    s = face_coordinate(mesh)                        # d_t l_F = l_{k+1}'(s), up to a factor
    c = np.polynomial.legendre.leg2poly(np.polynomial.legendre.legder([0] * (k + 1) + [1]))
    dl = sum(float(ci) * s ** i for i, ci in enumerate(c))

    Z = on(H1(mesh, order=k + 2))
    fes = (on(VectorL2(mesh, order=k + 1)) * on(Discontinuous(FacetFESpace(mesh, order=k)))
           * on(Discontinuous(FacetFESpace(mesh, order=0))) * on(L2(mesh, order=k)) * Z)
    (us, la, lb, lw, lz), (vs, ma, mb, mw, mz) = fes.TnT()

    def rows(u, a, b, w, z):
        J = grad(u)
        return ((u * n) * a * dS + b * (n * (J * t)) * dl * dS
                + u * grad(w) * dV + (J[1, 0] - J[0, 1]) * z * dV)
    lhs = BilinearForm(rows(us, ma, mb, mw, mz) + rows(vs, la, lb, lw, lz))
    rhs = LinearForm((uhat_h * n) * ma * dS + mb * tLn * dl * dS + u_h * grad(mw) * dV
                     + (L[1, 0] - L[0, 1]) * mz * dV)
    free = fes.FreeDofs()
    W, Zr = fes.Range(3), fes.Range(4)
    for el in mesh.Elements(VOL):                    # w in P_k / R: no constant
        if D is None or D[el.nr]:
            free.Clear(W.start + fes.components[3].GetDofNrs(el)[0])
    for i in range(Z.ndof):                          # b in b_K P_{k-1}: interior bubbles only
        if Z.CouplingType(i) != COUPLING_TYPE.LOCAL_DOF:
            free.Clear(Zr.start + i)
    lhs.Assemble()
    rhs.Assemble()
    sol = GridFunction(fes)
    sol.vec.data = lhs.mat.Inverse(free, inverse="umfpack") * rhs.vec
    ustar = GridFunction(on(HDiv(mesh, order=k + 1)))
    ustar.Set(sol.components[0], dual=True, definedonelements=D)
    ustar.l2 = sol.components[0]                     # the element solution, for checks
    return ustar


def checks(mesh, ustar, D=None):
    """max_K |div u**|_{L2(K)}, |div u**|_{L2(D_h)}, the normal jump of the element solution
    (sum over the interior faces of D_h of |[u . n]|_F^2)^(1/2), and |u** - element solution|
    (zero iff the element solution is H(div)-conforming)."""
    dV = dx(definedonelements=D, bonus_intorder=2)
    d2 = Integrate(div(ustar) ** 2 * dV, mesh, element_wise=True).NumPy()
    u, n = ustar.l2, specialcf.normal(2)
    F = FacetFESpace(mesh, order=ustar.space.globalorder)     # [u . n] in P_{k+1}(F): exact
    p, q = F.TnT()
    dF = interior_ds(mesh, D)                                 # each face from both sides
    M = BilinearForm(p * q * dF).Assemble().mat               # twice the face mass
    r = LinearForm(u * n * q * dF).Assemble().vec             # <[u . n], q>_F
    free = BitArray([M[i, i] > 0 for i in range(F.ndof)])
    Mr = r.CreateVector()
    Mr.data = M.Inverse(free, inverse="sparsecholesky") * r
    j = sqrt(2 * InnerProduct(r, Mr))
    diff = sqrt(Integrate(InnerProduct(ustar - u, ustar - u) * dV, mesh))
    return np.sqrt(d2.max()), np.sqrt(d2.sum()), j, diff

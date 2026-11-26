"""Linear elasticity, mixed HDG with weakly symmetric stress, 2D, fitted mesh.

Find the stress sigma, displacement u, rotation gamma (scalar) and trace uhat with

    A sigma - grad u + S(gamma) = 0,   -div sigma = f   in Omega,
    u = g on Gamma_D,   sigma n = g_N on Gamma_N,

where A sigma = sigma/(2 mu) - lam/(4 mu (lam + mu)) tr(sigma) I (plane strain) and
S(eta) = [[0, eta], [-eta, 0]]. The HDG method (Cockburn-Shi 2013) reads

    (A sigma, tau) - (grad u, tau) + <u - uhat, tau n> + (S(gamma), tau) = 0
    (sigma, S(eta)) = 0
    (sigma, grad v) - <sigma_hat n, v> = (f, v)
    <sigma_hat n, mu> = <g_N, mu>_{Gamma_N}
    sigma_hat n = sigma n - tau_S (u - uhat),   uhat = P_M g on Gamma_D,
    tau_S = tau mu (tau an O(1) constant, so that tau_S scales as the stress),

where the first row is (u, div tau) - <uhat, tau n> integrated by parts. Spaces, k >= 1:
u in P_k^2, gamma in P_k, uhat in P_k(e)^2, and sigma in P_k^{2x2} + B(K) with the PEERS
enrichment of the preprint (numerical-method.tex, from Cockburn-Shi 2013) and MATLAB
(PEERSBasis, curl_curl_A_dubiner_bubble_eval):

    B(K) = curl(curl(A(K)) b_K) = { curl(b_K grad p) : p in P_k(K) },

A(K) the skew P_k tensors, b_K the cubic bubble. Only the k + 1 modes of p of exact degree
k are new (lower p give curls in P_k^{2x2} and would make the local problems singular), so
sigma = sigma_P + curl(b_K grad p) with p in the top-degree span of L2(k). The rows of
curl(v) are (d_y v_i, -d_x v_i); d_j(b p_i) = b_j p_i + b p_ij.
"""
from math import log2

from ngsolve import *

from ngstpm import OwnElements, SideEval, restrict


def skew(eta):
    return CF((0, eta, -eta, 0), dims=(2, 2))


def compliance(sigma, lam, mu):
    """Plane-strain compliance; lam = inf gives the incompressible limit dev(sigma)/(2 mu)."""
    c2 = 1 / (4 * mu) if lam == float("inf") else lam / (4 * mu * (lam + mu))
    return sigma / (2 * mu) - c2 * Trace(sigma) * Id(2)


def stress(sP, p, _=None, at=None):
    """sigma = sigma_P + curl(b_K grad p); with at: extrapolated from the element of x (of
    the side of p's space under an interface measure) to at. The third argument is the
    empty component of spaces, kept so that the component layout is unchanged."""
    mesh = p.space.mesh
    b = _bubble(mesh)
    gp, H = grad(p), p.Operator("hesse")
    if at is None:
        b, gb = 1.0 * b, grad(b)
    else:
        D = OwnElements(p)
        b, gb = SideEval(b, at, mesh, D), SideEval(grad(b), at, mesh, D)
        sP, gp, H = sP.Extrapolate(at), gp.Extrapolate(at), H.Extrapolate(at)
    d = lambda i, j: gb[j] * gp[i] + b * H[i, j]          # d_j (b p_i)
    return sP + CF((d(0, 1), -d(0, 0), d(1, 1), -d(1, 0)), dims=(2, 2))


_bubbles = {}


def _bubble(mesh):
    """b_K on every element: the H1(3) GridFunction with its interior dofs = 1."""
    if id(mesh) not in _bubbles:
        V = H1(mesh, order=3)
        b = GridFunction(V)
        for d in range(V.ndof):
            b.vec[d] = 1.0 if V.CouplingType(d) == COUPLING_TYPE.LOCAL_DOF else 0.0
        _bubbles[id(mesh)] = (mesh, b)
    return _bubbles[id(mesh)][1]


def enrichment(mesh, order, D=None):
    """L2(order) on the elements of D (default: all) and the BitArray of its dofs of exact
    degree order (k + 1 per element; the local L2 basis is ordered (i, j), i + j <= k,
    j fastest). The other dofs are marked INTERFACE_DOF (not eliminated locally) and are
    kept at zero by free_dofs; all dofs of an element stay regular, as Extrapolate requires."""
    Z = L2(mesh, order=order)
    if D is not None:
        Z = restrict(Z, D)
    idx = [n for n, (i, j) in enumerate((i, j) for i in range(order + 1) for j in range(order + 1 - i))
           if i + j == order]
    top = BitArray(Z.ndof)
    top.Clear()
    for el in mesh.Elements(VOL):
        if D is None or D[el.nr]:
            dofs = Z.GetDofNrs(el)
            for i in idx:
                top.Set(dofs[i])
    for d in range(Z.ndof):
        if not top[d]:
            Z.SetCouplingType(d, COUPLING_TYPE.INTERFACE_DOF)
    return Z, top


def spaces(mesh, order, dirichlet=".*", D=None, split=False):
    """The product space (sigma_P, p, -, u, gamma, uhat) on the elements of D, and the
    BitArray of the p dofs in use (see enrichment). "-" is an empty space of order k + 2: it
    keeps the component layout and the default quadrature of the former bubble space
    H1(k + 2) (NGSolve takes the rule from the element orders, and curl(b_K grad p) has
    degree k + 1; with order k it is under-integrated). With split, two P_0 spaces are
    appended: xi (local) and rho (global), see forms."""
    on = (lambda V: V) if D is None else (lambda V: restrict(V, D))
    Z, top = enrichment(mesh, order, D)
    none = BitArray(mesh.ne)
    none.Clear()
    fes = (on(MatrixValued(L2(mesh, order=order), dim=2)) * Z * restrict(L2(mesh, order=order + 2), none)
           * on(VectorL2(mesh, order=order)) * on(L2(mesh, order=order))
           * on(FacetFESpace(mesh, order=order, dirichlet=dirichlet)) ** 2)
    # (restrict(F ** 2) would lose the Dirichlet dofs of F)
    if split:
        rho = on(L2(mesh, order=0))
        for d in range(rho.ndof):
            rho.SetCouplingType(d, COUPLING_TYPE.INTERFACE_DOF)
        fes = fes * on(L2(mesh, order=0)) * rho
    return fes, top


def free_dofs(fes, top, condense, free=None, first=0):
    """Free dofs without the unused p dofs of the component first+1."""
    free = fes.FreeDofs(coupling=condense) if free is None else free
    for i, d in enumerate(fes.Range(first + 1)):
        if not top[i]:
            free.Clear(d)
    return free


def solve_linear_system(lhs, rhs, gfu, freedofs, inverse=None):
    """poisson.solve_linear_system with the free dofs given; inverse(mat, freedofs): the
    inverse of the condensed matrix (default: a sparse factorization)."""
    lhs.Assemble()
    Ainv = lhs.mat.Inverse(freedofs) if inverse is None else inverse(lhs.mat, freedofs)
    rhs.Assemble()
    b = rhs.vec.CreateVector()
    b.data = rhs.vec - lhs.mat * gfu.vec
    Id = IdentityMatrix(lhs.mat.height)
    Ainv_full = (Id + lhs.harmonic_extension) @ (Ainv + lhs.inner_solve) @ (Id + lhs.harmonic_extension_trans)
    gfu.vec.data += lhs.harmonic_extension * gfu.vec
    gfu.vec.data += Ainv_full * b


def forms(fes, lam, mu, f, tau=1.0, D=None):
    """The HDG rows on the elements of D except the equation of uhat (condensed).
    Returns lhs, rhs, the trial functions (sP, p, -, u, gamma, uhat), the test
    function muh of uhat and sigma_hat n.

    If fes has the two P_0 spaces of spaces(split=True), the element mean of tr(sigma)
    is kept as the global unknown rho (pressure split of the MATLAB code,
    LinearElasticityMatrices.m, option pressure_split):

        row of tau:   ... + (xi, tr tau)_K = 0
        row of xi:    (tr sigma - rho, chi)_K = 0
        row of rho:   <uhat . n, psi>_dK - (rho, psi)_K / (2 (lam + mu)) = 0

    an exact reformulation (xi_h = 0): lam enters the local problems only through
    A, which stays bounded, and the global system through the 1/(lam + mu) diagonal."""
    trial, test = fes.TnT()
    lhs = BilinearForm(fes, condense=True)
    rhs = LinearForm(fes)
    sigmahat_n = hdg_rows(lhs, rhs, trial, test, lam, mu, f, tau, D)
    return lhs, rhs, tuple(trial[:6]), test[5], sigmahat_n


def hdg_rows(lhs, rhs, trial, test, lam, mu, f, tau=1.0, D=None):
    """Add the rows of forms for one set of trial and test functions (sP, p, -, u,
    gamma, uhat[, xi, rho]) to lhs, rhs (several sets may share one form); returns
    sigma_hat n."""
    n = specialcf.normal(2)
    dV = dx(definedonelements=D)
    dS = dx(element_boundary=True, definedonelements=D)
    (sP, p, _, u, gam, uhat), (tP, q, _, v, eta, muh) = trial[:6], test[:6]
    sigma, tau_ = stress(sP, p), stress(tP, q)
    sigmahat_n = sigma * n - tau * mu * (u - uhat)

    lhs += (InnerProduct(compliance(sigma, lam, mu), tau_) - InnerProduct(grad(u), tau_)
            + InnerProduct(skew(gam), tau_)) * dV + (u - uhat) * (tau_ * n) * dS
    lhs += InnerProduct(sigma, skew(eta)) * dV
    lhs += InnerProduct(sigma, grad(v)) * dV - sigmahat_n * v * dS
    rhs += f * v * dV
    if len(trial) == 8:
        (xi, rho), (chi, psi) = trial[6:], test[6:]
        lhs += (xi * Trace(tau_) + (Trace(sigma) - rho) * chi) * dV
        lhs += (uhat * n) * psi * dS - rho * psi / (2 * (lam + mu)) * dV
    return sigmahat_n


def finish(lhs, rhs, fes, top, uhat_bc, mesh, order, lam, mu, D=None):
    """Solve with uhat = uhat_bc on the Dirichlet faces; returns
    (sigma_h, u_h, gamma_h, uhat_h, ustar_h)."""
    gfu = GridFunction(fes)
    if uhat_bc is not None:
        gfu.components[5].Set(*uhat_bc)
    solve_linear_system(lhs, rhs, gfu, free_dofs(fes, top, lhs.condense))
    sP_h, p_h, _, u_h, gam_h, uhat_h = gfu.components[:6]
    u_h.gfu = gfu                                       # the whole solution, for study_io
    sigma_h = stress(sP_h, p_h)
    return sigma_h, u_h, gam_h, uhat_h, postprocess(mesh, order, lam, mu, sigma_h, u_h, gam_h, D)


def postprocess(mesh, order, lam, mu, sigma_h, u_h, gamma_h, elements=None):
    """u*_h in P_{k+1}(K)^2 (Cockburn-Shi 2013, (2.9)): on each K,
    (grad u*, grad w) = (A sigma_h + S(gamma_h), grad w) for w in P_{k+1}^2,
    (u*, 1) = (u_h, 1); the mean value enters through a P_0 multiplier."""
    on = (lambda V: V) if elements is None else (lambda V: restrict(V, elements))
    fes = on(VectorL2(mesh, order=order + 1)) * on(VectorL2(mesh, order=0))
    (us, m), (w, r) = fes.TnT()
    dV = dx(definedonelements=elements)
    a = BilinearForm(InnerProduct(grad(us), grad(w)) * dV + (m * w + us * r) * dV)
    b = LinearForm(InnerProduct(compliance(sigma_h, lam, mu) + skew(gamma_h), grad(w)) * dV
                   + u_h * r * dV)
    a.Assemble()
    b.Assemble()
    gf = GridFunction(fes)
    gf.vec.data = a.mat.Inverse(inverse="umfpack") * b.vec
    return gf.components[0]


def solve(mesh, order, lam, mu, f, g, gN=None, dirichlet=".*", tau=1.0, split=False):
    fes, top = spaces(mesh, order, dirichlet, split=split)
    lhs, rhs, (*_, uhat), muh, sigmahat_n = forms(fes, lam, mu, f, tau)
    lhs += sigmahat_n * muh * dx(element_boundary=True)
    if gN is not None:      # muh vanishes on the Dirichlet faces
        rhs += gN * muh * ds(skeleton=True)
    return finish(lhs, rhs, fes, top, (g, BND, mesh.Boundaries(dirichlet)), mesh, order, lam, mu)


def lame(E, nu):
    """Plane-strain Lame parameters (lam, mu) from Young's modulus and Poisson ratio."""
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def manufactured(u, lam, mu):
    """Exact sigma, gamma and load f for a displacement CF u."""
    gu = CF((u[0].Diff(x), u[0].Diff(y), u[1].Diff(x), u[1].Diff(y)), dims=(2, 2))
    eps = 0.5 * (gu + gu.trans)
    sigma = 2 * mu * eps + lam * Trace(eps) * Id(2)
    gamma = 0.5 * (u[0].Diff(y) - u[1].Diff(x))
    f = -CF((sigma[0, 0].Diff(x) + sigma[0, 1].Diff(y),
             sigma[1, 0].Diff(x) + sigma[1, 1].Diff(y)))
    return sigma, gamma, f


def errors(mesh, sol, u, sigma, gamma, D=None):
    """L2(D_h) errors of sigma, u, gamma; |u - uhat|_h, |P_M u - uhat|_h over the element
    boundaries of D_h (P_M: L2 projection on the faces); |u - u*|; |P_W u - u_h| and
    |P_{k-1}(u - u_h)| (P_W, P_{k-1}: elementwise L2 projections on P_k^2, P_{k-1}^2)."""
    sigma_h, u_h, gamma_h, uhat_h, ustar_h = sol
    dV = dx(definedonelements=D, bonus_intorder=4)
    dS = dx(element_boundary=True, definedonelements=D, bonus_intorder=4)
    l2 = lambda a: sqrt(Integrate(InnerProduct(a, a) * dV, mesh))
    h = specialcf.mesh_size
    hnorm = lambda a: sqrt(Integrate(h * InnerProduct(a, a) * dS, mesh))
    M = uhat_h.space                                    # P_M u: L2 projection on the faces
    p, q = M.TnT()
    mass, load = BilinearForm(p * q * dS).Assemble(), LinearForm(u * q * dS).Assemble()
    Pu = GridFunction(M)
    Pu.vec.data = mass.mat.Inverse(inverse="sparsecholesky") * load.vec
    k = u_h.space.globalorder

    def project(cf, order):
        P = GridFunction(VectorL2(mesh, order=order))
        P.Set(cf, definedonelements=D, bonus_intorder=4)
        return P
    return (l2(sigma_h - sigma), l2(u_h - u), l2(gamma_h - gamma), hnorm(u - uhat_h),
            hnorm(Pu - uhat_h), l2(u - ustar_h), l2(project(u, k) - u_h),
            l2(project(u - u_h, k - 1)))


def bounded_data(lam):
    """Displacement curl(psi) + u_1/lam: sigma and f stay bounded as lam -> infinity."""
    return CF((sin(pi * x) * cos(pi * y) * (1 + 1 / lam), cos(pi * x) * sin(pi * y) * (-1 + 1 / lam)))


COLUMNS = ("|s-sh|", "|u-uh|", "|g-gh|", "|u-uh^|h", "|Pu-uh^|h", "|u-u*|", "|PWu-uh|", "|Pk-1e|")
HEADER = f"  {'h':>7}" + "".join(f"  {c:>9}  eoc " for c in COLUMNS)


def eoc_line(h, e, prev):
    """One table row: the errors e at mesh size h, with EOCs against prev = (h, e)."""
    line = f"  {h:7.4f}"
    for i, ei in enumerate(e):
        r = f"{log2(prev[1][i] / ei) / log2(prev[0] / h):4.2f}" if prev else "  - "
        line += f"  {ei:9.3e}  {r}"
    return line


def study(meshes, order, E=1.0, nu=0.3, dirichlet=".*", traction=False, bounded=False, record=None):
    """Print the EOC table for the manufactured solution on a list of (h, mesh); record:
    a study_io.Record that collects the levels."""
    lam, mu = lame(E, nu)
    u = CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y)))
    if bounded:
        u = bounded_data(lam)
    sigma, gamma, f = manufactured(u, lam, mu)
    gN = sigma * specialcf.normal(2) if traction else None
    print(f"k = {order}, E = {E}, nu = {nu}, dirichlet = {dirichlet!r}")
    print(HEADER)
    prev = None
    rows = []
    for h, mesh in meshes:
        sol = solve(mesh, order, lam, mu, f, u, gN, dirichlet)
        e = errors(mesh, sol, u, sigma, gamma)
        print(eoc_line(h, e, prev))
        if record is not None:
            record.add(h, mesh, e, sol[1].gfu)
        prev = (h, e)
        rows.append(e)
    return rows


CASES = {"dirichlet": {}, "traction": dict(dirichlet="left|bottom", traction=True),
         "bounded": dict(bounded=True)}


if __name__ == "__main__":
    from netgen.geom2d import unit_square
    from study_io import Record, parse, parser, run_defaults, save

    p = parser(__doc__, list(CASES))
    p.add_argument("--E", type=float, default=1.0, help="Young's modulus (default 1)")
    p.add_argument("--nu", type=float, help="Poisson ratio (default 0.3; bounded: 0.49999999)")
    a = parse(p)
    if a.k is None:
        # one process per case; peak per case estimated below 2 GB (see cosmos-pending.md)
        run_defaults(__file__, [["--case", c, "--k", k, "--h", *hs] for k, hs in
                                ((1, (0.2, 0.1, 0.05, 0.025, 0.0125)), (2, (0.2, 0.1, 0.05, 0.025)))
                                for c in CASES])
    else:
        nu = a.nu if a.nu is not None else (0.49999999 if a.case == "bounded" else 0.3)
        record = Record(COLUMNS)
        meshes = [(h, Mesh(unit_square.GenerateMesh(maxh=h))) for h in a.h]
        study(meshes, a.k, E=a.E, nu=nu, record=record, **CASES[a.case])
        if a.out:
            save(a.out, record, dict(driver="elasticity", case=a.case, k=a.k, E=a.E, nu=nu, tau=1.0))

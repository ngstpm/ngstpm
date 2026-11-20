"""The proxy verbs Extrapolate, PathIntegral and Transfer (D7, D8, D11, D15, D17).

Ordinary NGSolve assembly only (no interface binding). P1-P10 use a
stand-in path with the attributes of a TransferPath, built from analytic
ell, sigma on the disk; P11 solves a BVP with transfer_paths.
"""

from math import log2
from types import SimpleNamespace

import numpy as np
import pytest
from netgen.geom2d import SplineGeometry
from ngsolve import (
    CF, BilinearForm, BitArray, Compress, FacetFESpace, GridFunction, InnerProduct, Integrate, L2, LinearForm,
    Mesh, Norm, VOL, VectorL2, cos, div, dx, grad, pi, sin, specialcf, sqrt, x, y,
)

import ngstpm
from ngstpm._ngstpm import SideEval


def background_mesh(maxh, L=1.3):
    geo = SplineGeometry()
    geo.AddRectangle((-L, -L), (L, L))
    return Mesh(geo.GenerateMesh(maxh=maxh))


def disk_elements(mesh):
    """Elements with all vertices in the unit disk."""
    D = BitArray(mesh.ne)
    D.Clear()
    for el in mesh.Elements(VOL):
        D[el.nr] = all(np.hypot(*mesh.vertices[v.nr].point) < 1 for v in el.vertices)
    return D


def compress(fes, D):
    active = BitArray(fes.ndof)
    active.Clear()
    for el in fes.mesh.Elements(VOL):
        if D[el.nr]:
            for d in fes.GetDofNrs(el):
                active[d] = True
    return Compress(fes, active_dofs=active)


ELL = 1 - sqrt(x * x + y * y)                                     # disk: closest-point paths
SIGMA = CF((x, y)) / sqrt(x * x + y * y)


def stand_in_path(mesh, D):
    side = lambda cf: SideEval(cf, None, mesh, D)
    return SimpleNamespace(mesh=mesh, elements=D, ell=side(ELL), sigma=side(SIGMA),
                           target=side(CF((x, y)) + ELL * SIGMA))


def matrix(fes, integrand, measure):
    a = BilinearForm(fes)
    a += integrand * measure
    a.Assemble()
    return a.mat.ToDense().NumPy()


def setup(k=2):
    mesh = background_mesh(0.3)
    D = disk_elements(mesh)
    fes = compress(VectorL2(mesh, order=k), D) * compress(L2(mesh, order=k), D)
    dG = dx(element_boundary=True, definedonelements=D)
    return mesh, D, fes, dG


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_extrapolate_reproduces_polynomials(k):
    """P1. q.Extrapolate(x + d) is the element polynomial at x + d, outside K too."""
    mesh = background_mesh(0.3)
    fes = VectorL2(mesh, order=k) * L2(mesh, order=k)
    (q, u), (r, v) = fes.TnT()
    d = CF((0.37 * y + 0.1, -0.2 * x * x + 0.05))           # O(h), leaves K
    target = CF((x, y)) + d
    p = lambda X, Y: CF((X**k - 2 * X * Y**(k - 1) + 1, Y**k + X))
    dS = dx(element_boundary=True, bonus_intorder=2 * k + 4)
    a = BilinearForm(fes)
    a += (q.Extrapolate(target) * r + u.Extrapolate(target=target) * v) * dS
    a.Assemble()
    f = LinearForm(fes)
    f += (p(x + d[0], y + d[1]) * r + p(x + d[0], y + d[1])[0] * v) * dS
    f.Assemble()
    gf = GridFunction(fes)
    gf.components[0].Set(p(x, y))
    gf.components[1].Set(p(x, y)[0])
    res = f.vec.CreateVector()
    res.data = a.mat * gf.vec - f.vec
    assert Norm(res) < 1e-11 * Norm(f.vec)


def test_extrapolate_at_start_is_the_proxy():
    """P2. Extrapolate() and Extrapolate(target=x) are the trial function itself
    (to roundoff: the plain proxy is assembled with SIMD, Extrapolate without)."""
    mesh, D, fes, dG = setup()
    (q, u), (r, v) = fes.TnT()
    A = matrix(fes, (q * r + u * v), dG)
    B = matrix(fes, q.Extrapolate() * r + u.Extrapolate() * v, dG)
    C = matrix(fes, q.Extrapolate(target=CF((x, y))) * r + u.Extrapolate(CF((x, y))) * v, dG)
    assert abs(B - A).max() < 1e-14 * abs(A).max()
    assert abs(C - B).max() == 0.0


def test_test_functions_raise():
    """P3. Test functions are never extrapolated: all three verbs raise."""
    mesh, D, fes, dG = setup()
    (q, u), (r, v) = fes.TnT()
    path = stand_in_path(mesh, D)
    with pytest.raises(Exception, match="test functions"):
        v.Extrapolate()
    with pytest.raises(Exception, match="test functions"):
        r.PathIntegral(path)
    with pytest.raises(Exception, match="test functions"):
        v.Transfer(path, gradient=q)


def test_extrapolate_without_dofs_raises():
    """P4. Under an ordinary measure, an element without dofs of the space is an error (D17)."""
    mesh, D, fes, dG = setup()
    (q, u), (r, v) = fes.TnT()
    with pytest.raises(Exception, match="no dofs on element"):
        matrix(fes, u.Extrapolate() * v, dx)


def tpm_by_hand(q, k):
    """int_0^ell (E q)(x + s sigma) . sigma ds, (k+1)-point Gauss, with Extrapolate
    (the legacy q.tpm(ell, sigma, k))."""
    t, w = np.polynomial.legendre.leggauss(k + 1)
    X = CF((x, y))
    return sum(0.5 * wj * ELL * InnerProduct(q.Extrapolate(X + 0.5 * (tj + 1) * ELL * SIGMA), SIGMA)
               for tj, wj in zip(t, w))


def test_path_integral_matches_tpm():
    """P5. With a constant coefficient, G.PathIntegral(path) is the legacy q.tpm(ell, sigma, k)."""
    k = 2
    mesh, D, fes, dG = setup(k)
    (q, u), (r, v) = fes.TnT()
    path = stand_in_path(mesh, D)
    old = matrix(fes, tpm_by_hand(q, k) * v, dG)
    for G, scale in ((q, 1), (CF(2.5) * q, 2.5)):
        new = matrix(fes, G.PathIntegral(path) * v, dG)
        assert abs(new - scale * old).max() < 1e-13 * abs(old).max()
    # a matrix G: its rows are dotted with dl
    M = CF((q[0], q[1], 2 * q[0], 2 * q[1]), dims=(2, 2))
    rows = matrix(fes, M.PathIntegral(path) * r, dG)
    by_row = matrix(fes, tpm_by_hand(q, k) * (r[0] + 2 * r[1]), dG)
    assert abs(rows - by_row).max() < 1e-13 * abs(by_row).max()


def kappa(X, Y):
    return 2 + X * X + 0.5 * Y


def path_quadrature(q, n):
    """int q / kappa . dl along x -> x + ell sigma, n-point Gauss, written out by hand."""
    t, w = np.polynomial.legendre.leggauss(n)
    dl = ELL * SIGMA
    total = 0
    for tj, wj in zip(t, w):
        s = 0.5 * (tj + 1)
        total = total + 0.5 * wj / kappa(x + s * dl[0], y + s * dl[1]) * InnerProduct(q.Extrapolate(CF((x, y)) + s * dl), dl)
    return total


def test_path_integral_variable_coefficient():
    """P6. G = q/kappa(x, y): kappa is evaluated along the path (D8), with k+3 points."""
    k = 2
    mesh, D, fes, dG = setup(k)
    (q, u), (r, v) = fes.TnT()
    path = stand_in_path(mesh, D)
    hand = matrix(fes, path_quadrature(q, k + 3) * v, dG)
    new = matrix(fes, (q / kappa(x, y)).PathIntegral(path) * v, dG)
    assert abs(new - hand).max() < 1e-13 * abs(hand).max()
    at_x = matrix(fes, (q.PathIntegral(path) / kappa(x, y)) * v, dG)
    assert abs(at_x - hand).max() > 1e-3 * abs(hand).max()        # kappa at x is not the same


def test_path_integral_gridfunction_coefficient():
    """P7. A GridFunction in G is extrapolated along the path like the trial functions (D17)."""
    k = 2
    mesh, D, fes, dG = setup(k)
    (q, u), (r, v) = fes.TnT()
    path = stand_in_path(mesh, D)
    kappa_h = GridFunction(L2(mesh, order=2))
    kappa_h.Set(kappa(x, y))                                        # exact: kappa is quadratic
    exact = matrix(fes, (q / kappa(x, y)).PathIntegral(path) * v, dG)
    discrete = matrix(fes, (q / kappa_h).PathIntegral(path) * v, dG)
    assert abs(discrete - exact).max() < 1e-12 * abs(exact).max()


def test_path_integral_side_check():
    """P8. A trial function whose space lives on the other side raises (D11)."""
    mesh, D, fes, dG = setup()
    outer = ~D
    fes2 = compress(VectorL2(mesh, order=2), outer) * compress(L2(mesh, order=2), outer)
    (q2, u2), _ = fes2.TnT()
    path = stand_in_path(mesh, D)
    with pytest.raises(ValueError, match="different sides"):
        q2.PathIntegral(path)
    with pytest.raises(ValueError, match="different sides"):
        u2.Transfer(path, gradient=CF((0, 0)))


def test_transfer_is_extrapolate_plus_path_integral():
    """P9. u.Transfer(path, gradient=G) = u.Extrapolate(side=path) + G.PathIntegral(path)."""
    mesh, D, fes, dG = setup()
    (q, u), (r, v) = fes.TnT()
    path = stand_in_path(mesh, D)
    G = -q / kappa(x, y)
    A = matrix(fes, u.Transfer(path, gradient=G) * v, dG)
    B = matrix(fes, (u * v + G.PathIntegral(path) * v), dG)
    assert abs(A - B).max() < 1e-14 * abs(B).max()


def test_shifted_eval_is_extrapolate():
    """P10. ngsxfem-style shifted_eval(u, forth=d) is u.Extrapolate(x + d); trial only."""
    mesh = background_mesh(0.3)
    V = L2(mesh, order=2)
    u, v = V.TnT()
    d = CF((0.1 * y, 0.05))
    dS = dx(element_boundary=True)
    A = matrix(V, ngstpm.shifted_eval(u, forth=d) * v, dS)
    B = matrix(V, u.Extrapolate(CF((x, y)) + d) * v, dS)
    assert abs(A - B).max() == 0.0
    with pytest.raises(NotImplementedError):
        ngstpm.shifted_eval(u, back=d, forth=d)
    with pytest.raises(Exception, match="test functions"):
        ngstpm.shifted_eval(v, forth=d)


def solve_variable_kappa(maxh, k, frozen, tau=1.0):
    """-div(kappa grad u) = f on the unfitted disk, u = g on Gamma, mixed HDG;
    the Dirichlet row with kappa along the path (Transfer) or frozen at x.
    Returns ||u - u_h||_{L2(D_h)}."""
    from ngstpm import boundary_ds, interior_ds, transfer_paths, unfitted_domain
    disk = lambda X, Y: sqrt(X * X + Y * Y) - 1
    g = lambda X, Y: sin(pi * X) * cos(pi * Y)
    mesh = background_mesh(maxh)
    D = unfitted_domain(mesh, disk)[0]
    p = transfer_paths(mesh, disk, D)
    fes = compress(VectorL2(mesh, order=k), D) * compress(L2(mesh, order=k), D) \
        * compress(FacetFESpace(mesh, order=k), D)
    (q, u, uhat), (r, v, mu) = fes.TnT()
    n = specialcf.normal(2)
    dV = dx(definedonelements=D)
    dS = dx(element_boundary=True, definedonelements=D)
    dB = boundary_ds(mesh, D, bonus_intorder=max(6, 2 * k))
    K, uex = kappa(x, y), g(x, y)
    f = -((K * uex.Diff(x)).Diff(x) + (K * uex.Diff(y)).Diff(y))
    qhat_n = q * n + tau * (u - uhat)
    lhs, rhs = BilinearForm(fes, condense=True), LinearForm(fes)
    lhs += 1 / K * q * r * dV - u * div(r) * dV + uhat * (r * n) * dS
    lhs += -q * grad(v) * dV + qhat_n * v * dS
    rhs += f * v * dV
    lhs += qhat_n * mu * interior_ds(mesh, D)
    ub = uhat + (-q).PathIntegral(p) / K if frozen else uhat.Transfer(p, gradient=-q / K)
    lhs += ub * mu * dB
    rhs += g(*p.target) * mu * dB
    lhs.Assemble()
    rhs.Assemble()
    gfu = GridFunction(fes)
    rhs.vec.data += lhs.harmonic_extension_trans * rhs.vec
    gfu.vec.data = lhs.mat.Inverse(fes.FreeDofs(True)) * rhs.vec
    gfu.vec.data += lhs.harmonic_extension * gfu.vec
    gfu.vec.data += lhs.inner_solve * rhs.vec
    uh = gfu.components[1]
    return sqrt(Integrate((uh - uex) ** 2 * dx(definedonelements=D, bonus_intorder=4), mesh))


@pytest.mark.parametrize("k", [1, 2])
def test_variable_kappa_rate(k):
    """P11. Space-varying kappa (D8): with gradient=-q/kappa(x, y), kappa is evaluated
    along the path and u converges at order k+1 (measured 1.94, 2.94 for k = 1, 2).
    With kappa frozen at x the error is O(ell^2 |grad kappa|) = O(h^2): measured
    rate 1.13 for k = 2 (k = 1 is not affected)."""
    errs = [solve_variable_kappa(maxh, k, frozen=False) for maxh in (0.1, 0.05)]
    assert log2(errs[0] / errs[1]) > k + 0.8
    if k == 2:
        frozen = [solve_variable_kappa(maxh, k, frozen=True) for maxh in (0.1, 0.05)]
        assert log2(frozen[0] / frozen[1]) < 2.0

"""ngstpm: the Transfer Path Method (Cockburn-Solano) for NGSolve.

Importing ngstpm adds methods to trial functions (test functions are never
extrapolated) and CoefficientFunctions:

    u.Extrapolate(target)      u of the element of x evaluated at target(x), also
                               outside it (polynomial extrapolation); default: x
    G.PathIntegral(p)          int G . dl along the transfer path p, from x to
                               p.target; G linear in trial functions, its trial
                               functions, GridFunctions and x, y, z moved along
                               the path; k+1 Gauss points (+2 for coefficients)
    u.Transfer(p, gradient=G)  u.Extrapolate() + G.PathIntegral(p), G ~ grad u
    f.RegionIntegral(p)        on Gamma_h, the integrand whose integral over Gamma_h is
                               int f over the region swept by the paths p (between
                               Gamma_h and Gamma); f as in PathIntegral, or a plain CF

and provides the unfitted geometry: the interior subdomain D_h of {phi < 0},
the facet measures interior_ds(mesh, D) and boundary_ds(mesh, D) (Gamma_h,
seen from D), the TransferPath of D from Gamma_h to Gamma = {phi = 0}
(transfer_paths(mesh, phi, D)), and, for interface problems, the Interface
pair(p1, p2) of the paths of the subdomains of {phi < 0} and {phi > 0}, with
its measure I.ds(side=i). Instead of a level set, Gamma can be a
ParametricCurve, a BSplineCurve or a PointCloud (closest-point paths).
plot(u, p, extend=True) draws a scalar field on D_h and, along the paths, on
the band up to Gamma (matplotlib, 2D).
"""

import ngsolve  # noqa: F401  (registers the NGSolve types used by _ngstpm)
import numpy as np
from ngsolve import CF, Cof, CoefficientFunction, Id, InnerProduct, specialcf, x, y, z
from ngsolve.comp import ProxyFunction

from ._ngstpm import ComponentSpace, Extrapolate, Leaves, OwnElements, SideEval
from .curves import BSplineCurve, ParametricCurve, PointCloud
from .geometry import boundary_ds, boundary_facets, interior_ds, restrict, snapped_domains, unfitted_domain
from .interface import Interface, pair
from .paths import TransferPath, normal, transfer_paths
from .plot import plot

__all__ = ["BSplineCurve", "Interface", "ParametricCurve", "PointCloud", "TransferPath", "boundary_ds",
           "boundary_facets", "interior_ds", "normal", "pair", "plot", "restrict", "shifted_eval", "snapped_domains",
           "transfer_paths", "unfitted_domain"]


def shifted_eval(f, back=None, forth=None):
    """ngsxfem-compatible shifted evaluation, extended to trial functions.

    Same signature as xfem.shifted_eval: f is evaluated at z with
    x + back(z) = x + forth(x). For a trial function only back=None is
    supported, i.e. z = x + forth(x) (forth: any vector CF), and the result
    is f.Extrapolate(x + forth). A GridFunction is passed on to
    xfem.shifted_eval when ngsxfem is installed.
    """
    if isinstance(f, ProxyFunction):
        if back is not None:
            raise NotImplementedError("shifted_eval on trial functions: only back=None")
        if forth is None:
            return f.Extrapolate()
        X = CF((x, y, z)[:f.space.mesh.dim])
        return f.Extrapolate(X + CF(forth))
    try:
        from xfem import shifted_eval as xfem_shifted_eval
    except ImportError:
        raise TypeError("shifted_eval: GridFunctions need ngsxfem; "
                        "ngstpm handles trial/test functions") from None
    return xfem_shifted_eval(f, back=back, forth=forth)


def _extrapolate(self, target=None, *, side=None):
    """The trial function evaluated at target (default: the start of the path,
    x on the own element, x' on the partner), by polynomial extrapolation.

    The element is the integration element K if the trial function's space
    has all its dofs on K, otherwise the partner K' of an interface measure
    (under dx and other ordinary measures that is an error). side: a
    TransferPath; pins the own elements to side.elements instead (needed
    when one space covers both sides); or 1, 2 under an interface measure
    I.ds: own if the measure is on that side. Trial functions only.
    """
    target = None if target is None else CF(target)
    if isinstance(side, int):
        if side not in (1, 2):
            raise ValueError("Extrapolate: side is 1, 2 or a TransferPath")
        return Extrapolate(self, target, None, side)
    return Extrapolate(self, target, None if side is None else side.elements)


def _check_side(proxy, path):
    """Raise unless the space of the trial function has all its dofs on every
    element of the path, on the path's mesh. Returns that space."""
    space = ComponentSpace(proxy)
    if space.mesh is not path.mesh:
        raise ValueError("path and trial function on different meshes")
    missing = path.elements & ~OwnElements(proxy)
    if missing.NumSet():
        raise ValueError(f"path and trial function on different sides: the space has no dofs on "
                         f"{missing.NumSet()} elements of the path")
    return space


def _path_integral(self, path):
    """int G . dl along the straight path from its start to path.target, G = self.

    G is a CF linear in trial functions (vector of the mesh dimension, or a
    matrix with as many columns, then G dl is a vector). At each path Gauss
    point y_j, the trial functions and GridFunctions in G are extrapolated to
    y_j from the element of the start of the path, and x, y, z become y_j.
    Gauss-Legendre with k+1 points (k the highest order of the trial
    functions' spaces), plus 2 if G has non-constant coefficients.
    Linearity in the trial functions is not checked.
    """
    trial, test, gfs, coefficients = Leaves(self)
    if test:
        raise TypeError("PathIntegral: test functions are never extrapolated (trial functions only)")
    k = max((_check_side(q, path).globalorder for q in trial), default=0)
    D = path.mesh.dim
    start = SideEval(CF((x, y, z)[:D]), None, path.mesh, path.elements)   # x, or x' on the partner
    dl = path.target - start                                               # ell sigma; x' -> xbar on the partner
    if self.shape == (D,):
        project = InnerProduct
    elif len(self.shape) == 2 and self.shape[1] == D:
        project = lambda G, v: G * v
    else:
        raise ValueError(f"PathIntegral: G must be a vector of dimension {D} or a matrix with {D} columns")
    t, w = np.polynomial.legendre.leggauss(k + 1 + (2 if coefficients else 0))
    terms = []
    for tj, wj in zip(t, w):
        yj = path.target - (1 - 0.5 * (tj + 1)) * dl
        rep = {q: q.Extrapolate(yj, side=path) for q in trial}
        rep.update({g: SideEval(g, yj, path.mesh, path.elements) for g in gfs})
        rep.update({X: yj[i] for i, X in enumerate((x, y, z)[:D])})
        terms.append(0.5 * wj * project(self.Replace(rep), dl))
    return sum(terms[1:], terms[0])


def _region_integral(self, path, order=None):
    """On Gamma_h of the path's element set, sum_j w_j J(s_j) f(x + s_j d), f = self:
    integrated over Gamma_h (boundary_ds(mesh, path.elements), or I.ds on the
    path's own side) it gives int f over the region swept by the paths,
    {x + s d : x in Gamma_h, s in [0, 1]}, d = ell sigma.

    J(s) = d . Cof(I + s grad d) n, n the outward normal of the element K of x:
    the signed volume element of the map (x, s) -> x + s d (in 2D,
    d x (tau + s (grad d) tau), tau = (-n_y, n_x)). It is kept signed, so where
    the patches of staircase facets fold the contributions cancel.
    As in PathIntegral, trial functions and GridFunctions in f are extrapolated
    from the element of x, and x, y, z become x + s_j d; test functions raise.
    Gauss-Legendre with order + dim points on [0, 1] (exact for f of degree
    order along the path); order default: the highest order of the spaces of
    the trial functions and GridFunctions in f, plus 2 if f has non-constant
    coefficients (coordinates, GridFunctions, other CFs).
    Not for the partner side of an interface measure: there n and the facet are
    the own side's while d would be the partner's (this is not detected).
    """
    trial, test, gfs, coefficients = Leaves(self)
    if test:
        raise TypeError("RegionIntegral: test functions are never extrapolated (trial functions only)")
    if path.kind == "facet-normal":
        raise NotImplementedError("RegionIntegral: not available for 'facet-normal' paths: they can miss Gamma "
                                  "on unfitted meshes, and the swept area is then O(h) wrong")
    if path.grad_d is None:
        raise NotImplementedError(f"RegionIntegral: not available for {path.kind!r} paths")
    if order is None:
        order = max([_check_side(q, path).globalorder for q in trial]
                    + [g.space.globalorder for g in gfs], default=0) + (2 if coefficients else 0)
    else:
        for q in trial:
            _check_side(q, path)
    D = path.mesh.dim
    start = SideEval(CF((x, y, z)[:D]), None, path.mesh, path.elements)
    d = path.ell * path.sigma
    n = specialcf.normal(D)
    t, w = np.polynomial.legendre.leggauss(order + D)
    terms = []
    for tj, wj in zip(t, w):
        s = 0.5 * (tj + 1)
        yj = start + s * d
        rep = {q: q.Extrapolate(yj, side=path) for q in trial}
        rep.update({g: SideEval(g, yj, path.mesh, path.elements) for g in gfs})
        rep.update({X: yj[i] for i, X in enumerate((x, y, z)[:D])})
        J = InnerProduct(d, Cof(Id(D) + s * path.grad_d) * n)
        terms.append(0.5 * wj * J * self.Replace(rep))
    return sum(terms[1:], terms[0])


def _transfer(self, path, *, gradient):
    """The trial function moved from the start of the path to path.target:
    self.Extrapolate(side=path) + gradient.PathIntegral(path), gradient ~ grad u."""
    start = self.Extrapolate(side=path)
    _check_side(self, path)
    return start + gradient.PathIntegral(path)


ProxyFunction.Extrapolate = _extrapolate
ProxyFunction.Transfer = _transfer
CoefficientFunction.PathIntegral = _path_integral
CoefficientFunction.RegionIntegral = _region_integral

from . import measure  # noqa: E402,F401  (I.ds; BilinearForm += splits interface integrals)
from . import twomesh  # noqa: E402,F401  (I.Space, I.BilinearForm, ...: two meshes)

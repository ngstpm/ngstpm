"""plot(u, p): draw a discrete scalar field on D_h with matplotlib (2D), optionally
extended along the transfer paths over the band between Gamma_h and Gamma."""

import numpy as np
from ngsolve import CF, InnerProduct, VOL, x, y

from ._ngstpm import Leaves, SideEval
from .geometry import whole_mesh

NUDGE = 1e-4        # facet points are moved this fraction towards the centroid of their element


def plot(u, paths=None, *, D=None, extend=False, gradient=None, ax=None, nsub=4, nband=(6, 8),
         edges=True, gamma=True, **kw):
    """Draw the scalar field u on the elements of D (2D, matplotlib); returns the Axes.

    u: scalar CF of GridFunctions (uh, qh[0], Norm(qh), uh - uex, ...).
    paths: the TransferPath of D; needed for extend and gamma. D defaults to
      paths.elements, else the whole mesh.
    extend: also draw u on the band between Gamma_h and Gamma: a point
      y = x + s (xbar - x) of the path of x in Gamma_h, with K the element of x, gets
        u_h|K(x) + int_x^y G . dl     if gradient=G (G ~ grad u, e.g. -qh), or
        u_h|K(y)                      (polynomial extrapolation) otherwise.
    ax: Axes to draw on (default: a new figure).
    nsub: sub-triangles per element edge; nband: samples per facet of Gamma_h
      (along it, along the paths).
    edges: draw the mesh; gamma: draw Gamma (the ends of the paths, needs paths).
    kw: passed to tripcolor (cmap, vmin, vmax, ...); shading defaults to "gouraud".
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if u.dim != 1:
        raise TypeError(f"plot: u must be scalar (dimension {u.dim}); pass e.g. qh[0] or Norm(qh)")
    if extend and paths is None:
        raise ValueError("plot: extend=True needs the TransferPath paths")
    if gradient is not None and not extend:
        raise ValueError("plot: gradient is only used with extend=True")
    mesh = _mesh(u, paths)
    if mesh.dim != 2:
        raise NotImplementedError("plot: 2D meshes only")
    D = (paths.elements if paths is not None else whole_mesh(mesh)) if D is None else D

    pieces = [_sample(mesh, u, D, nsub)]
    if extend or (gamma and paths is not None):
        band, ends = _band(mesh, u, paths, gradient if extend else None, *nband)
        if extend:
            pieces.append(band)
    if ax is None:
        ax = plt.subplots()[1]
    kw.setdefault("shading", "gouraud")
    values = np.concatenate([U for _, U in pieces])
    kw.setdefault("vmin", values.min())
    kw.setdefault("vmax", values.max())
    for tri, U in pieces:
        ax.tripcolor(tri, U, **kw)
    if edges:
        V = np.array([v.point for v in mesh.vertices])
        T = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(VOL)])
        ax.triplot(V[:, 0], V[:, 1], T, color="lightgray", lw=0.4)
    if gamma and paths is not None:
        ax.add_collection(LineCollection(np.stack([ends[:, :-1], ends[:, 1:]], axis=2).reshape(-1, 2, 2),
                                         colors="k", linewidths=1.2))
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def _mesh(u, paths):
    if paths is not None:
        return paths.mesh
    gfs = Leaves(u)[2]
    if not gfs:
        raise ValueError("plot: u has no GridFunction; pass the TransferPath paths")
    return gfs[0].space.mesh


def _sample(mesh, u, D, nsub):
    """u on a sub-triangulation of each element of D: (Triangulation, values)."""
    import matplotlib.tri as mtri

    ref = [(i / nsub, j / nsub) for j in range(nsub + 1) for i in range(nsub + 1 - j)]
    idx = {p: n for n, p in enumerate(ref)}
    tris = []
    for j in range(nsub):
        for i in range(nsub - j):
            a, b, c = idx[i / nsub, j / nsub], idx[(i + 1) / nsub, j / nsub], idx[i / nsub, (j + 1) / nsub]
            tris.append((a, b, c))
            if i + j < nsub - 1:
                tris.append((b, idx[(i + 1) / nsub, (j + 1) / nsub], c))
    lam = np.array([(1 - s - t, s, t) for s, t in ref])
    lam = NUDGE / 3 + (1 - NUDGE) * lam                # strictly inside: the element is unambiguous
    P, T = [], []
    for el in mesh.Elements(VOL):
        if D[el.nr]:
            T.extend(np.array(tris) + len(P) * len(ref))
            P.append(lam @ np.array([mesh.vertices[v.nr].point for v in el.vertices]))
    if not P:
        raise ValueError("plot: D has no elements")
    P = np.concatenate(P)
    return mtri.Triangulation(P[:, 0], P[:, 1], np.array(T)), _eval(u, mesh, P)


def _band(mesh, u, paths, gradient, nt, ns):
    """u on the band, sampled on an nt x ns grid per facet of Gamma_h (along it, along the
    paths): (Triangulation, values), and the path ends, shape (facets, nt, 2)."""
    import matplotlib.tri as mtri

    D = paths.elements
    X0 = []
    for f in mesh.facets:
        if paths.facets[f.nr]:
            el = mesh[[e for e in f.elements if D[e.nr]][0]]
            c = np.mean([mesh.vertices[v.nr].point for v in el.vertices], axis=0)
            A, B = (np.array(mesh.vertices[v.nr].point) for v in f.vertices)
            t = np.linspace(0, 1, nt)[:, None]
            X0.append((1 - t) * A + t * B)
            X0[-1] += NUDGE * (c - X0[-1])                 # into the element of D: the start of the path
    X0 = np.concatenate(X0)
    start = CF((x, y))
    dl = paths.target - start
    ends = _eval(paths.target, mesh, X0)
    if gradient is not None:
        if gradient.dim != 2:
            raise TypeError(f"plot: gradient must be a vector of dimension 2, not {gradient.dim}")
        order = max((g.space.globalorder for g in Leaves(gradient)[2]), default=0)
        tg, wg = np.polynomial.legendre.leggauss(order + 2)
    s_all = np.linspace(0, 1, ns)
    U = np.empty((len(X0), ns))
    for j, s in enumerate(s_all):
        if gradient is None:
            cf = SideEval(u, start + s * dl, mesh, D)
        else:
            cf = u + sum(0.5 * s * w * InnerProduct(SideEval(gradient, start + 0.5 * s * (t + 1) * dl, mesh, D), dl)
                         for t, w in zip(tg, wg))
        U[:, j] = _eval(cf, mesh, X0)
    P = X0[:, None, :] + s_all[None, :, None] * (ends - X0)[:, None, :]
    quads = [(i * ns + j, (i + 1) * ns + j, (i + 1) * ns + j + 1, i * ns + j + 1)
             for i in range(nt - 1) for j in range(ns - 1)]
    tris = np.array([(a, b, c) for a, b, c, d in quads] + [(a, c, d) for a, b, c, d in quads])
    nf = len(X0) // nt
    T = np.concatenate([tris + k * nt * ns for k in range(nf)])
    P = P.reshape(-1, 2)
    return (mtri.Triangulation(P[:, 0], P[:, 1], T), U.ravel()), ends.reshape(nf, nt, 2)


def _eval(cf, mesh, P):
    """cf at the points P (n x 2), each in the element that contains it."""
    vals = np.asarray(cf(mesh(P[:, 0], P[:, 1]))).reshape(len(P), cf.dim)
    return vals[:, 0] if cf.dim == 1 else vals

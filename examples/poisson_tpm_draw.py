"""Draw the solution of poisson_tpm.py on each of its geometries (one mesh, one k).

u_h lives on the interior subdomain D_h only; the cut elements between
Gamma_h and Gamma are left blank, and Gamma = {phi = 0} is drawn on top.
Needs matplotlib (pip install matplotlib). Writes poisson_tpm.png and opens a
window.
"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from ngsolve import *

from poisson_tpm import GEOMETRIES, background_mesh, solve

k, maxh = 2, 0.1
nsub = 4                                   # sub-triangles per element edge (u_h has degree k)


def sample(mesh, uh, elements, nsub):
    """u_h on a sub-triangulation of each element of the set (2D): points, triangles, values."""
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
    lam = 1e-9 / 3 + (1 - 1e-9) * lam                  # strictly inside: the element is unambiguous
    P, T = [], []
    for el in mesh.Elements(VOL):
        if not elements[el.nr]:
            continue
        V = np.array([mesh.vertices[v.nr].point for v in el.vertices])
        T.extend(np.array(tris) + len(P) * len(ref))
        P.append(lam @ V)
    P = np.concatenate(P)
    U = uh(mesh(P[:, 0], P[:, 1])).ravel()
    return mtri.Triangulation(P[:, 0], P[:, 1], np.array(T)), U


g = lambda X, Y: sin(pi * X) * cos(pi * Y)
f = 2 * pi**2 * g(x, y)
mesh = background_mesh(maxh)
V = np.array([v.point for v in mesh.vertices])
T = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(VOL)])
L = np.linspace(-1.3, 1.3, 400)
X, Y = np.meshgrid(L, L)
pts = mesh(X.ravel(), Y.ravel())

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, (name, phi) in zip(axes.flat, GEOMETRIES.items()):
    (qh, uh, _), inner = solve(mesh, k, f, g, phi)
    tri, U = sample(mesh, uh, inner, nsub)
    PHI = phi(x, y)(pts).reshape(X.shape)
    ax.tripcolor(tri, U, cmap="jet", shading="gouraud", vmin=-1, vmax=1)
    ax.triplot(V[:, 0], V[:, 1], T, color="lightgray", lw=0.4)
    ax.contour(X, Y, PHI, levels=[0], colors="k", linewidths=1.2)
    ax.set_title(name)
    ax.set_aspect("equal")
    ax.axis("off")
fig.savefig("poisson_tpm.png", dpi=200, bbox_inches="tight")
plt.show()

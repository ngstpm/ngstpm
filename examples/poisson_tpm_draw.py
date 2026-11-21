"""Draw the solution of poisson_tpm.py on each of its geometries (one mesh, one k)
with ngstpm.plot.

u_h lives on the interior subdomain D_h only. Two figures:

- poisson_tpm.png: u_h on D_h; the cut elements between Gamma_h and Gamma are
  left blank, and Gamma (the ends of the transfer paths) is drawn on top;
- poisson_tpm_reconstructed.png: the blank band filled from the transfer paths
  (extend=True). A point y of the band lies on the path x -> xbar of a point x
  of Gamma_h, with K the element of x, and

      u(y) ~ u_h|K(x) - int_x^y (E q_h|K) . dl      (q = -grad u, gradient=-q_h),

  the transfer condition of the method stopped at y.

Needs matplotlib (pip install matplotlib). Prints the error of u_h on D_h,
writes the two figures and opens them.
"""

import matplotlib.pyplot as plt
from ngsolve import *

from ngstpm import plot, transfer_paths
from poisson_tpm import GEOMETRIES, background_mesh, solve

k, maxh = 2, 0.1

g = lambda X, Y: sin(pi * X) * cos(pi * Y)
f = 2 * pi**2 * g(x, y)
mesh = background_mesh(maxh)

figs = [plt.subplots(2, 2, figsize=(10, 10)) for _ in range(2)]
for i, (name, phi) in enumerate(GEOMETRIES.items()):
    (qh, uh, _), inner = solve(mesh, k, f, g, phi)
    p = transfer_paths(mesh, phi, inner, order=k + 4)              # the paths solve() uses
    err = sqrt(Integrate((uh - g(x, y))**2 * dx(definedonelements=inner, bonus_intorder=6), mesh))
    print(f"{name:20s} ||u - u_h||_L2(D_h) = {err:.2e}")
    for (fig, axes), extend in zip(figs, (False, True)):
        ax = plot(uh, p, extend=extend, gradient=-qh if extend else None, ax=axes.flat[i],
                  cmap="jet", vmin=-1, vmax=1)
        ax.set_title(name)
figs[0][0].savefig("poisson_tpm.png", dpi=200, bbox_inches="tight")
figs[1][0].savefig("poisson_tpm_reconstructed.png", dpi=200, bbox_inches="tight")
plt.show()

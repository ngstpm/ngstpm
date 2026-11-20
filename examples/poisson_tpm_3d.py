"""The Dirichlet problem of poisson_tpm.py in 3D: -Delta u = f in
Omega = {phi < 0}, u = g on Gamma, on a background mesh of tetrahedra.
The weak form is the one of poisson_tpm.py (solve is imported from there).
"""

from math import log2

from netgen.csg import CSGeometry, OrthoBrick, Pnt
from ngsolve import *

from poisson_tpm import solve


def background_mesh(maxh, L=1.3):
    geo = CSGeometry()
    geo.Add(OrthoBrick(Pnt(-L, -L, -L), Pnt(L, L, L)))
    return Mesh(geo.GenerateMesh(maxh=maxh))


GEOMETRIES = {
    "ball": lambda X, Y, Z: sqrt(X*X + Y*Y + Z*Z) - 1,
    "torus": lambda X, Y, Z: sqrt((sqrt(X*X + Y*Y) - 0.8)**2 + Z*Z) - 0.4,
}


if __name__ == "__main__":
    g = lambda X, Y, Z: sin(X) * cos(Y) * exp(Z / 2)
    uex = g(x, y, z)
    qex = -CF((uex.Diff(x), uex.Diff(y), uex.Diff(z)))
    f = 1.75 * uex

    for name, phi in GEOMETRIES.items():
        print(name)
        for k in (1, 2):
            print(f"  k = {k}")
            prev = None
            for maxh in (0.4, 0.2, 0.1):
                mesh = background_mesh(maxh)
                (qh, uh, _), inner = solve(mesh, k, f, g, phi)
                dV = dx(definedonelements=inner)
                eu = sqrt(Integrate((uh - uex) ** 2 * dV, mesh))
                eq = sqrt(Integrate((qh - qex) ** 2 * dV, mesh))
                if prev:
                    ru, rq = log2(prev[0] / eu), log2(prev[1] / eq)
                    print(f"    h={maxh:5.3f}  |u-uh|={eu:.3e} ({ru:.2f})  |q-qh|={eq:.3e} ({rq:.2f})")
                else:
                    print(f"    h={maxh:5.3f}  |u-uh|={eu:.3e}         |q-qh|={eq:.3e}")
                prev = (eu, eq)
